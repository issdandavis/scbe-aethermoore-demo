"""
Browser Agent - Agentic Web Automation with AetherAuth

Integrates open-source browser automation (browser-use, playwright) with
AetherAuth's geometric security for safe, context-aware web interactions.

Key Features:
- browser-use integration for AI-driven web automation
- Playwright backend for cross-browser support
- AetherAuth security gates for action validation
- Cost-based path blocking for dangerous operations
- Audit logging for compliance

Supported Frameworks:
- browser-use (https://github.com/browser-use/browser-use)
- Playwright (Chromium, Firefox, WebKit)
- Agent-Browser for context-efficient automation

Usage:
    from aetherauth.browser_agent import SecureBrowserAgent

    async with SecureBrowserAgent() as agent:
        # Safe browsing with geometric security
        result = await agent.execute("Search for Python tutorials")

References:
- https://github.com/browser-use/browser-use
- https://github.com/esinecan/agentic-ai-browser
- https://www.browserbase.com (Stagehand)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from pathlib import Path

# AetherAuth imports
try:
    from .context_capture import capture_context_vector, ContextVector
    from .geoseal_gate import AetherAuthGate, AccessResult, TrustRing
except ImportError:
    from context_capture import capture_context_vector, ContextVector
    from geoseal_gate import AetherAuthGate, AccessResult, TrustRing


class ActionRisk(Enum):
    """Risk levels for browser actions."""
    SAFE = "safe"           # Read-only, no side effects
    LOW = "low"             # Minor side effects (cookies, history)
    MEDIUM = "medium"       # Form submissions, clicks
    HIGH = "high"           # Payments, account changes
    CRITICAL = "critical"   # Irreversible actions


@dataclass
class BrowserAction:
    """A browser action with security metadata."""
    action_type: str  # navigate, click, type, extract, etc.
    target: str       # URL, selector, or description
    risk_level: ActionRisk = ActionRisk.SAFE
    requires_confirmation: bool = False
    audit_required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Result of a browser action."""
    success: bool
    action: BrowserAction
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0
    blocked: bool = False
    block_reason: Optional[str] = None


# Risk classification for common browser actions
ACTION_RISK_MAP = {
    # Safe actions (read-only)
    "navigate": ActionRisk.SAFE,
    "extract_text": ActionRisk.SAFE,
    "screenshot": ActionRisk.SAFE,
    "get_url": ActionRisk.SAFE,
    "get_title": ActionRisk.SAFE,
    "read_page": ActionRisk.SAFE,

    # Low risk (minor side effects)
    "scroll": ActionRisk.LOW,
    "hover": ActionRisk.LOW,
    "accept_cookies": ActionRisk.LOW,

    # Medium risk (form interactions)
    "click": ActionRisk.MEDIUM,
    "type": ActionRisk.MEDIUM,
    "select": ActionRisk.MEDIUM,
    "fill_form": ActionRisk.MEDIUM,
    "search": ActionRisk.MEDIUM,

    # High risk (account/payment actions)
    "submit_form": ActionRisk.HIGH,
    "login": ActionRisk.HIGH,
    "logout": ActionRisk.HIGH,
    "change_settings": ActionRisk.HIGH,

    # Critical risk (irreversible)
    "purchase": ActionRisk.CRITICAL,
    "delete_account": ActionRisk.CRITICAL,
    "transfer_funds": ActionRisk.CRITICAL,
    "send_message": ActionRisk.HIGH,
    "post_content": ActionRisk.HIGH,
}

# URL patterns that require elevated trust
SENSITIVE_DOMAINS = [
    "bank", "banking", "finance", "paypal", "stripe",
    "admin", "dashboard", "settings", "account",
    "login", "auth", "oauth", "signin",
]


class BrowserSecurityGate:
    """
    Security gate for browser actions using AetherAuth's geometric model.

    Actions are validated against the current context vector and trust ring.
    High-risk actions require inner ring access (CORE or OUTER).
    Critical actions are always blocked without explicit user confirmation.
    """

    def __init__(self, gate: Optional[AetherAuthGate] = None):
        self.gate = gate or AetherAuthGate()
        self.action_log: List[Dict] = []

        # Risk thresholds (hyperbolic distance from origin)
        self.thresholds = {
            ActionRisk.SAFE: 0.9,      # Allowed in WALL ring
            ActionRisk.LOW: 0.7,       # Allowed in OUTER ring
            ActionRisk.MEDIUM: 0.5,    # Requires OUTER ring
            ActionRisk.HIGH: 0.3,      # Requires CORE ring
            ActionRisk.CRITICAL: 0.1,  # Requires explicit confirmation
        }

    def validate_action(
        self,
        action: BrowserAction,
        context: ContextVector,
        access: AccessResult
    ) -> ActionResult:
        """
        Validate a browser action against security constraints.

        Returns ActionResult with blocked=True if action is denied.
        """
        start_time = time.time()

        # Check trust ring access
        if not access.allowed:
            return ActionResult(
                success=False,
                action=action,
                blocked=True,
                block_reason=f"Access denied: {access.reason}",
                duration_ms=(time.time() - start_time) * 1000
            )

        # Get risk threshold
        threshold = self.thresholds.get(action.risk_level, 0.5)

        # Check if action is allowed at current trust distance
        if access.distance > threshold:
            return ActionResult(
                success=False,
                action=action,
                blocked=True,
                block_reason=f"Action '{action.action_type}' requires trust distance <= {threshold}, got {access.distance:.3f}",
                duration_ms=(time.time() - start_time) * 1000
            )

        # Check for sensitive domain access
        if self._is_sensitive_target(action.target):
            if access.ring not in [TrustRing.CORE, TrustRing.OUTER]:
                return ActionResult(
                    success=False,
                    action=action,
                    blocked=True,
                    block_reason=f"Sensitive domain requires CORE or OUTER ring, got {access.ring.value}",
                    duration_ms=(time.time() - start_time) * 1000
                )

        # Critical actions always require confirmation
        if action.risk_level == ActionRisk.CRITICAL:
            return ActionResult(
                success=False,
                action=action,
                blocked=True,
                block_reason="Critical action requires explicit user confirmation",
                duration_ms=(time.time() - start_time) * 1000
            )

        # Log the action
        if action.audit_required:
            self._log_action(action, context, access, allowed=True)

        return ActionResult(
            success=True,
            action=action,
            duration_ms=(time.time() - start_time) * 1000
        )

    def _is_sensitive_target(self, target: str) -> bool:
        """Check if target URL/selector involves sensitive domains."""
        target_lower = target.lower()
        return any(domain in target_lower for domain in SENSITIVE_DOMAINS)

    def _log_action(
        self,
        action: BrowserAction,
        context: ContextVector,
        access: AccessResult,
        allowed: bool
    ):
        """Log action for audit trail."""
        self.action_log.append({
            "timestamp": time.time(),
            "action_type": action.action_type,
            "target": action.target[:100],  # Truncate for privacy
            "risk_level": action.risk_level.value,
            "trust_ring": access.ring.value,
            "trust_distance": access.distance,
            "allowed": allowed,
            "context_hash": hash(str(context.dimensions)),
        })


class SecureBrowserAgent:
    """
    AI-powered browser agent with AetherAuth security integration.

    Wraps browser-use or Playwright for secure web automation.
    All actions are validated through the geometric security gate.

    Example:
        async with SecureBrowserAgent() as agent:
            result = await agent.execute("Find flights to Paris")
            print(result.output)
    """

    def __init__(
        self,
        headless: bool = True,
        browser: str = "chromium",
        gate: Optional[AetherAuthGate] = None
    ):
        self.headless = headless
        self.browser_type = browser
        self.gate = gate or AetherAuthGate()
        self.security = BrowserSecurityGate(self.gate)

        # Browser instance (initialized on enter)
        self._browser = None
        self._page = None
        self._playwright = None

        # Session state
        self.context: Optional[ContextVector] = None
        self.access: Optional[AccessResult] = None

    async def __aenter__(self):
        """Initialize browser and authenticate."""
        # Capture context
        self.context = capture_context_vector(caller_name="browser_agent")
        self.access = self.gate.check_access(self.context)

        if not self.access.allowed:
            raise PermissionError(f"Browser agent access denied: {self.access.reason}")

        # Try to import playwright
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()

            if self.browser_type == "chromium":
                self._browser = await self._playwright.chromium.launch(headless=self.headless)
            elif self.browser_type == "firefox":
                self._browser = await self._playwright.firefox.launch(headless=self.headless)
            elif self.browser_type == "webkit":
                self._browser = await self._playwright.webkit.launch(headless=self.headless)

            self._page = await self._browser.new_page()

        except ImportError:
            print("Playwright not installed. Install with: pip install playwright")
            print("Then run: playwright install")
            raise

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup browser resources."""
        if self._page:
            await self._page.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def navigate(self, url: str) -> ActionResult:
        """Navigate to a URL with security validation."""
        action = BrowserAction(
            action_type="navigate",
            target=url,
            risk_level=ACTION_RISK_MAP.get("navigate", ActionRisk.SAFE)
        )

        # Validate action
        validation = self.security.validate_action(action, self.context, self.access)
        if validation.blocked:
            return validation

        # Execute navigation
        try:
            start_time = time.time()
            await self._page.goto(url, wait_until="domcontentloaded")
            duration = (time.time() - start_time) * 1000

            return ActionResult(
                success=True,
                action=action,
                output={"url": self._page.url, "title": await self._page.title()},
                duration_ms=duration
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                error=str(e)
            )

    async def extract_text(self, selector: str = "body") -> ActionResult:
        """Extract text content from page."""
        action = BrowserAction(
            action_type="extract_text",
            target=selector,
            risk_level=ActionRisk.SAFE
        )

        validation = self.security.validate_action(action, self.context, self.access)
        if validation.blocked:
            return validation

        try:
            start_time = time.time()
            text = await self._page.inner_text(selector)
            duration = (time.time() - start_time) * 1000

            return ActionResult(
                success=True,
                action=action,
                output=text[:10000],  # Truncate for safety
                duration_ms=duration
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                error=str(e)
            )

    async def click(self, selector: str) -> ActionResult:
        """Click an element with security validation."""
        action = BrowserAction(
            action_type="click",
            target=selector,
            risk_level=ActionRisk.MEDIUM
        )

        validation = self.security.validate_action(action, self.context, self.access)
        if validation.blocked:
            return validation

        try:
            start_time = time.time()
            await self._page.click(selector)
            duration = (time.time() - start_time) * 1000

            return ActionResult(
                success=True,
                action=action,
                duration_ms=duration
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                error=str(e)
            )

    async def type_text(self, selector: str, text: str) -> ActionResult:
        """Type text into an input field."""
        action = BrowserAction(
            action_type="type",
            target=selector,
            risk_level=ActionRisk.MEDIUM,
            metadata={"text_length": len(text)}
        )

        validation = self.security.validate_action(action, self.context, self.access)
        if validation.blocked:
            return validation

        try:
            start_time = time.time()
            await self._page.fill(selector, text)
            duration = (time.time() - start_time) * 1000

            return ActionResult(
                success=True,
                action=action,
                duration_ms=duration
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                error=str(e)
            )

    async def screenshot(self, path: Optional[str] = None) -> ActionResult:
        """Take a screenshot of the current page."""
        action = BrowserAction(
            action_type="screenshot",
            target=path or "screenshot.png",
            risk_level=ActionRisk.SAFE
        )

        validation = self.security.validate_action(action, self.context, self.access)
        if validation.blocked:
            return validation

        try:
            start_time = time.time()
            screenshot_bytes = await self._page.screenshot(path=path)
            duration = (time.time() - start_time) * 1000

            return ActionResult(
                success=True,
                action=action,
                output={"path": path, "size": len(screenshot_bytes) if screenshot_bytes else 0},
                duration_ms=duration
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                error=str(e)
            )

    async def execute(self, task: str) -> ActionResult:
        """
        Execute a natural language task using browser-use or similar.

        This is a placeholder for integration with browser-use:
        https://github.com/browser-use/browser-use

        Example:
            result = await agent.execute("Search for Python tutorials on Google")
        """
        # For now, parse simple commands
        task_lower = task.lower()

        if "navigate to" in task_lower or "go to" in task_lower:
            # Extract URL
            import re
            url_match = re.search(r'https?://\S+', task)
            if url_match:
                return await self.navigate(url_match.group())

        if "screenshot" in task_lower:
            return await self.screenshot()

        if "extract" in task_lower or "get text" in task_lower:
            return await self.extract_text()

        # For complex tasks, would integrate with browser-use
        return ActionResult(
            success=False,
            action=BrowserAction(action_type="execute", target=task),
            error="Complex task execution requires browser-use integration. Install: pip install browser-use"
        )

    def get_audit_log(self) -> List[Dict]:
        """Get the security audit log."""
        return self.security.action_log


# Governance tier browser permissions
GOVERNANCE_TIERS = {
    "ELEMENTARY": {
        "allowed_actions": ["navigate", "extract_text", "screenshot"],
        "max_risk": ActionRisk.SAFE,
        "description": "Read-only web access for education"
    },
    "SECONDARY": {
        "allowed_actions": ["navigate", "extract_text", "screenshot", "scroll", "search"],
        "max_risk": ActionRisk.LOW,
        "description": "Basic interaction for research"
    },
    "UNDERGRADUATE": {
        "allowed_actions": ["navigate", "extract_text", "screenshot", "scroll", "search", "click", "type"],
        "max_risk": ActionRisk.MEDIUM,
        "description": "Form interaction for projects"
    },
    "GRADUATE": {
        "allowed_actions": ["navigate", "extract_text", "screenshot", "scroll", "search", "click", "type", "fill_form", "submit_form"],
        "max_risk": ActionRisk.HIGH,
        "description": "Full interaction for research"
    },
    "ENTERPRISE": {
        "allowed_actions": ["*"],
        "max_risk": ActionRisk.HIGH,
        "description": "Business operations"
    },
    "GOVERNMENT": {
        "allowed_actions": ["*"],
        "max_risk": ActionRisk.CRITICAL,
        "requires_mfa": True,
        "description": "Classified operations with full audit"
    }
}


def get_tier_permissions(tier: str) -> Dict:
    """Get browser permissions for a governance tier."""
    return GOVERNANCE_TIERS.get(tier.upper(), GOVERNANCE_TIERS["ELEMENTARY"])


async def demo():
    """Demo the secure browser agent."""
    print("=" * 60)
    print("Secure Browser Agent Demo")
    print("=" * 60)

    print("\n1. Governance Tier Permissions")
    print("-" * 40)
    for tier, perms in GOVERNANCE_TIERS.items():
        print(f"  {tier}: max_risk={perms['max_risk'].value}")

    print("\n2. Action Risk Classification")
    print("-" * 40)
    for action, risk in list(ACTION_RISK_MAP.items())[:10]:
        print(f"  {action}: {risk.value}")

    print("\n3. Security Gate Validation")
    print("-" * 40)
    gate = AetherAuthGate()
    security = BrowserSecurityGate(gate)
    context = capture_context_vector(caller_name="demo")
    access = gate.check_access(context)

    print(f"  Context captured, trust ring: {access.ring.value}")
    print(f"  Trust distance: {access.distance:.3f}")

    # Test actions at different risk levels
    test_actions = [
        BrowserAction("navigate", "https://google.com", ActionRisk.SAFE),
        BrowserAction("click", "#search-button", ActionRisk.MEDIUM),
        BrowserAction("login", "https://bank.com", ActionRisk.HIGH),
        BrowserAction("purchase", "Buy Now button", ActionRisk.CRITICAL),
    ]

    for action in test_actions:
        result = security.validate_action(action, context, access)
        status = "ALLOWED" if not result.blocked else f"BLOCKED: {result.block_reason}"
        print(f"  {action.action_type} ({action.risk_level.value}): {status}")

    print("\n4. Integration Options")
    print("-" * 40)
    print("  - browser-use: pip install browser-use")
    print("  - Playwright:  pip install playwright && playwright install")
    print("  - Stagehand:   pip install stagehand (via browserbase)")

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(demo())
