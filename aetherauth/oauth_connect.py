"""
OAuth Connect - One-Click API Authorization

Click a button, authorize in browser, done.
No more copying keys.

Supported:
- Notion (OAuth 2.0)
- Google (OAuth 2.0)
- GitHub (OAuth 2.0)

Usage:
    python -m aetherauth.oauth_connect notion
    python -m aetherauth.oauth_connect google
"""

import os
import sys
import json
import webbrowser
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import threading
import time

# Try requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Install requests: pip install requests")


# OAuth configurations (services with OAuth support)
OAUTH_CONFIGS = {
    "notion": {
        "name": "Notion",
        "auth_url": "https://api.notion.com/v1/oauth/authorize",
        "token_url": "https://api.notion.com/v1/oauth/token",
        "scopes": [],
        "client_id_env": "NOTION_OAUTH_CLIENT_ID",
        "client_secret_env": "NOTION_OAUTH_CLIENT_SECRET",
        "setup_url": "https://www.notion.so/my-integrations",
        "instructions": """
To set up Notion OAuth:
1. Go to https://www.notion.so/my-integrations
2. Click "New integration"
3. Give it a name (e.g., "AetherAuth")
4. Set type to "Public" for OAuth
5. Add redirect URI: http://localhost:8765/callback
6. Copy the OAuth client ID and secret
""",
    },
    "google": {
        "name": "Google",
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "scopes": ["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/calendar.readonly"],
        "client_id_env": "GOOGLE_OAUTH_CLIENT_ID",
        "client_secret_env": "GOOGLE_OAUTH_CLIENT_SECRET",
        "setup_url": "https://console.cloud.google.com/apis/credentials",
        "instructions": """
To set up Google OAuth:
1. Go to https://console.cloud.google.com/apis/credentials
2. Create OAuth 2.0 Client ID (Web application)
3. Add redirect URI: http://localhost:8765/callback
4. Copy client ID and secret
""",
    },
    "github": {
        "name": "GitHub",
        "auth_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "scopes": ["repo", "read:user"],
        "client_id_env": "GITHUB_OAUTH_CLIENT_ID",
        "client_secret_env": "GITHUB_OAUTH_CLIENT_SECRET",
        "setup_url": "https://github.com/settings/developers",
        "instructions": """
To set up GitHub OAuth:
1. Go to https://github.com/settings/developers
2. Click "New OAuth App"
3. Set callback URL: http://localhost:8765/callback
4. Copy client ID and secret
""",
    },
    "dropbox": {
        "name": "Dropbox",
        "auth_url": "https://www.dropbox.com/oauth2/authorize",
        "token_url": "https://api.dropboxapi.com/oauth2/token",
        "scopes": [],
        "client_id_env": "DROPBOX_OAUTH_CLIENT_ID",
        "client_secret_env": "DROPBOX_OAUTH_CLIENT_SECRET",
        "setup_url": "https://www.dropbox.com/developers/apps",
        "instructions": """
To set up Dropbox OAuth:
1. Go to https://www.dropbox.com/developers/apps
2. Create app -> Scoped access -> Full Dropbox
3. Add redirect URI: http://localhost:8765/callback
4. Copy App key (client_id) and App secret
""",
    },
    "slack": {
        "name": "Slack",
        "auth_url": "https://slack.com/oauth/v2/authorize",
        "token_url": "https://slack.com/api/oauth.v2.access",
        "scopes": ["channels:read", "chat:write", "users:read"],
        "client_id_env": "SLACK_OAUTH_CLIENT_ID",
        "client_secret_env": "SLACK_OAUTH_CLIENT_SECRET",
        "setup_url": "https://api.slack.com/apps",
        "instructions": """
To set up Slack OAuth:
1. Go to https://api.slack.com/apps
2. Create New App -> From scratch
3. OAuth & Permissions -> Add redirect URL: http://localhost:8765/callback
4. Copy Client ID and Client Secret from Basic Information
""",
    },
    "linear": {
        "name": "Linear",
        "auth_url": "https://linear.app/oauth/authorize",
        "token_url": "https://api.linear.app/oauth/token",
        "scopes": ["read", "write"],
        "client_id_env": "LINEAR_OAUTH_CLIENT_ID",
        "client_secret_env": "LINEAR_OAUTH_CLIENT_SECRET",
        "setup_url": "https://linear.app/settings/api",
        "instructions": """
To set up Linear OAuth:
1. Go to https://linear.app/settings/api
2. Create new OAuth application
3. Add redirect URI: http://localhost:8765/callback
4. Copy Client ID and Client Secret
""",
    },
}

# API Key only services (no OAuth, just open the page)
APIKEY_SERVICES = {
    "openai": {
        "name": "OpenAI",
        "key_url": "https://platform.openai.com/api-keys",
        "env_var": "OPENAI_API_KEY",
        "prefix": "sk-",
    },
    "anthropic": {
        "name": "Anthropic (Claude)",
        "key_url": "https://console.anthropic.com/settings/keys",
        "env_var": "ANTHROPIC_API_KEY",
        "prefix": "sk-ant-",
    },
    "perplexity": {
        "name": "Perplexity",
        "key_url": "https://www.perplexity.ai/settings/api",
        "env_var": "PERPLEXITY_API_KEY",
        "prefix": "pplx-",
    },
    "xai": {
        "name": "xAI (Grok)",
        "key_url": "https://console.x.ai/",
        "env_var": "XAI_API_KEY",
        "prefix": "xai-",
    },
    "together": {
        "name": "Together AI",
        "key_url": "https://api.together.xyz/settings/api-keys",
        "env_var": "TOGETHER_API_KEY",
        "prefix": "",
    },
    "replicate": {
        "name": "Replicate",
        "key_url": "https://replicate.com/account/api-tokens",
        "env_var": "REPLICATE_API_TOKEN",
        "prefix": "r8_",
    },
}

CALLBACK_PORT = 8765
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/callback"


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handles the OAuth callback."""

    token = None
    error = None

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs

    def do_GET(self):
        if self.path.startswith("/callback"):
            # Parse query params
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)

            if "code" in params:
                OAuthCallbackHandler.token = params["code"][0]
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"""
                    <html><body style="font-family: system-ui; text-align: center; padding: 50px;">
                    <h1>Authorization Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                    </body></html>
                """)
            elif "error" in params:
                OAuthCallbackHandler.error = params.get("error_description", params["error"])[0]
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(f"""
                    <html><body style="font-family: system-ui; text-align: center; padding: 50px;">
                    <h1>Authorization Failed</h1>
                    <p>{OAuthCallbackHandler.error}</p>
                    </body></html>
                """.encode())
            else:
                self.send_response(400)
                self.end_headers()


def wait_for_callback(timeout=120):
    """Start server and wait for OAuth callback."""
    server = HTTPServer(("localhost", CALLBACK_PORT), OAuthCallbackHandler)
    server.timeout = timeout

    OAuthCallbackHandler.token = None
    OAuthCallbackHandler.error = None

    start = time.time()
    while time.time() - start < timeout:
        server.handle_request()
        if OAuthCallbackHandler.token or OAuthCallbackHandler.error:
            break

    server.server_close()
    return OAuthCallbackHandler.token, OAuthCallbackHandler.error


def exchange_code_for_token(provider: str, code: str, client_id: str, client_secret: str) -> dict:
    """Exchange authorization code for access token."""
    config = OAUTH_CONFIGS[provider]

    if provider == "notion":
        # Notion uses Basic auth for token exchange
        import base64
        auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/json",
        }
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
        }
        response = requests.post(config["token_url"], json=data, headers=headers)

    elif provider == "github":
        headers = {"Accept": "application/json"}
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": REDIRECT_URI,
        }
        response = requests.post(config["token_url"], data=data, headers=headers)

    else:  # Google and others
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        }
        response = requests.post(config["token_url"], data=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Token exchange failed: {response.status_code} - {response.text}")


def save_token(provider: str, token_data: dict):
    """Save token to .aether/tokens/"""
    tokens_dir = Path(".aether/tokens")
    tokens_dir.mkdir(parents=True, exist_ok=True)

    token_file = tokens_dir / f"{provider}.json"
    with open(token_file, "w") as f:
        json.dump(token_data, f, indent=2)

    print(f"Token saved to {token_file}")
    return token_file


def load_token(provider: str) -> dict:
    """Load saved token."""
    token_file = Path(f".aether/tokens/{provider}.json")
    if token_file.exists():
        with open(token_file) as f:
            return json.load(f)
    return None


def connect(provider: str):
    """Main OAuth flow."""
    if provider not in OAUTH_CONFIGS:
        print(f"Unknown provider: {provider}")
        print(f"Available: {list(OAUTH_CONFIGS.keys())}")
        return False

    config = OAUTH_CONFIGS[provider]
    print(f"\n{'='*60}")
    print(f"Connect to {config['name']}")
    print('='*60)

    # Check for client credentials
    client_id = os.getenv(config["client_id_env"])
    client_secret = os.getenv(config["client_secret_env"])

    if not client_id or not client_secret:
        print(f"\nMissing OAuth credentials!")
        print(config["instructions"])
        print(f"\nThen set environment variables:")
        print(f"  ${config['client_id_env']}=your_client_id")
        print(f"  ${config['client_secret_env']}=your_client_secret")

        # Offer to enter manually
        print("\nOr enter them now:")
        client_id = input(f"Client ID: ").strip()
        client_secret = input(f"Client Secret: ").strip()

        if not client_id or not client_secret:
            print("Cancelled.")
            return False

    # Build authorization URL
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
    }

    if config["scopes"]:
        params["scope"] = " ".join(config["scopes"])

    if provider == "notion":
        params["owner"] = "user"

    auth_url = f"{config['auth_url']}?{urllib.parse.urlencode(params)}"

    print(f"\nOpening browser for authorization...")
    print(f"(If browser doesn't open, visit: {auth_url[:80]}...)")

    # Open browser
    webbrowser.open(auth_url)

    # Wait for callback
    print("\nWaiting for authorization...")
    code, error = wait_for_callback()

    if error:
        print(f"\nAuthorization failed: {error}")
        return False

    if not code:
        print("\nTimeout waiting for authorization.")
        return False

    print("\nAuthorization received! Exchanging for token...")

    # Exchange code for token
    try:
        token_data = exchange_code_for_token(provider, code, client_id, client_secret)

        # Save token
        save_token(provider, token_data)

        # Show what we got
        access_token = token_data.get("access_token", "")
        print(f"\n{'='*60}")
        print(f"Connected to {config['name']}!")
        print('='*60)
        print(f"Access token: {access_token[:20]}..." if access_token else "Token saved")

        if provider == "notion":
            workspace = token_data.get("workspace_name", "Unknown")
            print(f"Workspace: {workspace}")

        return True

    except Exception as e:
        print(f"\nError exchanging token: {e}")
        return False


def get_apikey(service: str):
    """Handle API key services (no OAuth, just open page and paste)."""
    if service not in APIKEY_SERVICES:
        print(f"Unknown service: {service}")
        return False

    config = APIKEY_SERVICES[service]
    print(f"\n{'='*60}")
    print(f"Get {config['name']} API Key")
    print('='*60)

    print(f"\nOpening {config['key_url']}...")
    webbrowser.open(config["key_url"])

    print(f"\n1. Create or copy your API key from the browser")
    if config["prefix"]:
        print(f"   (Keys typically start with '{config['prefix']}')")
    print(f"2. Paste it below\n")

    api_key = input(f"{config['name']} API Key: ").strip()

    if not api_key:
        print("Cancelled.")
        return False

    # Validate prefix if specified
    if config["prefix"] and not api_key.startswith(config["prefix"]):
        print(f"Warning: Key doesn't start with '{config['prefix']}'")
        proceed = input("Save anyway? [y/N]: ").strip().lower()
        if proceed != 'y':
            return False

    # Save to tokens
    tokens_dir = Path(".aether/tokens")
    tokens_dir.mkdir(parents=True, exist_ok=True)

    token_file = tokens_dir / f"{service}.json"
    with open(token_file, "w") as f:
        json.dump({
            "api_key": api_key,
            "env_var": config["env_var"],
            "service": config["name"],
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{config['name']} API Key Saved!")
    print('='*60)
    print(f"Stored in: {token_file}")
    print(f"\nTo use in shell:")
    print(f'  $env:{config["env_var"]} = "{api_key[:8]}..."')

    return True


def list_connections():
    """List saved connections."""
    tokens_dir = Path(".aether/tokens")
    if not tokens_dir.exists():
        print("No connections yet.")
        return

    print("\nSaved connections:")
    for token_file in tokens_dir.glob("*.json"):
        provider = token_file.stem
        with open(token_file) as f:
            data = json.load(f)

        if provider == "notion":
            info = data.get("workspace_name", "Connected")
        elif "api_key" in data:
            # API key service
            key = data["api_key"]
            info = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "Connected"
        else:
            info = "Connected"

        print(f"  {provider}: {info}")


def main():
    if len(sys.argv) < 2:
        print("OAuth Connect - One-Click Authorization")
        print("="*50)
        print("\nUsage:")
        print("  python -m aetherauth.oauth_connect <service>")
        print("  python -m aetherauth.oauth_connect list")
        print("\nOAuth Services (browser login):")
        for key, config in OAUTH_CONFIGS.items():
            print(f"  {key:12} - {config['name']}")
        print("\nAPI Key Services (copy/paste):")
        for key, config in APIKEY_SERVICES.items():
            print(f"  {key:12} - {config['name']}")

        list_connections()
        return

    command = sys.argv[1].lower()

    if command == "list":
        list_connections()
    elif command == "all":
        # Quick setup for all API key services
        print("Opening all API key pages...")
        for service in APIKEY_SERVICES:
            get_apikey(service)
            print()
    elif command in OAUTH_CONFIGS:
        if not REQUESTS_AVAILABLE:
            print("Install requests first: pip install requests")
            return
        connect(command)
    elif command in APIKEY_SERVICES:
        get_apikey(command)
    else:
        print(f"Unknown service: {command}")
        print(f"\nAvailable: {list(OAUTH_CONFIGS.keys()) + list(APIKEY_SERVICES.keys())}")


if __name__ == "__main__":
    main()
