"""
AWS Lambda Handler for SCBE-AETHERMOORE API
============================================

This module provides the Lambda entry point using Mangum to adapt
the FastAPI application for AWS Lambda + API Gateway.

Deployment:
- Package with dependencies into a Lambda deployment package
- Configure API Gateway HTTP API or REST API
- Set environment variables: SCBE_API_KEY, FIREBASE_CONFIG (optional)

Usage:
- Lambda handler: aws.lambda_handler.handler
- Memory: 512MB recommended (1024MB for heavy load)
- Timeout: 30 seconds recommended
"""

import os
import sys
import logging

# Configure logging for Lambda
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
)
logger = logging.getLogger("scbe-lambda")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mangum import Mangum
except ImportError:
    logger.error("Mangum not installed. Run: pip install mangum")
    raise

# Import the FastAPI app
try:
    from api.main import app
except ImportError as e:
    logger.error(f"Failed to import API: {e}")
    raise

# Create Mangum handler for Lambda
handler = Mangum(
    app,
    lifespan="off",  # Disable lifespan for Lambda cold starts
    api_gateway_base_path=None,  # Set if using custom domain
)


def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda handler function.

    This is the main entry point for Lambda invocations.
    Mangum handles the translation between Lambda events
    and ASGI requests.

    Args:
        event: Lambda event dict (from API Gateway)
        context: Lambda context object

    Returns:
        API Gateway response dict
    """
    # Log cold start detection
    if hasattr(context, 'aws_request_id'):
        logger.info(f"Lambda request: {context.aws_request_id}")

    # Handle warming requests (keep Lambda warm)
    if event.get("source") == "serverless-plugin-warmup":
        logger.info("Warmup request received")
        return {"statusCode": 200, "body": "Warmed"}

    # Handle health check for scheduled events
    if event.get("source") == "aws.events":
        logger.info("Scheduled health check")
        return {"statusCode": 200, "body": "Healthy"}

    # Delegate to Mangum for HTTP requests
    return handler(event, context)


# Export for different invocation patterns
__all__ = ["handler", "lambda_handler", "app"]
