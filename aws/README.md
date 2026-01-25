# AWS Lambda Deployment

This directory contains the AWS Lambda deployment configuration for SCBE-AETHERMOORE API.

## Files

| File | Description |
|------|-------------|
| `lambda_handler.py` | Lambda entry point using Mangum adapter |
| `requirements-lambda.txt` | Minimal dependencies for Lambda package |

## Prerequisites

1. **AWS Account** with Lambda and API Gateway access
2. **IAM Role** with Lambda execution permissions
3. **API Gateway** (HTTP API or REST API) configured

## Required GitHub Secrets

Configure these in your repository settings (`Settings > Secrets and variables > Actions`):

| Secret | Description | Required |
|--------|-------------|----------|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key | Yes |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key | Yes |
| `SCBE_API_KEY` | API key for production | Yes |
| `SCBE_API_KEY_STAGING` | API key for staging | For staging |
| `PRODUCTION_API_URL` | Production API base URL | Optional |
| `STAGING_API_URL` | Staging API base URL | Optional |

## Lambda Configuration

### Recommended Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Runtime | Python 3.11 | Latest supported |
| Handler | `lambda_handler.lambda_handler` | Entry point |
| Memory | 512 MB | Increase for heavy loads |
| Timeout | 30 seconds | Max for API Gateway |
| Architecture | x86_64 | Or arm64 for cost savings |

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SCBE_API_KEY` | API authentication key(s) | Yes |
| `ENVIRONMENT` | `production` or `staging` | Recommended |
| `LOG_LEVEL` | `INFO`, `DEBUG`, `WARNING` | Optional |
| `FIREBASE_ENABLED` | `true` or `false` | Optional |

## Manual Deployment

### Local Build

```bash
# Create package directory
mkdir -p lambda_package

# Install dependencies
pip install -r aws/requirements-lambda.txt -t lambda_package/

# Copy application code
cp -r api lambda_package/
cp aws/lambda_handler.py lambda_package/

# Create zip
cd lambda_package && zip -r ../deployment.zip . && cd ..

# Deploy (requires AWS CLI configured)
aws lambda update-function-code \
  --function-name scbe-aethermoore-api-production \
  --zip-file fileb://deployment.zip
```

### Using Docker

```bash
# Build deployment package in Lambda-compatible environment
docker run --rm -v $(pwd):/app -w /app python:3.11 bash -c "
  pip install -r aws/requirements-lambda.txt -t lambda_package/
  cp -r api lambda_package/
  cp aws/lambda_handler.py lambda_package/
  cd lambda_package && zip -r ../deployment.zip .
"
```

## API Gateway Setup

### HTTP API (Recommended)

1. Create HTTP API in API Gateway console
2. Add route: `ANY /{proxy+}`
3. Create Lambda integration pointing to your function
4. Enable `Lambda proxy integration`
5. Deploy to stage (e.g., `prod`)

### Custom Domain

```bash
# Create certificate in ACM (us-east-1 for edge-optimized)
aws acm request-certificate \
  --domain-name api.yourdomain.com \
  --validation-method DNS

# Create custom domain mapping
aws apigatewayv2 create-domain-name \
  --domain-name api.yourdomain.com \
  --domain-name-configurations CertificateArn=arn:aws:acm:...
```

## Monitoring

### CloudWatch Logs

Lambda logs are automatically sent to CloudWatch. Log group name:
`/aws/lambda/scbe-aethermoore-api-production`

### CloudWatch Alarms (Recommended)

```bash
# Create error alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "SCBE-API-Errors" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --dimensions Name=FunctionName,Value=scbe-aethermoore-api-production
```

## Cost Optimization

### Lambda Pricing (us-west-2)

- **Requests:** $0.20 per 1M requests
- **Duration:** $0.0000166667 per GB-second

### Cost Estimate

| Traffic | Memory | Est. Monthly Cost |
|---------|--------|-------------------|
| 100K req/month | 512MB | ~$1.50 |
| 1M req/month | 512MB | ~$15 |
| 10M req/month | 512MB | ~$150 |

### Tips

1. Use **arm64** architecture for ~20% cost savings
2. Enable **Provisioned Concurrency** for consistent latency
3. Use **Lambda SnapStart** for faster cold starts (when available for Python)
4. Set appropriate **timeout** to prevent runaway costs

## Troubleshooting

### Cold Starts

If cold starts are too slow:
1. Increase memory (more memory = more CPU)
2. Enable Provisioned Concurrency
3. Use Lambda warmer (scheduled CloudWatch event)

### Package Too Large

If deployment package exceeds 50MB:
1. Use Lambda Layers for large dependencies
2. Remove unused packages from requirements
3. Use docker-slim to minimize package

### Timeout Errors

If requests timeout:
1. Increase Lambda timeout (max 15 minutes, but API Gateway max is 30s)
2. Check for slow database connections
3. Enable connection pooling

## Security Best Practices

1. **Never** commit API keys - use GitHub Secrets
2. Enable **IAM authentication** for sensitive operations
3. Use **VPC** for database connections
4. Enable **AWS WAF** for DDoS protection
5. Rotate API keys regularly

## Architecture Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────>│ API Gateway │────>│   Lambda    │
│             │     │  (HTTP API) │     │ (FastAPI)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          │                    │                    │
                          v                    v                    v
                    ┌──────────┐        ┌──────────┐        ┌──────────┐
                    │CloudWatch│        │ Firebase │        │  Zapier  │
                    │  (Logs)  │        │  (Data)  │        │(Webhooks)│
                    └──────────┘        └──────────┘        └──────────┘
```

## Related Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Mangum - ASGI for Lambda](https://github.com/jordaneremieff/mangum)
- [AWS Lambda Python Runtime](https://docs.aws.amazon.com/lambda/latest/dg/lambda-python.html)
- [API Gateway HTTP APIs](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api.html)
