# AWS Lambda Deployment

Reference deployment for running bh-sentinel as an AWS Lambda function behind API Gateway.

This deployment is designed for zero data egress: the Lambda runs in a private VPC subnet with no internet access. All AWS service calls route through VPC endpoints. Clinical text is processed in-memory and discarded -- only flag metadata is logged.

See `docs/deployment-guide.md` for the full walkthrough and `deployment/terraform/` for the infrastructure modules.
