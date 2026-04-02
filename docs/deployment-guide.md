# Deployment Guide

AWS Lambda deployment walkthrough for bh-sentinel.

## Overview

bh-sentinel is designed for deployment inside a VPC with zero data egress. The reference architecture uses AWS Lambda with the following properties:

- **Zero data egress:** Lambda runs in a private VPC subnet with no internet access.
- **Sub-second latency:** ~60-120ms on warm invocations.
- **No PHI persistence:** Text is processed in-memory and discarded.
- **Cost-effective:** ~$50-115/month for the full stack.

See `deployment/aws-lambda/` for the Lambda handler and Dockerfile, and `deployment/terraform/` for infrastructure modules.

<!-- TODO: Expand with step-by-step deployment instructions, VPC configuration, API Gateway setup, and monitoring. -->
