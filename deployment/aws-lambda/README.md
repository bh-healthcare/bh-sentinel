# AWS Lambda Deployment

Reference deployment for running bh-sentinel as an AWS Lambda function behind API Gateway.

**Status:** Scaffold files for v0.2. The handler, Dockerfile, and requirements.txt are present as starting points. The ONNX model COPY is commented out until v0.2 (model export tooling). Full deployment instructions will be in `docs/deployment-guide.md` when v0.2 ships.

This deployment is designed for zero data egress: the Lambda runs in a private VPC subnet with no internet access. All AWS service calls route through VPC endpoints. Clinical text is processed in-memory and discarded — only flag metadata is logged.

## Config loading strategy

| Config | Location | Why |
|---|---|---|
| `flag_taxonomy.json` | Baked into container image | Version-coupled; image rebuild enforces coordination |
| `emotion_lexicon.json` | Baked into container image | Rarely changes; coupled to taxonomy version |
| `patterns.yaml` | S3 bucket (loaded at Lambda init) | Clinical teams update frequently without rebuild |
| `rules.json` | S3 bucket (loaded at Lambda init) | Clinical teams update frequently without rebuild |
| ONNX model | Baked into container image | ~65MB; changes only on retrain |

The init-time config validator verifies that S3-loaded configs are compatible with the baked-in taxonomy version before accepting requests. See `docs/architecture.md` Section 3.3 and Section 5.4 for details.

## Files

- `handler.py` — Lambda entry point
- `Dockerfile` — Container image definition
- `requirements.txt` — Python dependencies
- `../terraform/` — Infrastructure modules (VPC, Lambda, API Gateway, ECR, S3, monitoring)

See `docs/architecture.md` Section 5 for the full reference deployment architecture.
