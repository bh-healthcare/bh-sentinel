# bh-sentinel AWS Lambda Container Image
#
# Reference Dockerfile for deploying bh-sentinel as a Lambda container.
# Uses the AWS-provided Python base image for Lambda.
#
# Config loading strategy (see docs/architecture.md Section 5.4):
#   - flag_taxonomy.json, emotion_lexicon.json: baked into image (rarely change,
#     version-coupled). Rebuilding the image enforces version coordination.
#   - patterns.yaml, rules.json: loaded from S3 at init (clinical teams update
#     frequently without image rebuild). CONFIG_BUCKET env var set by Terraform.
#   - ONNX model: baked into image (~65MB, changes only on retrain).
#
# Build:
#   docker build -t bh-sentinel-lambda .
#
# Test locally:
#   docker run -p 9000:8080 bh-sentinel-lambda
#   curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
#     -d '{"body": "{\"text\": \"test input\"}"}'

FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

# ONNX model baked into image — no runtime download
# (model directory populated by scripts/export_onnx.py in v0.2)
# COPY models/ /opt/models/

# Configs baked into image (rarely change, version-coupled to taxonomy)
COPY ../../config/flag_taxonomy.json /opt/config/flag_taxonomy.json
COPY ../../config/emotion_lexicon.json /opt/config/emotion_lexicon.json

COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
