# bh-sentinel AWS Lambda Container Image
#
# Reference Dockerfile for deploying bh-sentinel as a Lambda container.
# Uses the AWS-provided Python base image for Lambda.
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

COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
