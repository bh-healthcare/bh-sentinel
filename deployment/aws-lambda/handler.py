"""AWS Lambda handler for bh-sentinel clinical safety signal detection.

This handler receives clinical text via API Gateway, runs the bh-sentinel
pipeline, and returns structured safety flags. No clinical text is persisted.
"""

from __future__ import annotations

import json
from typing import Any


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Process a clinical text analysis request.

    Args:
        event: API Gateway proxy event containing the clinical text in the body.
        context: Lambda runtime context.

    Returns:
        API Gateway proxy response with structured safety flags.
    """
    raise NotImplementedError(
        "Lambda handler is a reference stub. "
        "See docs/deployment-guide.md for implementation instructions."
    )


def _build_response(status_code: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
