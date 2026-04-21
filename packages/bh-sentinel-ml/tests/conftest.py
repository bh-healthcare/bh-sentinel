"""Shared test fixtures for bh-sentinel-ml.

The core artifact is `tiny_nli_model` -- a deterministic, few-KB ONNX
model that mimics the MNLI output shape (N, 3) without requiring
torch or any network access. Built once, cached to disk under
tests/fixtures/tiny_nli.onnx, and reused across every inference-path
test. The fixture is deterministic: same inputs produce the same logits
every run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TINY_ONNX_PATH = FIXTURES_DIR / "tiny_nli.onnx"


@pytest.fixture(scope="session")
def tiny_nli_model() -> Path:
    """Path to a deterministic tiny ONNX NLI-shaped model.

    Inputs: input_ids (int64, [batch, seq]), attention_mask (int64, [batch, seq]).
    Output: logits (float32, [batch, 3]).

    The model sums attention_mask per row, divides by seq_len, and
    projects into 3 logits via a fixed weight matrix. Deterministic --
    different inputs produce different outputs, but the same input
    always produces the same output.
    """
    FIXTURES_DIR.mkdir(exist_ok=True)
    if not TINY_ONNX_PATH.exists():
        _build_tiny_nli_onnx(TINY_ONNX_PATH)
    return TINY_ONNX_PATH


@pytest.fixture(scope="session")
def tiny_nli_sha256(tiny_nli_model: Path) -> str:
    """SHA256 of the tiny_nli_model file. Used for verify-on-load tests."""
    h = hashlib.sha256()
    with open(tiny_nli_model, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_tiny_nli_onnx(out_path: Path) -> None:
    """Build a tiny ONNX model programmatically and write it to disk.

    Shape contract:
        input_ids: int64, [batch, seq]
        attention_mask: int64, [batch, seq]
        logits: float32, [batch, 3]

    Logic: cast mask to float, reduce-sum along seq, project to 3
    outputs via a fixed weight matrix. Captures the shape and
    determinism our production tests need, without importing torch.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "seq"]
    )
    logits_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3])

    weight = np.array(
        [[0.1, 0.05, -0.05]],
        dtype=np.float32,
    )
    weight_init = numpy_helper.from_array(weight, name="W")

    # Axes as an int64 input (opset 13+ convention for ReduceSum).
    axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name="reduce_axes")

    cast_mask = helper.make_node(
        "Cast", inputs=["attention_mask"], outputs=["mask_f"], to=TensorProto.FLOAT
    )
    reduce_sum = helper.make_node(
        "ReduceSum",
        inputs=["mask_f", "reduce_axes"],
        outputs=["sum_per_row"],
        keepdims=1,
    )
    matmul = helper.make_node("MatMul", inputs=["sum_per_row", "W"], outputs=["logits"])
    graph = helper.make_graph(
        nodes=[cast_mask, reduce_sum, matmul],
        name="tiny_nli",
        inputs=[input_ids, attention_mask],
        outputs=[logits_out],
        initializer=[weight_init, axes_init],
    )
    opset = helper.make_opsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset])
    model.ir_version = 7
    onnx.save(model, str(out_path))
