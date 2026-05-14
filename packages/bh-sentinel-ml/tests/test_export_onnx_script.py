"""Tests for scripts/export_onnx.py.

The script itself is an operator tool, not a runtime import path. We test
two layers:

1. **Unit tests (default CI):** validate helpers without invoking optimum or
   downloading any real model. ONNX validation, manifest writing, tokenizer
   copying, SHA256 computation. All work off the tiny ONNX fixture from
   `conftest.py`.

2. **End-to-end real-model test (marked `real_model`, skipped in default CI):**
   actually invokes the script's main() with a small public source. Verifies
   the script produces a usable artifact dir. Slow; opt-in.

The script lives outside any installed Python package (it's at the repo
root under `scripts/`), so we import it via importlib by absolute path.
This keeps the test resilient to packaging changes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_onnx.py"


def _load_export_module() -> Any:
    """Import scripts/export_onnx.py as a module without it being on sys.path."""
    if not SCRIPT_PATH.exists():
        pytest.skip(f"scripts/export_onnx.py not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("export_onnx_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Unit tests: helpers without optimum / network / torch.
# ---------------------------------------------------------------------------


def test_module_imports_cleanly() -> None:
    """The script file is parseable and importable with no top-level side effects."""
    module = _load_export_module()
    assert hasattr(module, "main")
    assert hasattr(module, "_validate_onnx_io_contract")
    assert hasattr(module, "_quantize_dynamic_int8")
    assert hasattr(module, "_copy_tokenizer_files")
    assert hasattr(module, "_write_manifest")
    assert hasattr(module, "_sha256_of_file")


def test_help_runs_without_optimum_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--help` must not require optimum -- otherwise CI can't even inspect the script."""
    module = _load_export_module()
    with pytest.raises(SystemExit) as exc_info:
        module._parse_args(["--help"])
    assert exc_info.value.code == 0


def test_parse_args_requires_source_revision() -> None:
    """--source-revision is the only required arg (everything else has a default)."""
    module = _load_export_module()
    with pytest.raises(SystemExit):
        module._parse_args([])  # missing --source-revision


def test_parse_args_defaults_to_roberta_large_mnli() -> None:
    """Default source must match the v0.2.2 pinned source (per provenance doc)."""
    module = _load_export_module()
    args = module._parse_args(["--source-revision", "deadbeef"])
    assert args.source_model == "FacebookAI/roberta-large-mnli"
    assert args.onnx_filename == "model_int8.onnx"
    assert args.task == "zero-shot-classification"


def test_validate_onnx_io_contract_accepts_tiny_fixture(tiny_nli_model: Path) -> None:
    """The bundled tiny_nli fixture has the correct input/output contract."""
    module = _load_export_module()
    module._validate_onnx_io_contract(tiny_nli_model)


def test_validate_onnx_io_contract_rejects_extra_input(
    tmp_path: Path,
) -> None:
    """ONNX with `decoder_input_ids` as an extra input must be rejected."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "seq"]
    )
    decoder_input_ids = helper.make_tensor_value_info(
        "decoder_input_ids", TensorProto.INT64, ["batch", "seq"]
    )
    logits_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3])
    weight = np.array([[0.1, 0.05, -0.05]], dtype=np.float32)
    weight_init = numpy_helper.from_array(weight, name="W")
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
        name="bad_extra_input",
        inputs=[input_ids, attention_mask, decoder_input_ids],
        outputs=[logits_out],
        initializer=[weight_init, axes_init],
    )
    opset = helper.make_opsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset])
    model.ir_version = 7
    bad_path = tmp_path / "bad_extra_input.onnx"
    onnx.save(model, str(bad_path))

    module = _load_export_module()
    with pytest.raises(SystemExit, match="unexpected extra inputs"):
        module._validate_onnx_io_contract(bad_path)


def test_validate_onnx_io_contract_rejects_static_input_axes(
    tmp_path: Path,
) -> None:
    """ONNX with fixed (non-symbolic) input dims must be rejected.

    Regression guard for the static-axes bug that shipped in v0.2.1: passing
    `no_dynamic_axes=True` to optimum bakes example batch/seq dims into the
    graph. At inference the runtime then rejects any tensor whose dims
    differ from the baked-in example.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Both inputs have FIXED dims (dim_value, no dim_param). This is the bug.
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [2, 16])
    attention_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [2, 16])
    logits_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [2, 3])
    weight = np.array([[0.1, 0.05, -0.05]], dtype=np.float32)
    weight_init = numpy_helper.from_array(weight, name="W")
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
        name="static_axes_bug",
        inputs=[input_ids, attention_mask],
        outputs=[logits_out],
        initializer=[weight_init, axes_init],
    )
    opset = helper.make_opsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset])
    model.ir_version = 7
    bad_path = tmp_path / "static_axes.onnx"
    onnx.save(model, str(bad_path))

    module = _load_export_module()
    with pytest.raises(SystemExit, match="fixed .non-symbolic. dimension"):
        module._validate_onnx_io_contract(bad_path)


def test_validate_onnx_io_contract_rejects_wrong_class_dim(
    tmp_path: Path,
) -> None:
    """ONNX with logits output shape [batch, 2] (binary instead of MNLI 3-way) must be rejected."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "seq"]
    )
    logits_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])
    weight = np.array([[0.1, 0.05]], dtype=np.float32)
    weight_init = numpy_helper.from_array(weight, name="W")
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
        name="bad_class_dim",
        inputs=[input_ids, attention_mask],
        outputs=[logits_out],
        initializer=[weight_init, axes_init],
    )
    opset = helper.make_opsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset])
    model.ir_version = 7
    bad_path = tmp_path / "bad_class_dim.onnx"
    onnx.save(model, str(bad_path))

    module = _load_export_module()
    with pytest.raises(SystemExit, match="class dim is 2"):
        module._validate_onnx_io_contract(bad_path)


def test_copy_tokenizer_files_requires_tokenizer_json(tmp_path: Path) -> None:
    """The script must fail if tokenizer.json is missing from the export dir."""
    module = _load_export_module()
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    # Create only optional files, no tokenizer.json.
    (src / "tokenizer_config.json").write_text("{}")
    with pytest.raises(SystemExit, match="tokenizer.json"):
        module._copy_tokenizer_files(src, dst)


def test_copy_tokenizer_files_copies_optional_files(tmp_path: Path) -> None:
    """All present optional tokenizer files should make it into the artifact dir."""
    module = _load_export_module()
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "tokenizer.json").write_text('{"version": "1.0"}')
    (src / "tokenizer_config.json").write_text("{}")
    (src / "vocab.json").write_text("{}")
    (src / "merges.txt").write_text("# merges")
    # special_tokens_map.json intentionally omitted (not all tokenizers have it).
    module._copy_tokenizer_files(src, dst)
    assert (dst / "tokenizer.json").exists()
    assert (dst / "tokenizer_config.json").exists()
    assert (dst / "vocab.json").exists()
    assert (dst / "merges.txt").exists()
    assert not (dst / "special_tokens_map.json").exists()


def test_write_manifest_writes_expected_fields(tmp_path: Path) -> None:
    """manifest.json must contain the fields downstream tooling depends on."""
    module = _load_export_module()
    manifest = module._write_manifest(
        output_dir=tmp_path,
        source_model="facebook/bart-large-mnli",
        source_revision="d7645e127eaf1aefc7862fd59a17a5aa8558b8ce",
        task="zero-shot-classification",
        onnx_filename="model_int8.onnx",
        onnx_sha256="0" * 64,
        quant_versions={"onnx": "1.21.0", "onnxruntime": "1.26.0"},
    )
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk == manifest
    # Required keys for the HF model card + provenance doc.
    for key in (
        "source_model",
        "source_revision",
        "task",
        "onnx_filename",
        "sha256",
        "export_date_utc",
        "tooling",
        "quantization",
        "consumed_by",
        "provenance_doc",
    ):
        assert key in on_disk, f"manifest missing required key: {key}"


def test_sha256_of_file_matches_known_digest(tmp_path: Path) -> None:
    """SHA256 helper matches the same digest as `shasum -a 256`."""
    module = _load_export_module()
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world\n")
    # Known SHA256("hello world\n"):
    assert (
        module._sha256_of_file(f)
        == "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
    )


# ---------------------------------------------------------------------------
# End-to-end real-model test (opt-in).
# ---------------------------------------------------------------------------


@pytest.mark.real_model
def test_end_to_end_export_small_model(tmp_path: Path) -> None:
    """Exercise main() against a small public NLI model and verify the artifact layout.

    Marked real_model because it:
      - downloads ~250MB of model weights from HF Hub
      - runs optimum-cli (subprocess; slow)
      - actually quantizes (CPU-bound; ~30s)

    Default CI excludes this. Run manually with `pytest -m real_model`.
    """
    module = _load_export_module()
    output_dir = tmp_path / "artifact_staging"
    rc = module.main(
        [
            "--source-model",
            "cross-encoder/nli-distilroberta-base",
            "--source-revision",
            "main",
            "--output-dir",
            str(output_dir),
            "--onnx-filename",
            "model_int8.onnx",
            "--task",
            "text-classification",
        ]
    )
    assert rc == 0
    assert (output_dir / "model_int8.onnx").exists()
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "manifest.json").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["source_model"] == "cross-encoder/nli-distilroberta-base"
    assert manifest["onnx_filename"] == "model_int8.onnx"
    assert len(manifest["sha256"]) == 64
