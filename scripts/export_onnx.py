#!/usr/bin/env python3
"""Export the bh-sentinel-ml baseline NLI model to ONNX with INT8 quantization.

This script is the **operator tool** that produces the canonical ONNX artifact
referenced by `config/ml/ml_config.yaml`. It is NOT a runtime dependency of
`bh-sentinel-ml`; it lives at the repo root so it is invoked manually during
a release (see Phase 2 of the bh-sentinel-ml v0.2.1 release plan).

The output of this script is everything `hf upload` needs to publish a model
repo on HuggingFace Hub:

    artifact_staging/
    ├── model_int8.onnx               quantized weights
    ├── tokenizer.json                HF tokenizers library file
    ├── tokenizer_config.json         tokenizer config
    ├── vocab.json                    BPE vocab (if applicable to source)
    ├── merges.txt                    BPE merges (if applicable to source)
    ├── special_tokens_map.json       special token map
    └── manifest.json                 machine-readable provenance metadata

The downstream `TransformerClassifier.__init__` (see
`packages/bh-sentinel-ml/src/bh_sentinel/ml/transformer.py`) consumes
`model_int8.onnx` and `tokenizer.json` from this directory layout; the rest
of the tokenizer files travel with `tokenizer.json` for completeness.

ONNX shape contract enforced by this script (validated before quantization):

    Inputs:  input_ids       int64  [batch, seq]
             attention_mask  int64  [batch, seq]
    Output:  logits          float  [batch, 3]    (entailment, neutral, contradiction)

If the source model produces an ONNX with additional inputs (e.g. a separate
`decoder_input_ids` for raw seq2seq export), this script fails fast with a
clear error rather than silently producing an artifact `TransformerClassifier`
cannot consume.

License verification gate (Phase 0a) must pass BEFORE this script is run.
See `docs/ml-artifact-provenance.md` for the current pinned source.

Required tools (install into your operator venv; not bh-sentinel-ml runtime deps):
    optimum[onnxruntime]>=2.0,<3       # 2.x dropped the broken torch 2.11+ symbol import
    optimum-onnx>=0.1,<1               # auto-installed as a transitive dep of optimum 2.x
    onnx>=1.15,<2
    onnxruntime>=1.16,<2
    torch>=2.5                          # required by optimum 2.x's exporter

Note: bh-sentinel-ml's runtime deps (in `packages/bh-sentinel-ml/pyproject.toml`)
are intentionally NARROWER than these operator-tool deps. The runtime never
loads `optimum` or `torch`; it only loads `onnxruntime`, `tokenizers`, and
`huggingface_hub` to consume the artifact this script produces.

Usage:
    python scripts/export_onnx.py \\
        --source-model facebook/bart-large-mnli \\
        --source-revision d7645e127eaf1aefc7862fd59a17a5aa8558b8ce \\
        --output-dir ./artifact_staging \\
        --onnx-filename model_int8.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Files that must live next to model_int8.onnx in the HF artifact repo so the
# pipeline's `Tokenizer.from_file(...)` works off a single directory. Not all
# sources include every file (BPE-tokenizer models have vocab.json + merges.txt;
# SentencePiece-tokenizer models have spiece.model); we copy whichever subset
# exists. tokenizer.json is REQUIRED; the script fails if it is missing.
TOKENIZER_REQUIRED_FILES: tuple[str, ...] = ("tokenizer.json",)
TOKENIZER_OPTIONAL_FILES: tuple[str, ...] = (
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
)

# Expected ONNX I/O shape contract. If the source model's ONNX export does not
# match this, the artifact will not be consumable by TransformerClassifier and
# we fail fast at export time rather than at first inference.
EXPECTED_INPUT_NAMES: frozenset[str] = frozenset({"input_ids", "attention_mask"})
EXPECTED_OUTPUT_LOGIT_DIM: int = 3  # entailment, neutral, contradiction


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="export_onnx",
        description=(
            "Export the pinned NLI source model to ONNX, INT8-quantize it, and "
            "stage everything HF needs for a model repo upload."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-model",
        required=False,
        default="facebook/bart-large-mnli",
        help=(
            "HuggingFace Hub repo id of the source NLI model. Default is the "
            "v0.2.1 pinned source. See docs/ml-artifact-provenance.md."
        ),
    )
    parser.add_argument(
        "--source-revision",
        required=True,
        help=(
            "Pinned HuggingFace Hub commit SHA of the source. NEVER use 'main' "
            "or a branch name for a release artifact -- upstream can change."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        type=Path,
        default=Path("./artifact_staging"),
        help="Directory to write the staged artifact + tokenizer + manifest into.",
    )
    parser.add_argument(
        "--onnx-filename",
        required=False,
        default="model_int8.onnx",
        help="Name of the final quantized ONNX file inside --output-dir.",
    )
    parser.add_argument(
        "--task",
        required=False,
        default="zero-shot-classification",
        help=(
            "optimum-cli export task. Default matches the source model's HF "
            "pipeline_tag. Override only if you know the source uses a "
            "different task taxonomy."
        ),
    )
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help=(
            "Keep the intermediate FP32 export in <output-dir>/_fp32/ after "
            "quantization (useful for debugging quant drift). Omitted by "
            "default to save ~400MB on disk."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional HF cache dir for the optimum download step. Falls back "
            "to the default HF cache (~/.cache/huggingface)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir: Path = args.output_dir.resolve()
    fp32_dir = output_dir / "_fp32"
    onnx_path = output_dir / args.onnx_filename

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Source:        {args.source_model}@{args.source_revision}")
    print(f"      Task:          {args.task}")
    print(f"      Output dir:    {output_dir}")
    print(f"      Final artifact:{onnx_path}")
    print(f"      Python:        {sys.executable}")
    print(f"      Python ver:    {sys.version.split()[0]}")

    _check_optimum_importable()

    print("\n[2/6] Running optimum-cli export onnx (FP32) ...")
    _run_optimum_export(
        source_model=args.source_model,
        revision=args.source_revision,
        task=args.task,
        out_dir=fp32_dir,
        cache_dir=args.cache_dir,
    )

    print("\n[3/6] Validating ONNX I/O contract ...")
    fp32_onnx = _locate_fp32_onnx(fp32_dir)
    _validate_onnx_io_contract(fp32_onnx)

    print(f"\n[4/6] INT8-quantizing -> {onnx_path} ...")
    quant_versions = _quantize_dynamic_int8(fp32_onnx, onnx_path)

    print("\n[5/6] Copying tokenizer files to artifact dir ...")
    _copy_tokenizer_files(fp32_dir, output_dir)

    print("\n[6/6] Writing manifest.json + computing SHA256 ...")
    sha256 = _sha256_of_file(onnx_path)
    manifest = _write_manifest(
        output_dir=output_dir,
        source_model=args.source_model,
        source_revision=args.source_revision,
        task=args.task,
        onnx_filename=args.onnx_filename,
        onnx_sha256=sha256,
        quant_versions=quant_versions,
    )

    if not args.keep_fp32:
        shutil.rmtree(fp32_dir, ignore_errors=True)
        print(f"      Removed intermediate FP32 dir: {fp32_dir}")

    _print_summary(manifest=manifest, output_dir=output_dir)
    return 0


def _check_optimum_importable() -> None:
    """Fail fast if the optimum exporter cannot be imported in this venv.

    The lazy-import nature of `optimum-cli --help` means the CLI launches
    even when the actual exporter submodule is broken (typical cause:
    optimum 1.x paired with torch >= 2.11, which removed an internal
    symbol the older patcher uses). Without this check, the breakage
    surfaces only after the subprocess has already started and produced
    a Python traceback from inside the CLI -- a poor operator experience.

    This function imports the same submodule the CLI actually uses, so
    any compatibility issue surfaces in milliseconds with a clean message.
    """
    try:
        from optimum.exporters.onnx import main_export  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "optimum exporter is not importable in this venv: "
            f"{type(exc).__name__}: {exc}\n\n"
            "This is almost always a torch <-> optimum version skew. Fix by "
            "upgrading optimum to the 2.x line:\n\n"
            "    pip install --upgrade 'optimum[onnxruntime]>=2.0,<3'\n\n"
            "See the module docstring at the top of scripts/export_onnx.py "
            "for the full operator-tool dependency matrix."
        ) from exc


def _run_optimum_export(
    *,
    source_model: str,
    revision: str,
    task: str,
    out_dir: Path,
    cache_dir: Path | None,
) -> None:
    """Run optimum's ONNX export via its Python API (not the subprocess CLI).

    Why the Python API and not `subprocess.run(['optimum-cli', ...])`?

    1. `optimum-cli` is only on `$PATH` when the venv's `bin/` directory is
       activated. Invoking this script via an explicit interpreter path
       (e.g. `~/.venvs/bh-sentinel-export/bin/python scripts/export_onnx.py`)
       does NOT put the venv's `bin/` on `$PATH` for child processes, so
       `subprocess.run(['optimum-cli', ...])` resolves to whatever's first on
       PATH -- often the wrong venv or `FileNotFoundError`.
    2. When optimum throws, the subprocess CLI swallows the Python frames
       behind a generic non-zero exit code. The Python API surfaces the
       full traceback to this script's caller -- much easier to debug.
    3. Optimum 2.x's `main_export()` is the same entry point the CLI calls;
       behavior is identical.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"      Source:   {source_model}@{revision}")
    print(f"      Task:     {task}")
    print(f"      Target:   {out_dir}")
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as exc:
        raise SystemExit(
            f"Could not import optimum.exporters.onnx: {exc}. "
            "Run: pip install --upgrade 'optimum[onnxruntime]>=2.0,<3'"
        ) from exc

    try:
        main_export(
            model_name_or_path=source_model,
            output=str(out_dir),
            task=task,
            framework="pt",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            no_dynamic_axes=True,
        )
    except TypeError as exc:
        # Optimum 2.x sometimes raises TypeError on certain (transformers,
        # optimum) combos when introspecting model configs. Most common is
        # "NormalizedConfig.__init__() got multiple values for argument
        # 'allow_new'" -- a known bug when transformers >=4.53 paired with
        # optimum 2.1.0's NormalizedConfig kwargs forwarding.
        raise SystemExit(
            f"optimum export raised TypeError: {exc}\n\n"
            "Likely cause: transformers <-> optimum version skew. Try:\n\n"
            "    pip install --upgrade 'optimum[onnxruntime]>=2.0,<3' "
            "'transformers>=4.55'\n\n"
            "If the bug persists, switch to direct torch.onnx.export "
            "(see scripts/export_onnx.py docstring for plan B), or "
            "use a different --source-model whose ONNX export is "
            "better tested upstream."
        ) from exc


def _locate_fp32_onnx(fp32_dir: Path) -> Path:
    """Find the single FP32 ONNX file in the optimum export output.

    optimum-cli writes the model under different filenames depending on task
    and model architecture. For sequence-classification it is typically
    `model.onnx`. We pick the largest .onnx file under fp32_dir to be robust.
    """
    candidates = sorted(fp32_dir.rglob("*.onnx"))
    if not candidates:
        raise SystemExit(
            f"No .onnx files found under {fp32_dir}. optimum-cli may have "
            "failed silently or produced an unexpected file layout."
        )
    if len(candidates) == 1:
        return candidates[0]
    # Multiple .onnx files (encoder/decoder split for seq2seq tasks). We don't
    # support that layout because the runtime expects a single ONNX session.
    # Fail with guidance.
    found = "\n  - ".join(str(p) for p in candidates)
    raise SystemExit(
        f"Found multiple .onnx files under {fp32_dir}:\n  - {found}\n\n"
        "TransformerClassifier expects a single ONNX file. Re-run with "
        "--task text-classification or another task that produces a single-file "
        "classification head (not an encoder/decoder split)."
    )


def _validate_onnx_io_contract(onnx_path: Path) -> None:
    """Fail-fast if the exported ONNX inputs/outputs don't match the runtime contract.

    The bh-sentinel-ml pipeline (see packages/bh-sentinel-ml/tests/conftest.py
    shape contract and packages/bh-sentinel-core/src/bh_sentinel/core/pipeline.py
    line 226 onward) requires:

        Inputs:  input_ids       [batch, seq]
                 attention_mask  [batch, seq]
        Output:  logits          [batch, 3]
    """
    import onnx as _onnx

    model = _onnx.load(str(onnx_path), load_external_data=False)
    actual_inputs = {i.name for i in model.graph.input}
    extra = actual_inputs - EXPECTED_INPUT_NAMES
    missing = EXPECTED_INPUT_NAMES - actual_inputs
    if missing:
        raise SystemExit(
            f"ONNX input mismatch in {onnx_path.name}: "
            f"missing required inputs {sorted(missing)}. "
            "TransformerClassifier feeds only `input_ids` + `attention_mask`."
        )
    if extra:
        raise SystemExit(
            f"ONNX input mismatch in {onnx_path.name}: "
            f"unexpected extra inputs {sorted(extra)}. "
            "TransformerClassifier does not pass these. Re-run with a different "
            "--task, or add the extra inputs as fixed initializers in the ONNX graph."
        )

    outputs = list(model.graph.output)
    if not outputs:
        raise SystemExit(f"ONNX {onnx_path.name} has no outputs.")
    # Output 0 is the classification logits per HF convention.
    out0 = outputs[0]
    dims = list(out0.type.tensor_type.shape.dim)
    if len(dims) != 2:
        raise SystemExit(
            f"ONNX output 0 rank mismatch: expected 2 (batch, classes), got {len(dims)}. "
            f"Output name: {out0.name}, shape: {dims}."
        )
    classes_dim = dims[1]
    classes_value = classes_dim.dim_value if classes_dim.HasField("dim_value") else None
    if classes_value not in (None, 0, EXPECTED_OUTPUT_LOGIT_DIM):
        # dim_value == 0 OR HasField=False both mean "symbolic"; both are fine.
        raise SystemExit(
            f"ONNX output 0 class dim is {classes_value}, expected "
            f"{EXPECTED_OUTPUT_LOGIT_DIM} (entailment, neutral, contradiction). "
            "This source model does not produce MNLI-shaped logits."
        )
    print(
        f"      Contract OK: inputs={sorted(actual_inputs)}, "
        f"output={out0.name}, classes={classes_value or 'symbolic'}"
    )


def _quantize_dynamic_int8(fp32_onnx: Path, int8_onnx: Path) -> dict[str, str]:
    """Run onnxruntime dynamic INT8 quantization.

    Uses the library's default op_types_to_quantize set (MatMul, Gemm, etc.)
    rather than picking an explicit list, because BART's transformer ops are
    well-covered by the default and adding speculative op types (e.g.
    'Attention', which isn't a standard ONNX op until very recent opsets)
    can cause silent failures.

    Returns the version strings used, for inclusion in manifest.json.
    """
    import onnx as _onnx
    import onnxruntime as _ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_onnx.parent.mkdir(parents=True, exist_ok=True)
    if int8_onnx.exists():
        int8_onnx.unlink()
    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )
    if not int8_onnx.exists():
        raise SystemExit(
            f"Quantization completed without error but {int8_onnx} was not written."
        )
    return {
        "onnx": str(getattr(_onnx, "__version__", "unknown")),
        "onnxruntime": str(getattr(_ort, "__version__", "unknown")),
    }


def _copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy tokenizer.json (required) + any optional sibling tokenizer files."""
    src_files = {p.name: p for p in src_dir.iterdir() if p.is_file()}
    for required in TOKENIZER_REQUIRED_FILES:
        if required not in src_files:
            raise SystemExit(
                f"Required tokenizer file missing from optimum export: {required}. "
                "Without tokenizer.json the downstream pipeline's "
                "`Tokenizer.from_file()` cannot construct the tokenizer."
            )
        shutil.copy2(src_files[required], dst_dir / required)
        print(f"      Copied {required}")
    for optional in TOKENIZER_OPTIONAL_FILES:
        if optional in src_files:
            shutil.copy2(src_files[optional], dst_dir / optional)
            print(f"      Copied {optional} (optional)")


def _write_manifest(
    *,
    output_dir: Path,
    source_model: str,
    source_revision: str,
    task: str,
    onnx_filename: str,
    onnx_sha256: str,
    quant_versions: dict[str, str],
) -> dict[str, Any]:
    """Write manifest.json and return the dict so the summary can echo it."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        optimum_version = _pkg_version("optimum")
    except PackageNotFoundError:
        optimum_version = "unknown"
    manifest: dict[str, Any] = {
        "source_model": source_model,
        "source_revision": source_revision,
        "task": task,
        "onnx_filename": onnx_filename,
        "sha256": onnx_sha256,
        "export_date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "tooling": {
            "optimum": optimum_version,
            **quant_versions,
        },
        "quantization": {
            "method": "onnxruntime.quantization.quantize_dynamic",
            "weight_type": "QInt8",
            "per_channel": False,
            "reduce_range": False,
        },
        "consumed_by": "bh-sentinel-ml >= 0.2.1",
        "provenance_doc": "docs/ml-artifact-provenance.md",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _print_summary(*, manifest: dict[str, Any], output_dir: Path) -> None:
    sha = manifest["sha256"]
    print("\n" + "=" * 78)
    print("Export complete.")
    print("=" * 78)
    print(f"Artifact:         {output_dir / manifest['onnx_filename']}")
    print(f"SHA256:           {sha}")
    print(f"Source:           {manifest['source_model']}@{manifest['source_revision']}")
    print(f"Manifest:         {output_dir / 'manifest.json'}")
    print()
    print("Ready-to-paste snippet for config/ml/ml_config.yaml:")
    print()
    print("  model_repo: bh-healthcare/distilbart-mnli-12-3-int8-onnx")
    print("  model_revision: <PASTE HF COMMIT SHA AFTER hf upload>")
    print(f'  model_sha256: "{sha}"')
    print()
    print("Next step (Phase 2 step 2 in the release plan):")
    print("  1. Author artifact_staging/README.md (HF model card).")
    print(
        "  2. hf repos create bh-healthcare/distilbart-mnli-12-3-int8-onnx --type model --public"
    )
    print(f"  3. hf upload bh-healthcare/distilbart-mnli-12-3-int8-onnx {output_dir} .")
    print("  4. Grab the resulting commit SHA from `hf api ...` for ml_config.yaml.")
    print()


if __name__ == "__main__":
    sys.exit(main())
