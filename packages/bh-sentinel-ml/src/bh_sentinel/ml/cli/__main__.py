"""bh-sentinel-ml command-line entry point.

Three subcommands:

- download-model: pre-bake a pinned HF revision into a local cache dir.
  Intended primary use is inside a Dockerfile `RUN` step so Lambda
  cold-starts stay fully offline.
- calibrate: fit a TemperatureScaling calibrator on a JSONL labels file.
  Writes a small JSON artifact that can be committed alongside the
  ml_config.yaml for a future release.
- evaluate: run a pipeline against a fixtures file and print a
  per-fixture report -- L1 baseline by default, L1+L2 when
  --enable-transformer is passed.

All subcommands are static-template only: no raw input text is ever
included in error messages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from bh_sentinel.ml.calibration import TemperatureScaling, compute_ece


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bh-sentinel-ml",
        description=(
            "Operational CLIs for bh-sentinel-ml. Download the pinned "
            "model for a container bake, fit a calibration temperature, "
            "or run a fixture evaluation."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser(
        "download-model",
        help="Download a pinned HF Hub model revision into a local cache.",
    )
    dl.add_argument("--revision", required=True, help="Pinned HF revision SHA.")
    dl.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to download into.",
    )
    dl.add_argument(
        "--repo",
        default="valhalla/distilbart-mnli-12-3",
        help="HF Hub repo id (default matches ml_config.yaml default).",
    )
    dl.add_argument(
        "--onnx-filename",
        default="model_int8.onnx",
        help="Expected ONNX file in the downloaded snapshot.",
    )
    dl.add_argument(
        "--verify-sha256",
        default=None,
        help=(
            "Expected SHA256 of the ONNX file. If provided, the CLI "
            "fails the command on mismatch -- use this in Dockerfiles "
            "so a bad download fails the build, not the first request."
        ),
    )

    cal = sub.add_parser(
        "calibrate",
        help="Fit a TemperatureScaling calibrator on a JSONL labels file.",
    )
    cal.add_argument("--labels", required=True, type=Path)
    cal.add_argument("--out", required=True, type=Path)

    ev = sub.add_parser(
        "evaluate",
        help="Run a pipeline evaluation against a YAML fixtures file.",
    )
    ev.add_argument("--fixtures", required=True, type=Path)
    ev.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Optional shared corpus YAML (alternative to --fixtures).",
    )
    ev.add_argument(
        "--enable-transformer",
        action="store_true",
        help="Also run Layer 2; reports L1-only / L2-only / corroborated flags.",
    )

    args = parser.parse_args(argv)

    if args.cmd == "download-model":
        return _cmd_download_model(args)
    if args.cmd == "calibrate":
        return _cmd_calibrate(args)
    if args.cmd == "evaluate":
        return _cmd_evaluate(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


def _cmd_download_model(args: argparse.Namespace) -> int:
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    import huggingface_hub

    try:
        huggingface_hub.snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            local_dir=str(output),
        )
    except Exception as exc:
        print(f"error: download failed ({type(exc).__name__})", file=sys.stderr)
        return 1

    onnx_path = output / args.onnx_filename
    if not onnx_path.exists():
        print(
            f"error: expected ONNX file not found in snapshot: {args.onnx_filename}",
            file=sys.stderr,
        )
        return 1

    if args.verify_sha256:
        actual = _sha256_of_file(onnx_path)
        if actual.lower() != args.verify_sha256.lower():
            print(
                "error: SHA256 mismatch -- downloaded ONNX does not match the pinned digest.",
                file=sys.stderr,
            )
            return 1

    print(f"Downloaded {args.repo}@{args.revision[:7]} -> {output}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import numpy as np

    labels_path: Path = args.labels
    out_path: Path = args.out
    if not labels_path.exists():
        print(f"error: labels file not found: {labels_path}", file=sys.stderr)
        return 1

    logits_list: list[list[float]] = []
    labels_list: list[int] = []
    try:
        with open(labels_path) as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                    logits_list.append(list(record["logits"]))
                    labels_list.append(int(record["label"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    print(
                        f"error: malformed labels record on line {line_no}",
                        file=sys.stderr,
                    )
                    return 1
    except OSError as exc:
        print(f"error: could not read labels: {type(exc).__name__}", file=sys.stderr)
        return 1

    if not logits_list:
        print("error: labels file is empty", file=sys.stderr)
        return 1

    logits = np.asarray(logits_list, dtype=np.float64)
    labels = np.asarray(labels_list, dtype=np.int64)

    calibrator = TemperatureScaling()
    calibrator.fit(logits, labels)
    probs = calibrator.calibrate(logits)
    ece = compute_ece(probs, labels)

    result: dict[str, Any] = {
        "strategy": "temperature_scaling",
        "T": float(calibrator.T),
        "ece": float(ece),
        "n_examples": int(len(labels)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote calibration to {out_path} (T={result['T']:.4f}, ECE={result['ece']:.4f})")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Run L1 (and optionally L2) against a fixtures YAML and print a report.

    Soft diagnostic: produces a per-fixture summary listing detected
    flags, expected flags (if given), and a pass/fail indicator. No
    hard numeric gate -- the return code is 0 when every fixture's
    expected flags were detected, 1 otherwise. This matches the
    Phase 1 `test-patterns` CLI style.
    """
    import yaml
    from bh_sentinel.core.models.request import AnalysisConfig
    from bh_sentinel.core.models.response import AnalysisResponse
    from bh_sentinel.core.pipeline import Pipeline

    fixture_path: Path = args.corpus if args.corpus else args.fixtures
    if not fixture_path.exists():
        print(f"error: fixtures file not found: {fixture_path}", file=sys.stderr)
        return 1

    try:
        data = yaml.safe_load(fixture_path.read_text())
    except yaml.YAMLError as exc:
        print(f"error: could not parse YAML: {type(exc).__name__}", file=sys.stderr)
        return 1

    fixtures = data.get("fixtures") or data.get("entries") or []
    if not isinstance(fixtures, list):
        print("error: fixtures file must contain a list", file=sys.stderr)
        return 1

    try:
        pipeline = Pipeline(enable_transformer=args.enable_transformer)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    all_passed = True
    for fx in fixtures:
        fx_id = fx.get("id", "<no id>")
        text = fx.get("text", "")
        expected = set(fx.get("expect_flags") or fx.get("expected_flags_hint") or [])
        if not text:
            print(f"- {fx_id}: SKIP (no text)")
            continue

        response = pipeline.analyze_sync(text, AnalysisConfig(min_severity="LOW"))
        if not isinstance(response, AnalysisResponse):
            print(f"- {fx_id}: ERROR (pipeline returned error response)")
            all_passed = False
            continue

        detected = {f.flag_id for f in response.flags}
        detected |= {f.flag_id for f in response.protective_factors}

        missing = expected - detected if expected else set()
        passes = not missing
        status = "PASS" if passes else "GAP "
        if not passes:
            all_passed = False

        expected_str = ",".join(sorted(expected)) if expected else "-"
        detected_str = ",".join(sorted(detected)) if detected else "-"
        missing_str = ",".join(sorted(missing)) if missing else "-"
        print(
            f"- {fx_id}: {status}  "
            f"expected=[{expected_str}]  "
            f"detected=[{detected_str}]  "
            f"missing=[{missing_str}]"
        )

    return 0 if all_passed else 1


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
