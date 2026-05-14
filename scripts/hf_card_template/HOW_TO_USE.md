# How to use the HF model card template

These two files are the **template** for the HuggingFace Hub model card that ships with the canonical ONNX artifact under `bh-healthcare/distilbart-mnli-12-3-int8-onnx`. They are version-controlled here so the model card stays in sync with `docs/ml-artifact-provenance.md` across releases.

## When to use

You're here because you're running [Phase 2 of the `bh-sentinel-ml v0.2.1` release plan](../../../.cursor/plans/bh-sentinel-ml_0.2.1_release_v2_e0c9f464.plan.md): exporting the ONNX artifact and uploading it to HF.

## Workflow

After running `python scripts/export_onnx.py ...` (which writes everything except the model card to `./artifact_staging/`):

1. Copy both template files into the artifact dir:
   ```bash
   cp scripts/hf_card_template/README.md  artifact_staging/README.md
   cp scripts/hf_card_template/LICENSE    artifact_staging/LICENSE
   ```

2. Read `artifact_staging/manifest.json` — that has every value you need to substitute.

3. Find and replace the `{{PLACEHOLDER}}` tokens in `artifact_staging/README.md` with the corresponding `manifest.json` values:

   | Placeholder | Source |
   |---|---|
   | `{{SOURCE_REVISION}}` | `manifest.source_revision` |
   | `{{OPTIMUM_VERSION}}` | `manifest.tooling.optimum` |
   | `{{ONNXRUNTIME_VERSION}}` | `manifest.tooling.onnxruntime` |
   | `{{ONNX_VERSION}}` | `manifest.tooling.onnx` |
   | `{{EXPORT_DATE}}` | `manifest.export_date_utc` (first 10 chars: YYYY-MM-DD) |
   | `{{ONNX_SHA256}}` | `manifest.sha256` |
   | `{{FILE_SIZE_MB}}` | `ls -lh artifact_staging/model_int8.onnx \| awk '{print $5}'` — paste the human-readable size |

   On macOS:
   ```bash
   python3 -c "
   import json, os, math
   m = json.load(open('artifact_staging/manifest.json'))
   size_mb = math.ceil(os.path.getsize('artifact_staging/model_int8.onnx') / (1024*1024))
   subs = {
       '{{SOURCE_REVISION}}': m['source_revision'],
       '{{OPTIMUM_VERSION}}': m['tooling']['optimum'],
       '{{ONNXRUNTIME_VERSION}}': m['tooling']['onnxruntime'],
       '{{ONNX_VERSION}}': m['tooling']['onnx'],
       '{{EXPORT_DATE}}': m['export_date_utc'][:10],
       '{{ONNX_SHA256}}': m['sha256'],
       '{{FILE_SIZE_MB}}': str(size_mb),
   }
   text = open('artifact_staging/README.md').read()
   for k, v in subs.items():
       text = text.replace(k, v)
   open('artifact_staging/README.md', 'w').write(text)
   print('Substituted', len(subs), 'placeholders')
   "
   ```

4. Read the resulting `artifact_staging/README.md` end-to-end and edit any wording that needs adjustment for the specific release. The template is intentionally generic.

5. Confirm no `{{...}}` placeholders remain:
   ```bash
   rg '\{\{[A-Z_]+\}\}' artifact_staging/README.md && echo "FAIL" || echo "PASS"
   ```

6. Banned-string check — run the regex documented in [`.cursor/rules/no-real-org-names.mdc`](../../.cursor/rules/no-real-org-names.mdc) against the staging dir. Must return zero hits.

7. Proceed to Phase 2 step 4 of the release plan (`hf upload`).

## Editing this template

When the model card structure needs to evolve (new sections, updated trade-offs, new release version), edit `README.md` and `LICENSE` here in version control. Bump the relevant section of `docs/ml-artifact-provenance.md` in the same commit so the docs stay aligned.

The `LICENSE` file is paired tightly with the source model. If a future release pins a different upstream source, update the copyright attributions in `LICENSE` (currently Meta Platforms, Inc. for `facebook/bart-large-mnli`).
