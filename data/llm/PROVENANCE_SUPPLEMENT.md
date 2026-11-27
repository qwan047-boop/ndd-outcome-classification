# Offline verification of frozen LLM sources

**VERIFIED** evidence covers the prompt, spreadsheets' hashed batch sources, the ten successful raw JSON outputs, the 200 ordered Record_ID/NCT_ID links, transcript fingerprints, and saved message-level model metadata. **UNAVAILABLE** applies to the exact fixed backend snapshot. The package can verify archived labels and their provenance. It cannot regenerate the historical web outputs from a fixed backend.

- `CDP_MESSAGE_MODEL_EVIDENCE.csv` contains ten successful assistant responses and one empty first-round attempt. Neutral evidence keys link to both mapping tables. Times are stored assistant message creation times. Both conversation defaults and all eleven recorded message-level slugs are `gpt-6.1-sol-wm`. The failed attempt produced zero labels.
- `BATCH_PROVENANCE_10.csv` links the ten successful batches to original response JSON, workbook hashes, the prompt hash and version, PDF start-page anchors and their message evidence keys.
- `RECORD_PROVENANCE_200.csv` associates every original record with a batch and the model slug recorded for its producing message. Original label and quality states are preserved.
- `BATCH_INPUTS_200.json` preserves the six research input fields in original spreadsheet order. The workbooks themselves remain private.
- `PRIVATE_TRANSCRIPT_FINGERPRINTS.json` contains the 63-page and 60-page PDF fingerprints. These image-based transcripts corroborate visual batch anchors. Canonical response comparisons against the original JSON establish structured output equality. The PDFs, technical message IDs, parent IDs and any conversation links remain in separate private custody.
- `prompt.md` and the prompt reproduced in `docs/AI_DISCLOSURE_APPENDIX.md` preserve the frozen instructions. All 786 quoted evidence spans are checked against the archived inputs by the existing provenance verifier.

Run the two checks below from the package directory. They read only package files and do not connect to ChatGPT.

```sh
python -B verify_llm_provenance.py
python -B verify_cdp_freeze.py
```
