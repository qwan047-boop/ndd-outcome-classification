"""Verify the frozen public CDP evidence against the original batch files offline."""
import hashlib
import json
from collections import Counter

from reproduce import P, csv_rows, read, sha, verify_files


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def main():
    verified_files = verify_files()
    d = P / "data/llm"
    evidence = csv_rows(d / "CDP_MESSAGE_MODEL_EVIDENCE.csv")
    batches = csv_rows(d / "BATCH_PROVENANCE_10.csv")
    records = csv_rows(d / "RECORD_PROVENANCE_200.csv")
    assert len(evidence) == 11 and len(batches) == 10 and len(records) == 200
    successful = {r["public_message_key"]: r for r in evidence if int(r["result_count"]) > 0}
    failed = [r for r in evidence if int(r["result_count"]) == 0]
    assert len(successful) == 10 and len(failed) == 1
    assert failed[0]["original_batch_id"] == "CHATGPT_BATCH_03"
    assert failed[0]["archive_comparison"] == "FAILED_EMPTY_RESULTS_NO_LABELS"
    assert all(r["fixed_backend_snapshot"] == "UNAVAILABLE" and r["status"] == "VERIFIED" for r in evidence)
    assert all(r["model_slug"] == r["conversation_default_model_slug"] == "gpt-6.1-sol-wm" for r in evidence)
    assert len({r["Record_ID"] for r in records}) == 200
    for batch in batches:
        key = batch["Public_Message_Key"]
        m = successful[key]
        assert batch["Original_Batch_ID"] == m["original_batch_id"]
        assert batch["Backend_Snapshot"] == "UNAVAILABLE" and batch["CDP_Source_Status"] == "VERIFIED"
        assert m["archive_comparison"] == "CANONICAL_FULL_JSON_MATCH"
        source = P / batch["Raw_Output_Package_Path"]
        assert sha(source) == batch["Raw_Output_SHA256"]
        raw = read(source)
        assert canonical(raw) == m["canonical_live_json_sha256"]
        assert raw["Prompt_Version"] == m["prompt_version"] == batch["Prompt_Version"]
        arr = [r for r in records if r["Public_Message_Key"] == key]
        assert len(arr) == 20
        assert all(r["Message_Model_Slug"] == m["model_slug"] and r["CDP_Source_Status"] == "VERIFIED" for r in arr)
        ids = [[r["Record_ID"],r["NCT_ID"]] for r in arr]
        assert ids == [[r["Record_ID"],r["NCT_ID"]] for r in raw["Results"]]
        assert canonical(ids) == m["record_nct_order_sha256"]
    result = {"verified_package_files": verified_files, "verified_successes": len(successful),
              "failed_empty_attempts":len(failed), "verified_record_links":len(records),
              "verified_messages_by_round": dict(Counter(x["round"] for x in successful.values())),
              "fixed_backend_snapshot":"UNAVAILABLE"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
