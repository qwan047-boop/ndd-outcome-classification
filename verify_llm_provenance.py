"""Verify the local LLM evidence mapping without a PDF, account or network request."""
from collections import Counter
import json

from reproduce import P, csv_rows, read, sha, verify_files


def main():
    verify_files()
    source = P / "data/llm"
    prompt = source / "prompt.md"
    batches = csv_rows(source / "BATCH_PROVENANCE_10.csv")
    mapping = csv_rows(source / "RECORD_PROVENANCE_200.csv")
    inputs = read(source / "BATCH_INPUTS_200.json")
    source_pdfs = {x["conversation_evidence_id"]:x for x in read(source / "PRIVATE_TRANSCRIPT_FINGERPRINTS.json")}
    by = {r["Record_ID"]:r for r in mapping}
    combined = {r["Record_ID"]:r for r in read(source / "LLM_PREDICTIONS_200_MERGED.json")["records"]}
    assert len(batches) == len(inputs) == 10 and len(mapping) == len(by) == len(combined) == 200
    assert by.keys() == combined.keys()
    assert {int(r["pdf_pages"]) for r in source_pdfs.values()} == {63,60}
    totals = Counter()
    for batch in batches:
        orig = read(P/batch["Raw_Output_Package_Path"])
        assert batch["Prompt_SHA256"] == sha(prompt)
        assert batch["Raw_Output_SHA256"] == sha(P/batch["Raw_Output_Package_Path"])
        assert batch["Prompt_Version"] == orig["Prompt_Version"]
        assert batch["Original_Batch_ID"] == orig["Batch_ID"]
        assert batch["PDF_SHA256"] == source_pdfs[batch["Conversation_Evidence_ID"]]["pdf_sha256"]
        batch_records = inputs[batch["Canonical_Batch_ID"]]
        assert len(batch_records) == len(orig["Results"]) == 20
        spans = 0
        for index,(record,result) in enumerate(zip(batch_records,orig["Results"]),1):
            id_ = record["Record_ID"]
            assert result["Record_ID"] == id_ and result["NCT_ID"] == record["NCT_ID"] == by[id_]["NCT_ID"]
            assert result == combined[id_]["Original_LLM_Prediction"]
            assert int(by[id_]["Input_Row_1Based"]) == index+1
            assert int(by[id_]["Output_Result_Index_1Based"]) == index
            assert by[id_]["QC_Status"] == combined[id_]["QC_Status"]
            for field in ("Round","Conversation_Evidence_ID","Canonical_Batch_ID","Original_Batch_ID",
                          "Prompt_Version","Input_XLSX_SHA256","Raw_Output_Package_Path","Raw_Output_SHA256"):
                assert by[id_][field] == batch[field]
            assert by[id_]["PDF_Batch_Output_Start_Page"] == batch["PDF_Response_Start_Page"]
            for span in result["Evidence_Span"]:
                assert span["Quote"] and span["Quote"] in record[span["Source_Field"]]
                spans += 1
        assert spans == int(batch["Verbatim_Evidence_Spans_Checked"])
        totals[batch["Round"]] += spans
    assert totals == {"original_pilot100":386,"expansion100":400}
    print(json.dumps({"verified_batches":10,"verified_records":200,"verified_source_quotes":totals,
                      "scope":"local batch files and transcript fingerprint links, not the private PDFs"},
                     indent=2))


if __name__ == "__main__":
    main()
