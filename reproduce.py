"""Run the release candidate offline. All results go to a new output directory."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

P = Path(__file__).resolve().parent
B = P / "history/baseline"
G = P / "history/weight_grid"
D = P / "data"
MODELS = ("tfidf_svm", "minilm_lr")
METRICS = ("micro_f1", "macro_f1", "exact_match")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    return {q.relative_to(P).as_posix(): sha(q) for q in sorted(P.rglob("*"))
            if q.is_file() and q.relative_to(P).as_posix() not in {"SHA256_MANIFEST.json", "SHA256SUMS.txt"}
            and not {".git", "__pycache__"} & set(q.relative_to(P).parts)}


def verify_files():
    expected = read(P / "SHA256_MANIFEST.json")
    assert files() == expected, "The checked package set or SHA-256 changed."
    entries = {**expected, "SHA256_MANIFEST.json": sha(P / "SHA256_MANIFEST.json")}
    checksum_lines = "".join(f"{value}  {name}\n" for name, value in sorted(entries.items()))
    assert (P / "SHA256SUMS.txt").read_bytes() == checksum_lines.encode("utf-8")
    return len(expected)


def inputs():
    return (read(B / "EXPERIMENT_CONFIG.yaml"), read(D / "locked_partitions.json"),
            {r["outcome_id"]: r for r in read(D / "pediatric_outcomes.json")},
            {r["record_id"]: r["rule_labels"] for r in read(D / "initial_rule_targets_1934.json")})


def verify_archive_data():
    """Check only released records; do not infer unavailable text checks."""
    cfg = read(B / "EXPERIMENT_CONFIG.yaml")
    parts = read(D / "locked_partitions.json")
    train = read(D / "initial_rule_targets_1934.json")
    assert len(train) == len({r["record_id"] for r in train}) == 1934
    ids, test, pool = (set(parts[f"{key}_ids"]) for key in ("training", "test", "pool"))
    assert (len(ids), len(test), len(pool)) == (3038, 60, 40)
    assert {r["record_id"] for r in train} <= ids and pool <= ids and not test & ids
    ncts = lambda records: {i.split(":")[0] for i in records}
    assert not ncts(ids) & ncts(test) and not ncts(pool) & ncts(test)
    human = csv_rows(B / "HUMAN_REFERENCE_FINAL_FROZEN.csv")
    assert sha(B / "HUMAN_REFERENCE_FINAL_FROZEN.csv") == cfg["human_reference_sha256"]
    assert len(human) == len({r["Record_ID"] for r in human}) == 100
    assert {r["Record_ID"] for r in human if r["Partition"] == "Test_60"} == test
    assert {r["Record_ID"] for r in human if r["Partition"] == "Pool_40"} == pool
    assert sum(r["Current_Author_Label"] != "UNCLEAR" for r in human if r["Partition"] == "Test_60") == 58
    for row in human:
        assert set(row["Current_Author_Label"].split(";")) <= set(cfg["labels"]) | {"UNCLEAR"}
    llm = read(D / "llm/LLM_PREDICTIONS_200_MERGED.json")["records"]
    llm_ids = {r["Record_ID"] for r in llm}
    assert len(llm) == len(llm_ids) == 200 and llm_ids <= ids
    assert not ncts(llm_ids) & ncts(test | pool)
    originals = {}
    for prefix in ("original", "expansion"):
        for i in range(1, 6):
            records = read(D / f"llm/{prefix}_{i:02}.json")["Results"]
            assert len(records) == 20 and not set(originals) & {r["Record_ID"] for r in records}
            originals.update((r["Record_ID"], r) for r in records)
    assert all(originals[r["Record_ID"]] == r["Original_LLM_Prediction"] for r in llm)
    edges = csv_rows(D / "near_duplicate_pairs.csv")
    assert len(edges) == 1849
    assert not any((e["outcome_a"] in test and e["outcome_b"] in ids) or
                   (e["outcome_b"] in test and e["outcome_a"] in ids) for e in edges)
    absent = [name for name in ("pediatric_outcomes.json", "registry_source_extract_670.json")
              if not (D / name).exists()]
    result = dict(status="PASS_WITH_LIMITATIONS" if absent else "PASS",
                  initial_train=1934, test=60, decidable_test=58, pool=40, llm=200,
                  released_record_checks="PASS", missing_source_files=absent,
                  raw_text_and_full_component_checks="BLOCKED" if absent else "SEE_FULL_SOURCE_CHECK")
    if not absent:
        result["full_source_checks"] = verify_data()
    return result


def verify_archive_data():
    """Check only released records; do not infer unavailable text checks."""
    cfg = read(B / "EXPERIMENT_CONFIG.yaml")
    parts = read(D / "locked_partitions.json")
    train = read(D / "initial_rule_targets_1934.json")
    assert len(train) == len({r["record_id"] for r in train}) == 1934
    ids, test, pool = (set(parts[f"{key}_ids"]) for key in ("training", "test", "pool"))
    assert (len(ids), len(test), len(pool)) == (3038, 60, 40)
    assert {r["record_id"] for r in train} <= ids and pool <= ids and not test & ids
    ncts = lambda records: {i.split(":")[0] for i in records}
    assert not ncts(ids) & ncts(test) and not ncts(pool) & ncts(test)
    human = csv_rows(B / "HUMAN_REFERENCE_FINAL_FROZEN.csv")
    assert sha(B / "HUMAN_REFERENCE_FINAL_FROZEN.csv") == cfg["human_reference_sha256"]
    assert len(human) == len({r["Record_ID"] for r in human}) == 100
    assert {r["Record_ID"] for r in human if r["Partition"] == "Test_60"} == test
    assert {r["Record_ID"] for r in human if r["Partition"] == "Pool_40"} == pool
    assert sum(r["Current_Author_Label"] != "UNCLEAR" for r in human if r["Partition"] == "Test_60") == 58
    for row in human:
        assert set(row["Current_Author_Label"].split(";")) <= set(cfg["labels"]) | {"UNCLEAR"}
    llm = read(D / "llm/LLM_PREDICTIONS_200_MERGED.json")["records"]
    llm_ids = {r["Record_ID"] for r in llm}
    assert len(llm) == len(llm_ids) == 200 and llm_ids <= ids
    assert not ncts(llm_ids) & ncts(test | pool)
    originals = {}
    for prefix in ("original", "expansion"):
        for i in range(1, 6):
            records = read(D / f"llm/{prefix}_{i:02}.json")["Results"]
            assert len(records) == 20 and not set(originals) & {r["Record_ID"] for r in records}
            originals.update((r["Record_ID"], r) for r in records)
    assert all(originals[r["Record_ID"]] == r["Original_LLM_Prediction"] for r in llm)
    edges = csv_rows(D / "near_duplicate_pairs.csv")
    assert len(edges) == 1849
    assert not any((e["outcome_a"] in test and e["outcome_b"] in ids) or
                   (e["outcome_b"] in test and e["outcome_a"] in ids) for e in edges)
    absent = [name for name in ("pediatric_outcomes.json", "registry_source_extract_670.json")
              if not (D / name).exists()]
    result = dict(status="PASS_WITH_LIMITATIONS" if absent else "PASS",
                  initial_train=1934, test=60, decidable_test=58, pool=40, llm=200,
                  released_record_checks="PASS", missing_source_files=absent,
                  raw_text_and_full_component_checks="BLOCKED" if absent else "SEE_FULL_SOURCE_CHECK")
    if not absent:
        result["full_source_checks"] = verify_data()
    return result


def verify_data():
    config, parts, outcomes, train = inputs()
    assert len(outcomes) == 3412 and len(train) == 1934
    assert train == {x["record_id"]: x["rule_labels"] for x in read(D / "training_weak_labels.json")
                    if x["rule_status"] == "Rule_candidate"}
    source = {r["nct"]: r for r in read(D / "registry_source_extract_670.json")}
    assert len(source) == 670
    assert sum(sum(map(len, r["registry_outcomes"].values())) for r in source.values()) == 4653
    eligible = {i for i, r in source.items()
                if r["frozen_eligibility_and_roles"]["Eligibility status"] == "Eligible_structured"}
    assert eligible == {r["nct"] for r in outcomes.values()} and len(eligible) == 463
    for i, r in outcomes.items():
        nct, kind, index = i.split(":")
        source_row = source[nct]
        raw = source_row["registry_outcomes"][kind][int(index)]
        assert all(r[k] == raw.get(v, "") for k, v in
                   (("measure", "measure"), ("description", "description"), ("timeframe", "timeFrame")))
        assert r["source_hash"] == source_row["source_sha256"]
    ids, test, pool = (set(parts[f"{key}_ids"]) for key in ("training", "test", "pool"))
    assert len(ids) == 3038 and len(test) == 60 and len(pool) == 40
    assert set(train) <= ids and pool <= ids and not (test & ids)
    human = csv_rows(B / "HUMAN_REFERENCE_FINAL_FROZEN.csv")
    assert sha(B / "HUMAN_REFERENCE_FINAL_FROZEN.csv") == config["human_reference_sha256"]
    assert len(human) == len({r["Record_ID"] for r in human}) == 100
    assert {r["Record_ID"] for r in human if r["Partition"] == "Test_60"} == test
    assert {r["Record_ID"] for r in human if r["Partition"] == "Pool_40"} == pool
    assert sum(r["Current_Author_Label"] != "UNCLEAR" for r in human if r["Partition"] == "Test_60") == 58
    llm = read(D / "llm/LLM_PREDICTIONS_200_MERGED.json")["records"]
    llm_ids = {r["Record_ID"] for r in llm}
    assert len(llm) == len(llm_ids) == 200 and llm_ids <= ids
    ledger = {}
    for prefix in ("original", "expansion"):
        for i in range(1, 6):
            items = read(D / f"llm/{prefix}_{i:02}.json")["Results"]
            assert len(items) == 20 and not (set(ledger) & {x["Record_ID"] for x in items})
            ledger.update((x["Record_ID"], x) for x in items)
    assert all(ledger[r["Record_ID"]] == r["Original_LLM_Prediction"] for r in llm)
    for field in ("nct", "strict_split_group", "normalized_hash"):
        values = lambda group: {outcomes[i][field] for i in group}
        assert not values(ids) & values(test)
        assert not values(llm_ids) & values(test | pool)
    edges = csv_rows(D / "near_duplicate_pairs.csv")
    assert len(edges) == 1849
    assert not any((e["outcome_a"] in test and e["outcome_b"] in ids) or
                   (e["outcome_b"] in test and e["outcome_a"] in ids) for e in edges)
    return {"source_nct": len(source), "eligible_outcomes": len(outcomes), "initial_train": len(train),
            "test": len(test), "decidable_test": 58, "pool": len(pool), "llm": len(llm),
            "qc": dict(Counter(r["QC_Status"] for r in llm)), "near_duplicate_edges": len(edges)}


def child(script, *args):
    subprocess.run([sys.executable, "-B", "-X", "utf8", str(P / "code" / script), *map(str, args)],
                   check=True, cwd=Path.cwd())


def seed_run(out, prediction_archive=False):
    for name in ("EXPERIMENT_CONFIG.yaml", "HUMAN_REFERENCE_FINAL_FROZEN.csv"):
        shutil.copyfile(B / name, out / name)
    if prediction_archive:
        for model in MODELS:
            for suffix in ("PREDICTIONS", "QUERY_RELEASE_LOG", "TRAINING_SNAPSHOTS"):
                name = f"{model}_{suffix}.json"
                shutil.copyfile(B / name, out / name)


def compare_csv(a, b, keys, fields):
    import numpy as np
    key = lambda row: tuple(row[k] for k in keys)
    left = {key(r): r for r in csv_rows(a)}
    right = {key(r): r for r in csv_rows(b)}
    assert left.keys() == right.keys(), (a.name, len(left), len(right))
    for k, r in left.items():
        for f in fields:
            x, y = r[f], right[k][f]
            if x or y:
                assert np.isclose(float(x), float(y), rtol=0, atol=1e-12), (a.name, k, f, x, y)
    return len(left)*len(fields)


def encode(labels, rows):
    import numpy as np
    return np.array([[int(label in tags) for label in labels] for tags in rows])


def numeric(y, p):
    from sklearn.metrics import accuracy_score, f1_score
    return [f1_score(y, p, average="micro", zero_division=0),
            f1_score(y, p, average="macro", zero_division=0), accuracy_score(y, p)]


def boot_metrics(y, p, boot):
    import numpy as np
    y, p = y[boot], p[boot]
    tp = (y*p).sum(1)
    den = 2*tp + ((1-y)*p).sum(1) + (y*(1-p)).sum(1)
    micro = np.divide(2*tp.sum(1), den.sum(1), out=np.zeros(len(boot)), where=den.sum(1) != 0)
    macro = np.divide(2*tp, den, out=np.zeros_like(den, dtype=float), where=den != 0).mean(1)
    return np.stack([micro, macro, (y == p).mean((1, 2)), (y == p).all(axis=2).mean(1)], axis=1)


def evaluate_archive(out):
    import numpy as np
    seed_run(out, prediction_archive=True)
    child("evaluate_engine.py", out)
    keys = ("model", "seed", "strategy", "budget")
    selected = ("micro_f1", "macro_f1", "exact_match", "micro_ci_low", "micro_ci_high",
                "macro_ci_low", "macro_ci_high", "reference_decidable_n")
    comparisons = {name: compare_csv(out / name, B / name, keys, selected)
                   for name in ("BASELINE_RESULTS.csv", "ACTIVE_LEARNING_RESULTS.csv")}
    comparisons["CLASS_WISE_METRICS.csv"] = compare_csv(out / "CLASS_WISE_METRICS.csv",
        B / "CLASS_WISE_METRICS.csv", (*keys, "label"), ("precision", "recall", "f1", "support"))
    config = read(B / "EXPERIMENT_CONFIG.yaml")
    parts = read(D / "locked_partitions.json")
    labels = config["labels"]
    gold = {r["Record_ID"]: r["Current_Author_Label"].split(";")
            for r in csv_rows(B / "HUMAN_REFERENCE_FINAL_FROZEN.csv")}
    valid = [i for i in parts["test_ids"] if gold[i] != ["UNCLEAR"]]
    y = encode(labels, [gold[i] for i in valid])
    boot = np.random.default_rng(20261124).integers(0, 58, size=(2000, 58))
    manuscript_checks = []
    resampled = {}
    binary_predictions = {}
    for old in csv_rows(B / "MODEL_METRICS_WITH_UNCERTAINTY.csv"):
        model = old["model"]
        if model not in MODELS:
            continue
        pred = baseline_predictions(model)
        z = encode(labels, [pred[i]["predicted_labels"] for i in valid])
        binary_predictions[model] = z
        resampled[model] = boot_metrics(y, z, boot)
        q = np.quantile(resampled[model], [.025, .975], axis=0)
        for j, metric in enumerate(("micro_f1", "macro_f1", "hamming_loss", "exact_match")):
            actual = q[:, j] if metric != "hamming_loss" else 1-q[::-1, j]
            expected = [float(old[metric+"_ci_low"]), float(old[metric+"_ci_high"])]
            assert np.allclose(actual, expected, rtol=0, atol=1e-12)
            manuscript_checks.append({"model": model, "metric": metric, "interval": actual.tolist()})
    paired = read(B / "COMPLEMENTARITY_SUMMARY.json")
    contrast = resampled[MODELS[0]] - resampled[MODELS[1]]
    for j, metric in enumerate(("micro", "macro")):
        assert np.allclose(np.quantile(contrast[:, j], [.025, .975]), paired[metric+"_CI95"], rtol=0, atol=1e-12)
    per_class_checks = 0
    for row in csv_rows(B / "CLASS_CONFUSION_AND_INTERVALS.csv"):
        col = labels.index(row["class"])
        v, z = y[:, col], binary_predictions[row["model"]][:, col]
        actual = [int((v*z).sum()), int(((1-v)*z).sum()), int((v*(1-z)).sum()), int(((1-v)*(1-z)).sum())]
        assert actual == [int(row[k]) for k in ("TP","FP","FN","TN")]
        draw_v, draw_z = v[boot], z[boot]
        tp = (draw_v*draw_z).sum(1)
        denom = draw_v.sum(1)+draw_z.sum(1)
        interval = np.quantile(np.divide(2*tp,denom,out=np.zeros(len(boot)),where=denom!=0), [.025,.975])
        assert np.allclose(interval,[float(row["f1_ci_low"]),float(row["f1_ci_high"])],rtol=0,atol=1e-12)
        per_class_checks += 1
    write(out / "ARCHIVE_VERIFICATION.json", dict(comparisons=comparisons,
        manuscript_2000_draw_intervals=manuscript_checks, paired_representation_intervals_verified=2,
        class_confusion_and_intervals_verified=per_class_checks))
    return comparisons


def baseline_predictions(model):
    return {r["Record_ID"]: r for r in read(B / f"{model}_PREDICTIONS.json")
            if (r["seed"], r["strategy"], r["budget"]) == (20261123, "random", 0)}


def score_bundle(bundle, x):
    import numpy as np
    return np.stack([m.decision_function(x) if bundle["backend"] == "tfidf_svm"
                     else 2*m.predict_proba(x)[:, 1]-1 for m in bundle["estimators"]], axis=1)


def baseline_matrices(*, fit_vectorizer=True):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    cfg, parts, outcomes, weak = inputs()
    all_ids = parts["training_ids"] + parts["test_ids"]
    features = [outcomes[i]["measure"]+"\n"+outcomes[i]["description"] for i in all_ids]
    vectorizer = None
    if fit_vectorizer:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        vectorizer.fit(features[:len(parts["training_ids"])])
        assert len(vectorizer.vocabulary_) == read(B / "TFIDF_FEATURE_PROVENANCE.json")["vocabulary_n"]
    dense = np.load(B / "MINILM_EMBEDDINGS.npy", allow_pickle=False)
    assert all_ids == read(B / "MINILM_FEATURE_PROVENANCE.json")["ids"]
    assert dense.shape == (len(all_ids), 384) and np.isfinite(dense).all()
    return cfg, parts, weak, all_ids, features, vectorizer, dense


def reconstruct(out):
    import joblib
    import numpy as np
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    cfg, parts, weak, all_ids, texts, vec, dense = baseline_matrices()
    loc = {i:j for j,i in enumerate(all_ids)}
    train = sorted(weak)
    targets = encode(cfg["labels"], [weak[i] for i in train])
    results = []
    for model, mat in zip(MODELS, (vec.transform(texts), dense)):
        estimators = []
        for col in range(targets.shape[1]):
            solver = (LinearSVC(C=1, class_weight="balanced", random_state=20261123)
                      if model == MODELS[0] else LogisticRegression(
                          C=1, class_weight="balanced", max_iter=2000, random_state=20261123))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                solver.fit(mat[[loc[i] for i in train]], targets[:, col])
            estimators.append(solver)
        bundle = dict(status="RECONSTRUCTED_MODEL", backend=model, estimators=estimators,
                      vectorizer=vec if model == MODELS[0] else None, labels=cfg["labels"],
                      sklearn=sklearn.__version__, seed=20261123, classification_threshold=0.)
        snapshot = out / f"RECONSTRUCTED_MODEL_{model}.joblib"
        with snapshot.open("xb") as stream:
            joblib.dump(bundle, stream, compress=3)
        loaded = joblib.load(snapshot)
        indexes = [loc[i] for i in parts["test_ids"]]
        test = loaded["vectorizer"].transform([texts[j] for j in indexes]) if model == MODELS[0] else mat[indexes]
        pred = score_bundle(loaded, test)
        assert np.array_equal(score_bundle(bundle, test), pred)
        hist = baseline_predictions(model)
        old = np.array([hist[i]["scores"] for i in parts["test_ids"]])
        history_hard = encode(cfg["labels"], [hist[i]["predicted_labels"] for i in parts["test_ids"]])
        assert np.array_equal(old > 0, history_hard)
        changes = [i for i, a, b in zip(parts["test_ids"], pred > 0, history_hard) if not np.array_equal(a,b)]
        results.append(dict(model=model, status=bundle["status"], n_training=len(train), n_compared=len(old),
                            changed_records=changes, max_abs_score_delta=float(np.max(np.abs(pred-old))),
                            snapshot=snapshot.name, sha256=sha(snapshot), reload_equal=True))
        write(out / f"reconstructed_{model}_predictions.json", [
            dict(Record_ID=i, scores=s.tolist(), predicted_labels=[l for l,v in zip(cfg["labels"],s) if v>0])
            for i,s in zip(parts["test_ids"], pred)])
    write(out / "MODEL_RECONSTRUCTION.json", results)
    return results


def load_models(out):
    """Load both bundles; score only inputs actually distributed."""
    import joblib
    import numpy as np
    cfg = read(B / "EXPERIMENT_CONFIG.yaml")
    parts = read(D / "locked_partitions.json")
    prov = read(B / "MINILM_FEATURE_PROVENANCE.json")
    dense = np.load(B / "MINILM_EMBEDDINGS.npy", allow_pickle=False)
    assert prov["ids"] == parts["training_ids"] + parts["test_ids"]
    assert dense.shape == (len(prov["ids"]), 384) and np.isfinite(dense).all()
    ix = [prov["ids"].index(i) for i in parts["test_ids"]]
    outcomes_path = D / "pediatric_outcomes.json"
    outcomes = {r["outcome_id"]: r for r in read(outcomes_path)} if outcomes_path.exists() else None
    result = []
    for model in MODELS:
        bundle = joblib.load(P / "models" / f"RECONSTRUCTED_MODEL_{model}.joblib")
        assert bundle["status"] == "RECONSTRUCTED_MODEL" and bundle["labels"] == cfg["labels"]
        row = dict(model=model, artifact_status=bundle["status"], deserialization="PASS")
        if model == "tfidf_svm" and outcomes is None:
            assert bundle["vectorizer"] is not None
            assert len(bundle["estimators"]) == len(cfg["labels"])
            row.update(inference_status="BLOCKED_MISSING_SOURCE_TEXT",
                       reason="Frozen test text is withheld; no SVM inference or agreement claimed.")
            result.append(row)
            continue
        if model == "tfidf_svm":
            texts = [outcomes[i]["measure"] + "\n" + outcomes[i]["description"] for i in parts["test_ids"]]
            test = bundle["vectorizer"].transform(texts)
        else:
            test = dense[ix]
        scores = score_bundle(bundle, test)
        archive = baseline_predictions(model)
        old = np.array([archive[i]["scores"] for i in parts["test_ids"]])
        changes = [i for i, a, b in zip(parts["test_ids"], scores > 0, old > 0) if not np.array_equal(a, b)]
        assert not changes, f"{model}: reconstructed hard predictions differ from archive."
        row.update(inference_status="PASS", n=len(scores), hard_prediction_differences=changes,
                   input_source="cached_embeddings" if model == "minilm_lr" else "frozen_source_text",
                   max_abs_score_delta=float(np.abs(scores-old).max()))
        result.append(row)
    write(out / "PACKAGED_MODEL_LOAD.json", result)
    return result


def replay(out):
    import numpy as np
    seed_run(out)
    for model in MODELS:
        child("replay_engine.py", out, model)
    child("evaluate_engine.py", out)
    result = []
    for model in MODELS:
        old = read(B / f"{model}_PREDICTIONS.json")
        new = read(out / f"{model}_PREDICTIONS.json")
        assert len(old) == len(new) == 4500
        keys = ("Record_ID", "strategy", "seed", "budget", "training_n", "human_trainable_n")
        assert all(all(a[k] == b[k] for k in keys) for a, b in zip(old, new))
        delta = np.abs(np.array([a["scores"] for a in new])-np.array([a["scores"] for a in old]))
        q0 = read(B / f"{model}_QUERY_RELEASE_LOG.json")
        q1 = read(out / f"{model}_QUERY_RELEASE_LOG.json")
        assert len(q0) == len(q1) == 60
        result.append(dict(model=model, prediction_rows=len(old),
                           changed_hard_rows=sum(a["predicted_labels"] != b["predicted_labels"] for a,b in zip(old,new)),
                           max_abs_score_delta=float(delta.max()), changed_query_batches=sum(a!=b for a,b in zip(q0,q1)),
                           snapshots_match=read(B/f"{model}_TRAINING_SNAPSHOTS.json")==read(out/f"{model}_TRAINING_SNAPSHOTS.json")))
    write(out / "REPLAY_COMPARISON.json", result)
    return result


def weight_check(out):
    child("weight_grid_engine.py", out)
    keys = ("model", "condition", "scope")
    summary = compare_csv(out / "WEIGHT_SENSITIVITY_RESULTS.csv", G / "WEIGHT_SENSITIVITY_RESULTS.csv", keys,
                          METRICS+tuple(m+"_fold_"+stat for m in METRICS for stat in ("mean","sd","min","max"))
                          +tuple("delta_"+m+suffix for m in METRICS for suffix in ("","_ci_low","_ci_high")))
    folds = compare_csv(out / "GROUP_CV_RESULTS.csv", G / "GROUP_CV_RESULTS.csv",
        ("model","condition","fold"), METRICS+("changed_records","changed_label_decisions"))
    result = dict(summary_fields_checked=summary, fold_fields_checked=folds)
    write(out / "PACKAGE_WEIGHT_COMPARISON.json", result)
    return result


def rerun_encoder(out, path):
    import os
    import numpy as np
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    provenance = read(B / "MINILM_FEATURE_PROVENANCE.json")
    for name, digest in provenance["model_files"].items():
        assert sha(path / name.replace("\\", "/")) == digest, name
    prov = {r["outcome_id"]:r for r in read(D / "pediatric_outcomes.json")}
    features = [prov[i]["measure"]+"\n"+prov[i]["description"] for i in provenance["ids"]]
    sys.path.insert(0, str(P / "code"))
    from encoder import encode as run_encoder
    x = run_encoder(features, path, log=lambda s:print(s,flush=True))
    archived = np.load(B/"MINILM_EMBEDDINGS.npy", allow_pickle=False)
    result = dict(rows=len(x), dimension=x.shape[1], max_abs_feature_delta=float(np.abs(x-archived).max()),
                  exact_feature_match=bool(np.array_equal(x,archived)))
    # Refit the downstream LR on the regenerated features before assessing historical predictions.
    from sklearn.linear_model import LogisticRegression
    config, parts, _, weak = inputs()
    indexes = {i:j for j,i in enumerate(provenance["ids"])}
    tr = [indexes[i] for i in sorted(weak)]
    ev = [indexes[i] for i in parts["test_ids"]]
    targets = encode(config["labels"], [weak[i] for i in sorted(weak)])
    scores = np.stack([2*LogisticRegression(C=1, class_weight="balanced", max_iter=2000,
                                            random_state=20261123).fit(x[tr], targets[:,j]).predict_proba(x[ev])[:,1]-1
                       for j in range(targets.shape[1])],axis=1)
    history = baseline_predictions("minilm_lr")
    old = np.array([history[i]["scores"] for i in parts["test_ids"]])
    result.update(hard_prediction_differences=[i for i, z in zip(parts["test_ids"], ((scores>0)!=(old>0)).any(1)) if z],
                  max_abs_prediction_score_delta=float(np.abs(scores-old).max()))
    write(out / "ENCODER_REGENERATION.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "metrics", "train", "load-models", "weight-grid", "replay", "encoder"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--encoder-dir", type=Path)
    args = parser.parse_args()
    hashes = verify_files()
    if args.command == "verify":
        result = {"hashes_verified": hashes, **verify_archive_data()}
    else:
        if args.out is None or P in args.out.resolve().parents or args.out.resolve() == P:
            parser.error("Provide --out as a fresh directory outside the package.")
        if args.command == "encoder" and args.encoder_dir is None:
            parser.error("encoder requires --encoder-dir containing the pinned local model files.")
        out = args.out.resolve()
        out.mkdir(parents=True, exist_ok=False)
        actions = {"metrics": evaluate_archive, "train": reconstruct, "load-models": load_models,
                   "weight-grid": weight_check, "replay": replay,
                   "encoder": lambda q:rerun_encoder(q,args.encoder_dir.resolve())}
        result = actions[args.command](out)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
