"""Replay the frozen sensitivity design and test a fixed five-weight grid."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits

OUT = Path(sys.argv[1]).resolve()
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT
BASE = DATA / "history/baseline"
HIST = DATA / "history/sensitivity"
MERGED = DATA / "data/llm/LLM_PREDICTIONS_200_MERGED.json"
KINDS = ("tfidf_svm", "minilm_lr")
GRID = {f"W{weight:.2f}": weight for weight in (0., .25, .5, .75, 1.)}
DELETE = "DELETE_CLASSIFIER"
REFIT = "DELETE_REFIT_TFIDF"
OLD = {"A_original": "W1.00", "B_exclude": DELETE, "C_downweight": "W0.25"}
SEED = 20261123
TZ = timezone(timedelta(hours=8))
METRICS = ("micro_f1", "macro_f1", "exact_match")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(name, value):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def write_csv(name, records):
    keys = list(dict.fromkeys(k for row in records for k in row))
    with (OUT / name).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(records)


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def log(message):
    print(now(), message, flush=True)


def counts(y, p):
    return (y*p).sum(axis=0), ((1-y)*p).sum(axis=0), (y*(1-p)).sum(axis=0)


def metrics(y, p):
    tp, fp, fn = counts(y, p)
    den = 2*tp + fp + fn
    per_class = np.divide(2*tp, den, out=np.zeros(len(tp)), where=den != 0)
    return dict(zip(METRICS, [float(2*tp.sum()/den.sum()) if den.sum() else 0.,
                              float(per_class.mean()), float(np.all(y == p, axis=1).mean())]))


def prediction_delta(scores, reference):
    changed = (scores > 0) != (reference > 0)
    return {"changed_records": int(changed.any(axis=1).sum()),
            "changed_label_decisions": int(changed.sum()),
            "max_abs_score_delta": float(np.abs(scores-reference).max()),
            "scores_close": bool(np.allclose(scores, reference, atol=1e-9, rtol=1e-7))}


def vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)


def run():
    start = now()
    input_paths = [
        DATA / "data/locked_partitions.json", DATA / "data/pediatric_outcomes.json",
        DATA / "data/training_weak_labels.json", MERGED,
        BASE / "EXPERIMENT_CONFIG.yaml", BASE / "MINILM_FEATURE_PROVENANCE.json",
        BASE / "MINILM_EMBEDDINGS.npy", BASE / "HUMAN_REFERENCE_FINAL_FROZEN.csv",
        BASE / "BASELINE_RESULTS.csv", BASE / "ACTIVE_LEARNING_RESULTS.csv",
        ROOT / "code/replay_engine.py",
        *(BASE / f"{kind}_PREDICTIONS.json" for kind in KINDS),
        *(HIST / name for name in ("FINAL_ANALYSIS_CONFIG_FROZEN.json", "NOISE_PREDICTIONS_FROZEN.json",
                                    "NOISE_TRAINING_COMPOSITION.csv", "WEAK_LABEL_NOISE_SENSITIVITY.csv",
                                    "GROUPED_CV_FOLD_ASSIGNMENTS.csv", "code/final_optimization_analysis.py")),
        Path(__file__).resolve(),
    ]
    manifest = {str(path.relative_to(ROOT)): sha(path) for path in input_paths}
    historical_runtime = read(BASE / "minilm_lr_EXECUTION.json")
    versions = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                    sklearn=sklearn.__version__)
    assert versions == {key: historical_runtime[key] for key in versions}, versions
    config = read(BASE / "EXPERIMENT_CONFIG.yaml")
    historical_config = read(HIST / "FINAL_ANALYSIS_CONFIG_FROZEN.json")
    labels = config["labels"]
    parts = read(DATA / "data/locked_partitions.json")
    outcomes = {r["outcome_id"]: r for r in read(DATA / "data/pediatric_outcomes.json")}
    weak = {r["record_id"]: r for r in read(DATA / "data/training_weak_labels.json")}
    ids = sorted(i for i, row in weak.items() if row["rule_status"] == "Rule_candidate")
    eligible, test_ids, pool_ids = (parts[k] for k in ("training_ids", "test_ids", "pool_ids"))
    assert len(ids) == 1934 and len(eligible) == 3038 and set(ids) <= set(eligible)
    assert len(set(test_ids)) == 60 and len(set(pool_ids)) == 40 and set(pool_ids) <= set(eligible)
    assert len(set(eligible) | set(test_ids)) == len(eligible) + 60
    isolation = ("nct", "strict_split_group", "normalized_hash")
    for key in isolation:
        assert not {outcomes[i][key] for i in eligible} & {outcomes[i][key] for i in test_ids}
    merged = read(MERGED)["records"]
    assert len(merged) == len({r["Record_ID"] for r in merged}) == 200
    disagreement_rows = []
    for row in merged:
        i = row["Record_ID"]
        if i not in ids or row["QC_Status"] != "NONABSTAIN_WEAK_CANDIDATE_NO_IDENTIFIED_FLAGS":
            continue
        original = row["Original_LLM_Prediction"]
        assert set(row["Rule_Labels"]) == set(weak[i]["rule_labels"])
        assert not original["Abstain"]
        if set(row["Rule_Labels"]) != set(original["Outcome_Domain"]):
            disagreement_rows.append(dict(Record_ID=i, NCT_ID=outcomes[i]["nct"],
                component=outcomes[i]["strict_split_group"], rule=";".join(row["Rule_Labels"]),
                llm=";".join(original["Outcome_Domain"]), qc=row["QC_Status"]))
    disputes = {r["Record_ID"] for r in disagreement_rows}
    assert sorted(disputes) == historical_config["conflict_ids"] and len(disputes) == 10
    for entry in rows(HIST / "NOISE_TRAINING_COMPOSITION.csv"):
        i, strategy = entry["Record_ID"], entry["strategy"]
        expected = historical_config["strategies"][strategy]["conflict_weight"] if i in disputes else 1.
        assert i in ids and float(entry["weight"]) == expected
        assert (entry["included"] == "True") == (expected > 0)
    assignment = rows(HIST / "GROUPED_CV_FOLD_ASSIGNMENTS.csv")
    fold_by_id = {r["Record_ID"]: int(r["Fold"]) for r in assignment}
    assert len(assignment) == len(fold_by_id) == len(ids) and set(fold_by_id) == set(ids)
    assert all(r["Strict_Group"] == outcomes[r["Record_ID"]]["strict_split_group"] for r in assignment)
    group_cv = list(GroupKFold(5).split(ids, groups=[outcomes[i]["strict_split_group"] for i in ids]))
    for k, (tr, va) in enumerate(group_cv, 1):
        assert set(va) == {j for j, i in enumerate(ids) if fold_by_id[i] == k}
        for key in isolation:
            assert not {outcomes[ids[j]][key] for j in tr} & {outcomes[ids[j]][key] for j in va}

    def text(i):
        return outcomes[i]["measure"] + "\n" + outcomes[i]["description"]

    def encode(tags):
        assert tags and set(tags) <= set(labels)
        return [int(label in tags) for label in labels]

    targets = np.array([encode(weak[i]["rule_labels"]) for i in ids])
    embedding_info = read(BASE / "MINILM_FEATURE_PROVENANCE.json")
    embeddings = np.load(BASE / "MINILM_EMBEDDINGS.npy", allow_pickle=False)
    eloc = {i: j for j, i in enumerate(embedding_info["ids"])}
    assert len(eloc) == embeddings.shape[0] == len(embedding_info["ids"])
    assert np.isfinite(embeddings).all()
    write_json("FROZEN_RUN_SPEC.json", dict(
        started=start, versions=versions, seeds={"estimator": SEED, "test_bootstrap": 20261124},
        model_settings=config, weights=GRID, untouched_weight=1.,
        deletion_conditions={DELETE: "frozen historical feature matrices, remove classifier rows",
                             REFIT: "TF-IDF diagnostic only; also remove 10 texts from vectorizer corpus"},
        fold_source=str(HIST / "GROUPED_CV_FOLD_ASSIGNMENTS.csv"), target="historical rule multi-hot",
        primary="five fixed training folds against weak targets; all conditions reported; no selection",
        secondary="post hoc reuse of 58 decidable Test_60 rows; 2000 paired NCT bootstrap draws",
        thresholds={"svm": "decision_function > 0", "lr": "predict_proba(class 1) > .5"},
        diagnostic_tolerance={"atol": 1e-9, "rtol": 1e-7},
        input_sha256=manifest, nct_per_partition={k: len({outcomes[i]["nct"] for i in v})
              for k, v in {"classifier": ids, "test": test_ids, "pool": pool_ids}.items()},
        pool_texts_in_feature_corpus=len(set(pool_ids) & set(eligible)),
        pool_rows_with_original_rule_targets=len(set(pool_ids) & set(ids)),
        support=dict(zip(labels, targets.sum(axis=0).tolist())),
        support_after_deletion=dict(zip(labels, targets[[i not in disputes for i in ids]].sum(axis=0).tolist()))))
    write_csv("DISAGREEMENT_RECORD_AUDIT.csv", disagreement_rows)
    log(f"Protocol frozen. 1934 targets, 10 disagreements, {len({r['NCT_ID'] for r in disagreement_rows})} NCTs.")

    cv_rows, class_weights, vector_audit, zero_audit, history_checks = [], [], [], [], []
    per_class, scores_rows, changed_test, summary = [], [], [], []
    cv_scores = {}
    test_scores = {}
    fits = 0

    for kind in KINDS:
        cv_scores[kind] = {c: np.zeros_like(targets, dtype=float)
                           for c in [*GRID, DELETE] + ([REFIT] if kind == KINDS[0] else [])}
        for fold, (tr, va) in enumerate([*group_cv, (np.arange(len(ids)), None)], 1):
            is_test = va is None
            fold = 0 if is_test else fold
            training = [ids[j] for j in tr]
            ev = test_ids if is_test else [ids[j] for j in va]
            conflict = np.array([i in disputes for i in training])
            retained = ~conflict
            if kind == KINDS[0]:
                corpus = eligible if is_test else training
                unchanged = vectorizer().fit([text(i) for i in corpus])
                rebuilt_corpus = [i for i in corpus if i not in disputes]
                rebuilt = vectorizer().fit([text(i) for i in rebuilt_corpus])
                common = sorted(unchanged.vocabulary_.keys() & rebuilt.vocabulary_.keys())
                old_idf = unchanged.idf_[[unchanged.vocabulary_[t] for t in common]]
                new_idf = rebuilt.idf_[[rebuilt.vocabulary_[t] for t in common]]
                vector_audit.append(dict(fold=fold, shared_corpus_n=len(corpus), refit_corpus_n=len(rebuilt_corpus),
                    shared_vocab=len(unchanged.vocabulary_), refit_vocab=len(rebuilt.vocabulary_),
                    identical_shared_vocab_idf_for_grid_and_classifier_deletion=True,
                    common_terms=len(common), changed_common_idf=int((old_idf != new_idf).sum()),
                    max_common_idf_delta=float(np.abs(old_idf-new_idf).max())))
                x = unchanged.transform([text(i) for i in training])
                evaluation = unchanged.transform([text(i) for i in ev])
                refit_x = rebuilt.transform([text(i) for i in training])
                refit_eval = rebuilt.transform([text(i) for i in ev])
            else:
                x = embeddings[[eloc[i] for i in training]]
                evaluation = embeddings[[eloc[i] for i in ev]]
            conditions = cv_scores[kind]
            results = {}
            for condition in conditions:
                keep = retained if condition in (DELETE, REFIT) else np.ones(len(tr), dtype=bool)
                wt = np.where(conflict, GRID.get(condition, 0.), 1.)[keep]
                y = targets[tr][keep]
                xx = (refit_x if condition == REFIT else x)[keep]
                ee = refit_eval if condition == REFIT else evaluation
                columns = []
                for col, label in enumerate(labels):
                    yc = y[:, col]
                    assert set(yc[wt > 0]) == {0, 1}
                    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=yc, sample_weight=wt)
                    estimator = (LinearSVC(C=1, class_weight="balanced", random_state=SEED)
                                 if kind == KINDS[0] else LogisticRegression(
                                     C=1, class_weight="balanced", max_iter=2000, random_state=SEED))
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        estimator.fit(xx, yc, sample_weight=wt)
                    for warning in caught:
                        log(f"{kind} fold {fold} {condition} {label}: {warning.category.__name__}: {warning.message}")
                    fits += 1
                    score = (estimator.decision_function(ee) if kind == KINDS[0]
                             else 2*estimator.predict_proba(ee)[:, 1]-1)
                    columns.append(score)
                    class_weights.append(dict(model=kind, fold=fold, condition=condition, label=label,
                        input_n=len(yc), unweighted_positive=int(yc.sum()), weight_total=float(wt.sum()),
                        weighted_positive=float(wt @ yc), class_weight_0=cw[0], class_weight_1=cw[1]))
                scores = np.stack(columns, axis=1)
                assert np.isfinite(scores).all()
                results[condition] = scores
                if not is_test:
                    conditions[condition][va] = scores
                else:
                    test_scores[kind, condition] = scores
            for condition, scores in results.items():
                p = scores > 0
                if not is_test:
                    cv_rows.append(dict(model=kind, condition=condition, fold=fold,
                        n_train=len(training)-int(conflict.sum()) if condition in (DELETE, REFIT) else len(training),
                        n_effective_train=len(training)-int(conflict.sum()) if GRID.get(condition, 0.) == 0. else len(training),
                        n_disagreement_in_training=int(conflict.sum()), n_validation=len(ev),
                        **metrics(targets[va], p), **prediction_delta(scores, results["W1.00"])))
                for i, sc in zip(ev, scores):
                    scores_rows.append(dict(model=kind, condition=condition, fold=fold,
                        Record_ID=i, NCT_ID=outcomes[i]["nct"],
                        predicted_labels=";".join(label for label, s in zip(labels, sc) if s > 0),
                        **dict(zip(labels, sc))))
            for condition in (DELETE, REFIT) if kind == KINDS[0] else (DELETE,):
                zero_audit.append(dict(model=kind, fold=fold, comparison=f"W0.00 vs {condition}",
                                       **prediction_delta(results["W0.00"], results[condition])))
            log(f"{kind} fold {fold} completed ({len(results)} conditions)")
    write_csv("PREDICTION_TRACE.csv", scores_rows)
    write_csv("GROUP_CV_RESULTS.csv", cv_rows)
    write_csv("CLASS_WEIGHT_TRACE.csv", class_weights)
    write_csv("TFIDF_FEATURE_AUDIT.csv", vector_audit)
    write_csv("ZERO_WEIGHT_COMPARISONS.csv", zero_audit)

    # Verify historical CV rows and all 60 historical hard predictions before opening human labels.
    historical_scores = read(HIST / "NOISE_PREDICTIONS_FROZEN.json")
    old_metrics = rows(HIST / "WEAK_LABEL_NOISE_SENSITIVITY.csv")
    cv_lookup = {(r["model"], r["condition"], r["fold"]): r for r in cv_rows}
    for row in old_metrics:
        fold = int(row["fold"])
        kind, condition = row["model"], OLD[row["strategy"]]
        if fold:
            current = cv_lookup[kind, condition, fold]
            delta = max(abs(current[m]-float(row[m])) for m in METRICS)
            assert delta <= 1e-12, (row, current)
            assert current["n_train"] == int(row["n_train"])
            assert current["n_validation"] == int(row["n_evaluated"])
            history_checks.append(dict(model=kind, condition=condition, fold=fold, metric_max_delta=delta))
        else:
            old = historical_scores[kind + "__" + row["strategy"]]
            current = test_scores[kind, condition]
            assert old["ids"] == test_ids and np.array_equal(current > 0, old["predictions"])
            history_checks.append(dict(model=kind, condition=condition, fold=0,
                **prediction_delta(current, np.asarray(old["scores"]))))
    log("Historical A/B/C reproduced. Opening frozen Test_60 strictly for evaluation.")
    gold_rows = [r for r in rows(BASE / "HUMAN_REFERENCE_FINAL_FROZEN.csv") if r["Partition"] == "Test_60"]
    gold_by_id = {r["Record_ID"]: r["Current_Author_Label"] for r in gold_rows}
    assert len(gold_rows) == len(gold_by_id) == 60 and set(gold_by_id) == set(test_ids)
    assert len({outcomes[i]["nct"] for i in test_ids}) == 60
    valid = np.array([gold_by_id[i] != "UNCLEAR" for i in test_ids])
    assert valid.sum() == 58
    gold = np.array([encode(gold_by_id[i].split(";")) for i, keep in zip(test_ids, valid) if keep])
    boot = np.random.default_rng(20261124).integers(0, len(gold), size=(2000, len(gold)))

    def boot_stats(pred):
        y, p = gold[boot], pred[boot]
        tp, fp, fn = (y*p).sum(1), ((1-y)*p).sum(1), (y*(1-p)).sum(1)
        den = 2*tp + fp + fn
        denom = den.sum(1)
        return [np.divide(2*tp.sum(1), denom, out=np.zeros(len(boot)), where=denom != 0),
                np.divide(2*tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0).mean(1),
                np.all(y == p, axis=2).mean(1)]

    for kind in KINDS:
        baseline_scores = test_scores[kind, "W1.00"][valid]
        baseline = baseline_scores > 0
        baseline_metrics = metrics(gold, baseline)
        baseline_boot = boot_stats(baseline)
        for condition, oof in cv_scores[kind].items():
            fold_metrics = [cv_lookup[kind, condition, fold] for fold in range(1, 6)]
            cv = dict(model=kind, condition=condition, scope="TRAINING_GROUP_CV_WEAK_REFERENCE", n_evaluated=len(ids),
                      **metrics(targets, oof > 0), **prediction_delta(oof, cv_scores[kind]["W1.00"]))
            for key in METRICS:
                values = [d[key] for d in fold_metrics]
                cv.update({key+"_fold_mean": float(np.mean(values)), key+"_fold_sd": float(np.std(values, ddof=1)),
                           key+"_fold_min": min(values), key+"_fold_max": max(values)})
            summary.append(cv)
            sc = test_scores[kind, condition][valid]
            pred = sc > 0
            test_metrics = metrics(gold, pred)
            test = dict(model=kind, condition=condition, scope="POST_HOC_REUSED_HUMAN_TEST_58",
                        n_evaluated=len(gold), **test_metrics, **prediction_delta(sc, baseline_scores))
            for key, stats, base_stats in zip(METRICS, boot_stats(pred), baseline_boot):
                low, high = np.quantile(stats-base_stats, [.025, .975])
                test.update({"delta_"+key: test_metrics[key]-baseline_metrics[key],
                             "delta_"+key+"_ci_low": low, "delta_"+key+"_ci_high": high})
            summary.append(test)
            tp, fp, fn = counts(gold, pred)
            for col, label in enumerate(labels):
                per_class.append(dict(model=kind, condition=condition, label=label,
                    support=int(gold[:, col].sum()), TP=int(tp[col]), FP=int(fp[col]), FN=int(fn[col])))
            for i, g, p, b in zip(np.array(test_ids)[valid], gold, pred, baseline):
                for label, truth, revised, prior in zip(labels, g, p, b):
                    if revised != prior:
                        changed_test.append(dict(model=kind, condition=condition, Record_ID=i,
                            label=label, human_reference=int(truth), baseline=int(prior), revised=int(revised)))
    for old in old_metrics:
        if int(old["fold"]) == 0:
            current = next(r for r in summary if r["model"] == old["model"] and
                           r["condition"] == OLD[old["strategy"]] and r["scope"].startswith("POST"))
            assert all(abs(float(old[key])-current[key]) < 1e-12 for key in METRICS)
            for key in ("micro_f1", "macro_f1"):
                for bound in ("low", "high"):
                    column = f"delta_{key.split('_')[0]}_ci_{bound}"
                    assert abs(float(old[column])-current[f"delta_{key}_ci_{bound}"]) < 1e-12
    write_csv("WEIGHT_SENSITIVITY_RESULTS.csv", summary)
    write_csv("TEST_CLASS_COUNTS.csv", per_class)
    write_csv("CHANGED_TEST_LABEL_DECISIONS.csv", changed_test)
    write_csv("HISTORICAL_REPRODUCTION.csv", history_checks)
    assert all(sha(ROOT / relative) == digest for relative, digest in manifest.items())
    write_json("RUN_VALIDATION.json", dict(started=start, completed=now(), estimator_fits=fits,
        historical_cv_rows_reproduced=30, historical_test_conditions_reproduced=6,
        cv_rows=len(cv_rows), test_summary_rows=len(summary)//2, input_hashes_unchanged=True,
        isolation_keys=list(isolation), cv_nct_disjoint=True, reused_test_not_used_in_fit=True,
        test_excluded_unclear=[i for i, included in zip(test_ids, valid) if not included],
        historical_composition_rows=len(ids)*len(OLD)*len(KINDS)))
    log(f"Completed {fits} binary estimator fits. Protected input hashes unchanged.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
