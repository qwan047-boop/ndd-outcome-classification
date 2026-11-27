# Pediatric neurodevelopmental trial outcome classification

This research compares rule-based classifiers, TF-IDF with LinearSVC, and MiniLM with logistic regression for assigning five domains to pediatric trial outcome descriptions. It also audits ChatGPT web weak labels and examines offline active-learning replay.

The author confirmed on 27 November 2026 that the authorized project rights holder approved publication of the original project code, human reference, machine weak labels, derived research data and reconstructed classifiers. Target visibility is Public. No remote repository, DOI or project open-source license has been assigned. Third-party material retains its source terms. See PUBLICATION_AUTHORIZATION.md and DATA_SOURCE_AND_TERMS.md.

## Third-party text hold

Publication of the complete package is currently BLOCKED. Two required frozen-input files are
held outside this directory: data/pediatric_outcomes.json and data/registry_source_extract_670.json.
The source outcome NCT05413187:secondaryOutcomes:3 contains a sponsor confidentiality and prior-written-
permission statement. Project-owned asset publication is approved, but this explicit source notice
has not been resolved. No historical text, label, prediction or model was changed to bypass it.
The checksums cover the files actually included here. Dataset-dependent commands need the two
original files restored with their verified hashes after the third-party issue is resolved.
Full-input scoring and model loading were verified before the hold was applied.

## Files

|Path|Contents|
|---|---|
|data/ and 03_annotation/|ClinicalTrials.gov source fields from 670 frozen NCTs, 3,412 eligible outcomes from 463 trials, 1,934 rule targets, splits, duplicate links, and annotation rules.|
|data/llm/|Codebook, versioned Prompt, 200 original and merged weak predictions, quality states, and sanitized provenance.|
|history/baseline/|100 single-reviewer reference labels, archived predictions, cached MiniLM features, model configurations, and baseline and replay results.|
|history/weight_grid/ and history/sensitivity/|Five-weight evaluation and earlier exploratory diagnostic outputs.|
|history/chronology/|[Experiment timeline](history/chronology/EXPERIMENT_TIMELINE_PUBLIC.md) and stage classifications.|
|code/|Rule definitions and code for evaluation, active-learning replay, the weight grid and optional embedding regeneration.|
|models/|[Two reconstructed classifiers and loading instructions](models/README.md).|

Test_60 contains 60 single-reviewer records, including 58 decidable outcomes used for scoring. Pool_40 contains 40 records for offline replay. The implicit weak-negative encoding marks unassigned domains as training negatives under the historical convention. Those targets lack clinical negative confirmation. Machine weak labels and semantic QC states do not serve as a human reference.

## Setup and checks from saved artifacts

The tested environment is 64-bit Windows with Python 3.12.14 and the versions pinned in requirements.txt. Use a Python environment outside the repository and fresh result directories outside it. Commands run offline once the dependencies are available.

~~~
python -m pip install -r requirements.txt
python -B reproduce.py verify
python -B verify_llm_provenance.py
python -B verify_cdp_freeze.py
python -B reproduce.py metrics --out ../run_metrics
python -B reproduce.py load-models --out ../run_load
~~~

verify checks both SHA-256 lists, dataset counts, protected partitions, recognized leakage boundaries, and original outcome fields. The provenance scripts verify archived batch IDs, ordered record mappings and source fingerprints offline. metrics recomputes the baseline and replay scores and Bootstrap intervals from frozen predictions. load-models scores the fixed test records using the packaged reconstructed estimators, saved vectorizer and cached embeddings. Joblib is pickle-based. Load only snapshots from a trusted, independently verified package under the pinned software versions. A checksum verifies bytes and does not establish that an artifact is trustworthy.

SHA256_MANIFEST.json covers the analysis files. SHA256SUMS.txt also includes that JSON manifest. The SHA256SUMS.txt file's own hash is held in the external upload manifest. The Git attributes file preserves file bytes across platforms.

## Optional workflows involving refitting

The Stage 3 standalone evaluation exercised these workflows. Use the recorded conditions and fresh output directories. The release-preparation validation checks saved artifacts without refitting.

~~~
python -B reproduce.py train --out ../run_train
python -B reproduce.py weight-grid --out ../run_weights
python -B reproduce.py replay --out ../run_replay
~~~

train refits the original TF-IDF and the two baseline heads on frozen data and writes explicitly reconstructed snapshots. weight-grid refits the five prespecified weights and deletion conditions with the same NCT/duplicate-component folds. Fold metrics measure agreement with weak rule targets. replay reruns fixed selections, label releases and refits and compares them with the archived traces. The initial baseline, fixed model evaluation and initial replay preceded the first complete evaluation report. The targeted second LLM round, detailed error analysis and both sensitivity analyses are post hoc. The weights are heuristic and uncalibrated. The reused test set provides exploratory comparisons with residual risk of post hoc bias.

## Encoder and ChatGPT source

The [official MiniLM model card at the pinned revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) states Apache-2.0. Upstream weights are obtained separately. The expected model.safetensors SHA-256 is 53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db. See history/baseline/MINILM_FEATURE_PROVENANCE.json for the required file hashes and requirements-encoder.txt for optional packages.

~~~
hf download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --include "*.json" "1_Pooling/config.json" "vocab.txt" "README.md" "model.safetensors" --local-dir ../encoder
python -B reproduce.py encoder --encoder-dir ../encoder --out ../run_encoder
~~~

The optional encoder command checks the files, regenerates features with remote code disabled, and refits the LR head. The local archive uses the author-confirmed display name GPT-6.1 Sol. All ten successful messages recorded the historical slug gpt-6.1-sol-wm and one empty attempt is separately documented. The exact backend snapshot and weights are unavailable. The release includes neutral evidence keys, original output JSON and the Prompt. Private source files and original message identifiers are held separately. See [LLM provenance supplement](data/llm/PROVENANCE_SUPPLEMENT.md).

## Limits

The original fitted models, original TF-IDF vectorizer, per-budget trained estimators, and complete unseen-pool score arrays were not retained. Available snapshots are labeled RECONSTRUCTED_MODEL. Archived prediction scoring can be verified independently of training, and results from the local environment do not establish cross-platform reproducibility. The package starts from historically frozen registry extracts and saved ChatGPT outputs. A live registry query or current web model would not reproduce those inputs. See [Distribution and provenance notice](DISTRIBUTION_NOTICE.md) for source attribution, transformations, and access conditions.


## Authorization and citation

Original project assets are APPROVED for the planned publication on the basis of the recorded rights-holder confirmation. Mixed registry text remains under ClinicalTrials.gov and applicable third-party terms, as recorded in ASSET_APPROVAL_STATUS.csv. No MIT, CC BY or other project reuse license is selected or granted. Pretrained encoder weights are obtained separately.

CITATION.cff contains the confirmed authors and manuscript title, without a DOI, publication year or acceptance assertion. The package starts from historical frozen source fields and preserved annotations. It preserves the distinction between original fixed analyses and later exploratory expansion and sensitivity analysis. The heuristic weights are uncalibrated. Model training and replay commands refit classifiers and were not rerun in this final packaging check; the archive-scoring and saved-model loading commands were run.
