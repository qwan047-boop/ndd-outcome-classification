# Reconstructed baseline snapshots

The two joblib bundles contain the fitted five binary classifiers, ordered labels, and fixed decision threshold. The LinearSVC bundle also includes the fitted TF-IDF vectorizer. The logistic regression bundle expects the pinned encoder's 384-dimensional features. Use the packaged archived feature matrix for the fixed-record audit, or the optional encoder entry for raw outcome text processing. The original settings are in `history/baseline/EXPERIMENT_CONFIG.yaml`.

Both bundles carry the status `RECONSTRUCTED_MODEL`. Use `reproduce.py load-models` to verify the package hashes before loading. Joblib is pickle-based and can execute code during deserialization. Load these artifacts only from a trusted, independently verified copy under the recorded scikit-learn version. The `train` command provides a way to refit the models locally without loading a distributed pickle.
