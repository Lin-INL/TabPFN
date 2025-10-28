from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import tabpfn.base as base_module
import tabpfn.classifier as classifier_module
from tabpfn import TabPFNClassifier
from tabpfn._telemetry import configure_telemetry
from tabpfn.preprocessing import ClassifierEnsembleConfig, PreprocessorConfig


class _DummyEngine:
    def __init__(
        self,
        y_encoded: np.ndarray,
        n_classes: int,
        ensemble_configs: list[ClassifierEnsembleConfig],
    ) -> None:
        self.y_encoded = y_encoded
        self.n_classes = n_classes
        self.ensemble_configs = ensemble_configs

    def use_torch_inference_mode(self, *, use_inference: bool) -> None:  # noqa: D401
        """Compatibility shim for the real inference engine."""

    def iter_outputs(self, X, *, devices, autocast):  # noqa: D401
        """Yield a deterministic prediction tensor for the provided inputs."""

        X = np.asarray(X)
        logits = torch.from_numpy(
            np.tile(
                np.linspace(0.1, 0.9, self.n_classes, dtype=np.float32),
                (len(X), 1),
            )
        )
        config = self.ensemble_configs[0]
        yield logits, config


@pytest.fixture(autouse=True)
def _disable_telemetry(monkeypatch):
    monkeypatch.setenv("TABPFN_OFFLINE_MODE", "1")
    configure_telemetry(None)


def test_tabpfn_classifier_offline_fit_predict(monkeypatch):
    # Any attempt to download telemetry config should fail the test immediately.
    import tabpfn_common_utils.telemetry.core.config as telemetry_config

    def _fail(*_args, **_kwargs):  # pragma: no cover - used to guard behaviour
        raise AssertionError("network access attempted")

    monkeypatch.setattr(telemetry_config.requests, "get", _fail)

    def fake_initialize_model_variables(self):
        self.model_ = object()
        self.config_ = SimpleNamespace()
        self.devices_ = (torch.device("cpu"),)
        self.use_autocast_ = False
        self.forced_inference_dtype_ = None
        self.memory_saving_mode = False
        from tabpfn.config import ModelInterfaceConfig

        self.interface_config_ = ModelInterfaceConfig()
        return 4, np.random.default_rng(0)

    def fake_initialize_dataset_preprocessing(self, X, y, rng):  # noqa: ARG001
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.feature_names_in_ = np.arange(X.shape[1])
        self.n_features_in_ = X.shape[1]

        self.inferred_categorical_indices_ = []
        self.preprocessor_ = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.preprocessor_.fit(X)
        X_transformed = self.preprocessor_.transform(X)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        self.label_encoder_ = label_encoder
        self.classes_ = label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        self.class_counts_ = np.bincount(y_encoded)

        ensemble_config = ClassifierEnsembleConfig(
            preprocess_config=PreprocessorConfig(name="none", categorical_name="none"),
            add_fingerprint_feature=False,
            polynomial_features="no",
            feature_shift_count=0,
            feature_shift_decoder="shuffle",
            subsample_ix=None,
            class_permutation=None,
        )
        return [ensemble_config], X_transformed, y_encoded

    def fake_create_inference_engine(**kwargs):
        y_train = np.asarray(kwargs["y_train"])
        n_classes = len(np.unique(y_train))
        ensemble_configs = kwargs.get("ensemble_configs", [])
        return _DummyEngine(y_train, n_classes, ensemble_configs)

    monkeypatch.setattr(
        TabPFNClassifier,
        "_initialize_model_variables",
        fake_initialize_model_variables,
    )
    monkeypatch.setattr(
        TabPFNClassifier,
        "_initialize_dataset_preprocessing",
        fake_initialize_dataset_preprocessing,
    )
    monkeypatch.setattr(base_module, "create_inference_engine", fake_create_inference_engine)
    monkeypatch.setattr(
        classifier_module,
        "create_inference_engine",
        fake_create_inference_engine,
    )

    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4))
    y = rng.integers(0, 2, size=30)

    clf = TabPFNClassifier(device="cpu", telemetry_enabled=None)
    clf.fit(X, y)

    preds = clf.predict(X)
    probas = clf.predict_proba(X)

    assert preds.shape == (30,)
    assert probas.shape == (30, len(np.unique(y)))
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
