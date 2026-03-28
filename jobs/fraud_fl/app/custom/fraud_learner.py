"""NVFlare ModelLearner wrapping XGBoost for federated fraud detection."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.app_common.app_constant import AppConstants
from config import FEATURE_COLS, TARGET_COL, XGBOOST_PARAMS, PARTITIONED_DIR, FL_ESTIMATORS_PER_ROUND
from utils.metrics import compute_metrics


SITE_TO_BANK = {
    "site-1": "a",
    "site-2": "b",
    "site-3": "c",
}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


class FraudLearner(ModelLearner):
    def __init__(
        self,
        train_task_name: str = AppConstants.TASK_TRAIN,
        validate_task_name: str = AppConstants.TASK_VALIDATION,
    ) -> None:
        super().__init__()
        self.train_task_name = train_task_name
        self.validate_task_name = validate_task_name
        self._model: XGBClassifier | None = None
        self._X_train: np.ndarray | None = None
        self._X_test: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_test: np.ndarray | None = None

    def initialize(self) -> None:
        site_name = self.site_name
        bank = SITE_TO_BANK.get(site_name, "a")
        src = PARTITIONED_DIR / f"bank_{bank}_engineered.csv"

        if not src.exists():
            raise FileNotFoundError(
                f"Engineered data not found at {src} — run feature_engineering.py first"
            )

        df = pd.read_csv(src)
        feature_cols = _get_feature_cols(df)
        X = df[feature_cols].fillna(0.0).values
        y = df[TARGET_COL].astype(int).values

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self._scale_pos_weight = int(
            (self._y_train == 0).sum() / max((self._y_train == 1).sum(), 1)
        )
        self.info(f"Loaded bank_{bank}: {len(self._X_train)} train, {len(self._X_test)} test")

    def _serialize_model(self) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self._model.get_booster().save_model(tmp_path)
            return Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _deserialize_model(self, model_bytes: bytes) -> None:
        booster = xgb.Booster()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_path = tmp.name
        try:
            booster.load_model(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Wrap the loaded booster in the sklearn API so predict/predict_proba work.
        self._model = XGBClassifier(**XGBOOST_PARAMS)
        n_features = self._X_train.shape[1]
        dummy_X = np.zeros((2, n_features), dtype=np.float32)
        dummy_y = np.array([0, 1], dtype=np.int32)
        self._model.fit(dummy_X, dummy_y)
        self._model._Booster = booster

    def train(self, model: FLModel) -> FLModel:
        fl_params = {
            **XGBOOST_PARAMS,
            "n_estimators": FL_ESTIMATORS_PER_ROUND,
            "scale_pos_weight": self._scale_pos_weight,
        }
        xgb_warm_start = None

        # If global model weights provided (round > 0), warm-start from them
        if model is not None and model.params:
            model_bytes = model.params.get("model_bytes")
            if model_bytes:
                self._deserialize_model(model_bytes)
                xgb_warm_start = self._model.get_booster()

        # Always create a fresh classifier with the correct FL tree budget
        self._model = XGBClassifier(**fl_params)
        self._model.fit(
            self._X_train,
            self._y_train,
            eval_set=[(self._X_test, self._y_test)],
            verbose=False,
            xgb_model=xgb_warm_start,
        )

        model_bytes = self._serialize_model()
        return FLModel(
            params={"model_bytes": model_bytes},
            params_type="FULL",
            meta={"num_samples": len(self._X_train)},
        )

    def validate(self, model: FLModel) -> FLModel:
        if model is not None and model.params:
            model_bytes = model.params.get("model_bytes")
            if model_bytes:
                self._deserialize_model(model_bytes)

        if self._model is None:
            self.warning("No model to validate")
            return FLModel(params={}, params_type="FULL")

        y_pred = self._model.predict(self._X_test)
        y_prob = self._model.predict_proba(self._X_test)[:, 1]
        metrics = compute_metrics(self._y_test, y_pred, y_prob)

        self.info(f"Validation — F1: {metrics['f1']:.4f}  AUC-PR: {metrics['auc_pr']:.4f}")
        return FLModel(
            params={"metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"}},
            params_type="FULL",
            meta={"num_samples": len(self._X_test)},
        )
