"""
Demand Forecasting Module

Builds a demand model that predicts quantity_sold as a function of price,
competitor price, and seasonality. This is the "predict-then-optimize" bridge
between ML and OR — the most critical technical decision in the project.

The model outputs a demand function d_i(p_i) consumed by the optimization layer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import pickle
import warnings

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODELS_DIR = Path(__file__).parent.parent / "data" / "processed" / "models"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


FEATURE_COLS = ["price", "competitor_price", "week", "category_encoded", "location_encoded"]
TARGET_COL = "quantity_sold"


class DemandModel:
    """Demand forecasting model that predicts quantity sold given features."""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.label_encoders = {}
        self.feature_cols = FEATURE_COLS
        self.metrics = {}
        self.is_fitted = False

    def _get_model(self):
        """Initialize the appropriate model."""
        if self.model_type == "xgboost" and HAS_XGB:
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
        elif self.model_type == "lightgbm" and HAS_LGBM:
            return lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        else:
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()

        for col in ["category", "location"]:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le is not None:
                        df[f"{col}_encoded"] = le.transform(df[col].astype(str))
                    else:
                        df[f"{col}_encoded"] = 0

        if "category_encoded" not in df.columns:
            df["category_encoded"] = 0
        if "location_encoded" not in df.columns:
            df["location_encoded"] = 0

        return df

    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the demand model with time-based validation."""
        df = self._prepare_features(df, fit=True)

        available_features = [c for c in self.feature_cols if c in df.columns]

        df = df.sort_values("week").reset_index(drop=True)

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df[available_features].values
        y_train = train_df[TARGET_COL].values
        X_test = test_df[available_features].values
        y_test = test_df[TARGET_COL].values

        self.model = self._get_model()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        self.metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100,
            "train_size": len(train_df),
            "test_size": len(test_df),
        }

        if hasattr(self.model, "feature_importances_"):
            self.metrics["feature_importance"] = dict(
                zip(available_features, self.model.feature_importances_)
            )

        self.is_fitted = True
        print(f"Demand model ({self.model_type}) trained:")
        print(f"  MAE: {self.metrics['mae']:.2f}")
        print(f"  RMSE: {self.metrics['rmse']:.2f}")
        print(f"  R²: {self.metrics['r2']:.4f}")
        print(f"  MAPE: {self.metrics['mape']:.1f}%")

        return self.metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict demand for given features."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        df = self._prepare_features(df, fit=False)
        available_features = [c for c in self.feature_cols if c in df.columns]
        X = df[available_features].values
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)

    def predict_single(self, price: float, competitor_price: float, week: int,
                       category: str = "default", location: str = "default") -> float:
        """Predict demand for a single SKU/location/price combination."""
        df = pd.DataFrame([{
            "price": price,
            "competitor_price": competitor_price,
            "week": week,
            "category": category,
            "location": location,
        }])
        return float(self.predict(df)[0])

    def get_demand_curve(self, competitor_price: float, week: int,
                         category: str = "default", location: str = "default",
                         price_range: Tuple[float, float] = (5.0, 200.0),
                         n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a demand curve for a given SKU/location over a price range."""
        prices = np.linspace(price_range[0], price_range[1], n_points)
        demands = [
            self.predict_single(p, competitor_price, week, category, location)
            for p in prices
        ]
        return prices, np.array(demands)

    def create_demand_function(self, competitor_price: float, week: int,
                               category: str = "default",
                               location: str = "default") -> Callable[[float], float]:
        """Create a callable demand function for the optimizer.

        This is the key interface between ML and OR:
        d_i(p_i) -> expected_quantity
        """
        def demand_fn(price: float) -> float:
            return self.predict_single(price, competitor_price, week, category, location)
        return demand_fn

    def create_discretized_demand_function(self, competitor_price: float, week: int,
                                           category: str = "default",
                                           location: str = "default",
                                           price_grid: np.ndarray = None) -> Dict[float, float]:
        """Create a discretized demand function for MILP (maps price grid to demand)."""
        if price_grid is None:
            price_grid = np.linspace(5.0, 200.0, 20)

        demands = {}
        for p in price_grid:
            demands[p] = self.predict_single(p, competitor_price, week, category, location)

        return demands

    def save(self, path: Optional[Path] = None):
        """Save the model to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / f"demand_model_{self.model_type}.pkl"

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "label_encoders": self.label_encoders,
                "model_type": self.model_type,
                "metrics": self.metrics,
                "feature_cols": self.feature_cols,
            }, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None, model_type: str = "xgboost") -> "DemandModel":
        """Load a saved model."""
        if path is None:
            path = MODELS_DIR / f"demand_model_{model_type}.pkl"

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(model_type=data["model_type"])
        instance.model = data["model"]
        instance.label_encoders = data["label_encoders"]
        instance.metrics = data["metrics"]
        instance.feature_cols = data["feature_cols"]
        instance.is_fitted = True
        return instance


def build_demand_models(df: pd.DataFrame, model_type: str = "xgboost") -> DemandModel:
    """Build and evaluate a demand model for the entire dataset.

    This function builds a single global model that predicts demand across
    all SKUs and locations, using category/location as features.

    For per-SKU models, train separate models in the notebook.
    """
    print("=" * 60)
    print("DEMAND MODEL TRAINING")
    print("=" * 60)

    model = DemandModel(model_type=model_type)
    metrics = model.fit(df)
    model.save()

    print("\n" + "=" * 60)
    print("DEMAND MODEL COMPLETE")
    print("=" * 60)

    return model


def build_sku_demand_models(df: pd.DataFrame,
                            model_type: str = "gradient_boosting") -> Dict[str, DemandModel]:
    """Build separate demand models per SKU for more accurate per-product forecasts."""
    models = {}
    for sku in df["product_id"].unique():
        sku_data = df[df["product_id"] == sku].copy()
        if len(sku_data) < 30:
            print(f"Skipping {sku}: insufficient data ({len(sku_data)} rows)")
            continue

        print(f"\nTraining model for {sku}...")
        model = DemandModel(model_type=model_type)
        model.fit(sku_data)
        models[sku] = model

    return models


if __name__ == "__main__":
    from data_prep import load_processed

    weekly_df, elasticity_df = load_processed()
    global_model = build_demand_models(weekly_df)

    sample = weekly_df.head(5)
    print("\nSample predictions:")
    print(global_model.predict(sample))
