"""
Demand Intelligence Agent
-------------------------
Trains a RandomForestRegressor on SupplyGraph Sales Order history augmented
with M5 calendar / SNAP signals to produce SKU-level demand forecasts with
confidence intervals, MAPE, and RMSE.
"""

import joblib
import os
import warnings
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class DemandIntelligenceAgent:
    """
    Agent 1 - Demand Intelligence

    Input : unified master dataframe (from UnifiedDataLoader)
    Output: forecast dataframe, evaluation metrics dict, confidence intervals
    """

    AGENT_NAME = "DemandIntelligenceAgent"

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_cols: list = []
        self.is_trained = False
        self.log: list = []

    # -- logging ----------------------------------------------------------

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] [{self.AGENT_NAME}] {msg}"
        self.log.append(entry)
        try:
            print(entry)
        except UnicodeEncodeError:
            print(entry.encode("ascii", errors="replace").decode("ascii"))

    # -- feature engineering ----------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag, rolling, and calendar features for the RF model."""
        self._log("Building temporal features ...")
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values(["SkuId", "Date"]).reset_index(drop=True)

        # Calendar features
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["month"] = df["Date"].dt.month
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)

        # Per-SKU lag / rolling features
        df["lag_7"] = df.groupby("SkuId")["SalesOrderQty"].shift(7).fillna(0)
        df["lag_14"] = df.groupby("SkuId")["SalesOrderQty"].shift(14).fillna(0)
        df["rolling_mean_7"] = (
            df.groupby("SkuId")["SalesOrderQty"]
            .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
            .fillna(0)
        )
        df["rolling_std_7"] = (
            df.groupby("SkuId")["SalesOrderQty"]
            .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
            .fillna(0)
        )

        # Promotional flag (from M5 SNAP / calendar)
        if "Promotional_Flag" not in df.columns:
            df["Promotional_Flag"] = 0

        # Encode categorical columns
        for col in ["SkuId", "ProductGroup", "SubGroup"]:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        return df

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        candidates = [
            "day_of_week", "month", "day_of_year", "week_of_year",
            "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7",
            "Promotional_Flag",
            "SkuId_enc", "ProductGroup_enc", "SubGroup_enc",
        ]
        return [c for c in candidates if c in df.columns]

    # -- training ---------------------------------------------------------

    def train(self, master_df: pd.DataFrame) -> "DemandIntelligenceAgent":
        model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'demand_model.pkl')
        if os.path.exists(model_path):
            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.feature_cols = bundle["feature_cols"]
            self.label_encoders = bundle["label_encoders"]
            self.rmse = bundle.get("rmse", 0.0)
            self.mape = bundle.get("mape", 0.0)
            self.is_trained = True
            self._log("Loaded saved model from disk -- skipping retraining.")
            return self

        self._log("Starting training pipeline ...")
        df = self._build_features(master_df)
        self.feature_cols = self._get_feature_cols(df)

        # Drop rows with zero or NaN target (start-of-series lags)
        df = df[df["SalesOrderQty"] > 0].dropna(subset=self.feature_cols + ["SalesOrderQty"])

        X = df[self.feature_cols].values
        y = df["SalesOrderQty"].values

        # Time-based split: never shuffle time series data as it leaks future information into training
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        self._log(f"Time-based split: {len(X_train):,} train rows, {len(X_test):,} test rows")

        self._log(f"Training RandomForest on {len(X_train):,} rows ({len(self.feature_cols)} features) ...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        self.mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100)
        self.is_trained = True

        self._log(f"Training complete - RMSE={self.rmse:.2f}, MAPE={self.mape:.2f}%")

        joblib.dump({
            "model": self.model,
            "feature_cols": self.feature_cols,
            "label_encoders": self.label_encoders,
            "rmse": self.rmse,
            "mape": self.mape,
        }, model_path)
        self._log(f"Model saved to {model_path}")
        return self

    # -- inference --------------------------------------------------------

    def _encode_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stored label encoders to inference dataframe."""
        for col, le in self.label_encoders.items():
            if col in df.columns:
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[f"{col}_enc"] = le.transform(df[col])
        return df

    def forecast(
        self,
        master_df: pd.DataFrame,
        horizon_days: int = 30,
        sku_filter: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate demand forecasts for the next `horizon_days` days.

        Returns:
            forecast_df : SKU x date forecast with confidence intervals
            metrics     : dict with MAPE, RMSE, feature importances
        """
        if not self.is_trained:
            self._log("Model not trained - training now ...")
            self.train(master_df)

        self._log(f"Generating {horizon_days}-day horizon forecasts ...")

        df = self._build_features(master_df)
        df = self._encode_inference(df)

        skus = sku_filter if sku_filter else df["SkuId"].unique().tolist()
        last_date = df["Date"].max()

        all_records = []
        for sku in skus:
            sku_df = df[df["SkuId"] == sku].sort_values("Date")
            if sku_df.empty:
                continue

            for d in range(1, horizon_days + 1):
                forecast_date = last_date + pd.Timedelta(days=d)
                latest = sku_df.iloc[-1]

                row = {
                    "day_of_week": forecast_date.dayofweek,
                    "month": forecast_date.month,
                    "day_of_year": forecast_date.dayofyear,
                    "week_of_year": forecast_date.isocalendar()[1],
                    "lag_7": float(latest["SalesOrderQty"]),
                    "lag_14": float(sku_df.iloc[-min(14, len(sku_df))]["SalesOrderQty"]),
                    "rolling_mean_7": float(sku_df["SalesOrderQty"].tail(7).mean()),
                    "rolling_std_7": float(sku_df["SalesOrderQty"].tail(7).std() or 0),
                    "Promotional_Flag": 0,
                    "SkuId_enc": latest.get("SkuId_enc", 0),
                    "ProductGroup_enc": latest.get("ProductGroup_enc", 0),
                    "SubGroup_enc": latest.get("SubGroup_enc", 0),
                }

                feat_vec = np.array([[row.get(c, 0) for c in self.feature_cols]])

                # Predict using all trees for CI
                tree_preds = np.array([t.predict(feat_vec)[0] for t in self.model.estimators_])
                point_forecast = float(tree_preds.mean())
                ci_lower = float(np.percentile(tree_preds, 10))
                ci_upper = float(np.percentile(tree_preds, 90))

                all_records.append({
                    "SkuId": sku,
                    "Date": forecast_date,
                    "ForecastQty": max(0, point_forecast),
                    "CI_Lower": max(0, ci_lower),
                    "CI_Upper": max(0, ci_upper),
                    "ProductGroup": latest.get("ProductGroup", "UNK"),
                    "SubGroup": latest.get("SubGroup", "UNK"),
                    "PlantId": latest.get("PlantId", "UNKNOWN"),
                })

        forecast_df = pd.DataFrame(all_records)
        self._log(f"Forecast complete - {len(forecast_df):,} SKUxday records generated")

        # Also attach actuals for the historical window
        actuals = master_df[master_df["SkuId"].isin(skus)][
            ["Date", "SkuId", "SalesOrderQty", "ProductGroup"]
        ].copy()
        actuals.rename(columns={"SalesOrderQty": "ActualQty"}, inplace=True)

        metrics = {
            "MAPE": round(self.mape, 2),
            "RMSE": round(self.rmse, 2),
            "feature_importance": dict(zip(
                self.feature_cols,
                self.model.feature_importances_.tolist()
            )),
            "skus_forecasted": len(skus),
            "horizon_days": horizon_days,
        }

        return forecast_df, actuals, metrics

    def run(self, master_df: pd.DataFrame, **kwargs) -> Dict:
        """Unified entry point called by the orchestrator."""
        self._log("Agent activated by orchestrator.")
        forecast_df, actuals_df, metrics = self.forecast(master_df, **kwargs)
        return {
            "agent": self.AGENT_NAME,
            "forecast_df": forecast_df,
            "actuals_df": actuals_df,
            "metrics": metrics,
            "log": self.log.copy(),
        }
