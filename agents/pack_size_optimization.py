"""
Pack Size Optimization Agent
------------------------------
Clusters SKUs by demand velocity (KMeans, 3 clusters) and maps clusters
to plant capacity data from SupplyGraph to recommend pack configurations.
Accepts demand forecasts from DemandIntelligenceAgent as input.
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


PACK_CONFIGS = {
    "fast_mover": {
        "label": "Bulk Logistics Case",
        "description": "High-volume bulk case pack for continuous replenishment",
        "multiplier": 24,
        "unit": "case",
        "cost_factor": 0.85,
    },
    "medium_mover": {
        "label": "Standard Consumer Pack",
        "description": "Standard shelf-ready pack for regular distribution",
        "multiplier": 12,
        "unit": "pack",
        "cost_factor": 1.00,
    },
    "slow_mover": {
        "label": "Promotional Multipack",
        "description": "Promotional bundle to accelerate slow-moving inventory",
        "multiplier": 6,
        "unit": "multipack",
        "cost_factor": 1.10,
    },
}


class PackSizeOptimizationAgent:
    """
    Agent 2 - Pack Size Optimization

    Input : forecast_df from DemandIntelligenceAgent, master_df metadata
    Output: pack recommendation dataframe with justification scores
    """

    AGENT_NAME = "PackSizeOptimizationAgent"
    N_CLUSTERS = 3

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.N_CLUSTERS, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.cluster_labels: Dict[int, str] = {}
        self.log: list = []

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] [{self.AGENT_NAME}] {msg}"
        self.log.append(entry)
        try:
            print(entry)
        except UnicodeEncodeError:
            print(entry.encode("ascii", errors="replace").decode("ascii"))

    # -- velocity profiling ------------------------------------------------

    def _build_velocity_profile(
        self, forecast_df: pd.DataFrame, master_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate per-SKU demand velocity features."""
        self._log("Building SKU velocity profiles ...")

        # Historical statistics from master
        hist = master_df.groupby("SkuId").agg(
            hist_mean=("SalesOrderQty", "mean"),
            hist_std=("SalesOrderQty", "std"),
            hist_cv=("SalesOrderQty", lambda x: x.std() / (x.mean() + 1e-9)),
            hist_total=("SalesOrderQty", "sum"),
            n_days=("Date", "nunique"),
        ).reset_index()

        # Forecast statistics
        fcast = forecast_df.groupby("SkuId").agg(
            fcast_mean=("ForecastQty", "mean"),
            fcast_total=("ForecastQty", "sum"),
        ).reset_index()

        # Plant capacity from master
        plant_info = master_df.groupby("SkuId").agg(
            PlantId=("PlantId", "first"),
            StorageLocationId=("StorageLocationId", "first"),
            ProductGroup=("ProductGroup", "first"),
            SubGroup=("SubGroup", "first"),
        ).reset_index()

        profile = hist.merge(fcast, on="SkuId", how="outer").merge(
            plant_info, on="SkuId", how="left"
        )
        profile = profile.fillna(0)
        return profile

    # -- clustering --------------------------------------------------------

    def _cluster_skus(self, profile: pd.DataFrame) -> pd.DataFrame:
        self._log("Running KMeans clustering (3 velocity classes) ...")

        feature_cols = ["hist_mean", "hist_std", "hist_cv", "fcast_mean"]
        X = profile[feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        profile["cluster"] = self.kmeans.fit_predict(X_scaled)

        # Map clusters to fast / medium / slow by centroid hist_mean rank
        centroids = self.kmeans.cluster_centers_
        # Use first feature (hist_mean) dimension ranking
        hist_mean_col_idx = 0
        sorted_clusters = np.argsort(
            [centroids[k][hist_mean_col_idx] for k in range(self.N_CLUSTERS)]
        )[::-1]  # descending = fast first
        velocity_names = ["fast_mover", "medium_mover", "slow_mover"]
        self.cluster_labels = {
            int(sorted_clusters[i]): velocity_names[i]
            for i in range(self.N_CLUSTERS)
        }

        profile["velocity_class"] = profile["cluster"].map(self.cluster_labels)
        self._log(
            f"Cluster map: {self.cluster_labels} | "
            f"counts: {profile['velocity_class'].value_counts().to_dict()}"
        )
        return profile

    # -- recommendation engine ---------------------------------------------

    def _generate_recommendations(self, profile: pd.DataFrame) -> pd.DataFrame:
        self._log("Generating pack recommendations ...")
        rows = []
        for _, row in profile.iterrows():
            vc = row.get("velocity_class", "medium_mover")
            config = PACK_CONFIGS.get(vc, PACK_CONFIGS["medium_mover"])

            # Justification score (0-100): higher velocity + lower CV = more confident
            velocity_score = min(100, row["hist_mean"] / (profile["hist_mean"].max() + 1e-9) * 60)
            stability_score = max(0, 40 * (1 - min(1, row["hist_cv"])))
            justification_score = round(velocity_score + stability_score, 1)

            # Optimal pack quantity = forecast mean * multiplier / days
            recommended_pack_qty = max(
                1,
                int(round(row["fcast_mean"] * config["multiplier"] / 30))
            )

            # Estimated inventory reduction
            inv_reduction_pct = {
                "fast_mover": 30.0,
                "medium_mover": 23.0,
                "slow_mover": 15.0,
            }.get(vc, 20.0)

            rows.append({
                "SkuId": row["SkuId"],
                "ProductGroup": row["ProductGroup"],
                "SubGroup": row["SubGroup"],
                "PlantId": row["PlantId"],
                "StorageLocationId": row["StorageLocationId"],
                "VelocityClass": vc,
                "PackConfig": config["label"],
                "PackDescription": config["description"],
                "PackMultiplier": config["multiplier"],
                "RecommendedPackQty": recommended_pack_qty,
                "JustificationScore": justification_score,
                "HistMeanDailyDemand": round(row["hist_mean"], 1),
                "ForecastMeanDailyDemand": round(row["fcast_mean"], 1),
                "DemandCV": round(row["hist_cv"], 3),
                "EstInventoryReductionPct": round(inv_reduction_pct, 1),
                "CostFactor": config["cost_factor"],
            })

        return pd.DataFrame(rows)

    # -- capacity validation -----------------------------------------------

    def _validate_against_plant_capacity(
        self, reco_df: pd.DataFrame, master_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Flag SKUs where recommended qty exceeds plant average production capacity."""
        self._log("Validating recommendations against plant capacity boundaries ...")

        plant_capacity = master_df.groupby("PlantId")["ProductionQty"].mean().reset_index()
        plant_capacity.columns = ["PlantId", "AvgPlantCapacity"]

        reco_df = reco_df.merge(plant_capacity, on="PlantId", how="left")
        reco_df["AvgPlantCapacity"] = reco_df["AvgPlantCapacity"].fillna(
            master_df["ProductionQty"].mean()
        )
        reco_df["CapacityFeasible"] = (
            reco_df["RecommendedPackQty"] <= reco_df["AvgPlantCapacity"]
        )

        infeasible = (~reco_df["CapacityFeasible"]).sum()
        if infeasible > 0:
            self._log(
                f"[WARN] {infeasible} SKUs exceed plant capacity - "
                "flagging for financial impact review."
            )
        return reco_df

    # -- main entry --------------------------------------------------------

    def optimize(
        self,
        forecast_df: pd.DataFrame,
        master_df: pd.DataFrame,
        unprofitable_skus: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run pack size optimization.

        Args:
            forecast_df      : output from DemandIntelligenceAgent
            master_df        : unified master dataframe
            unprofitable_skus: SKUs flagged by FinancialImpactAgent for revision
        """
        self._log("Agent activated.")
        if unprofitable_skus:
            self._log(
                f"Re-optimizing {len(unprofitable_skus)} unprofitable SKUs "
                "with conservative pack strategy ..."
            )

        profile = self._build_velocity_profile(forecast_df, master_df)
        profile = self._cluster_skus(profile)
        reco_df = self._generate_recommendations(profile)

        # Force unprofitable SKUs to slower velocity class
        if unprofitable_skus:
            mask = reco_df["SkuId"].isin(unprofitable_skus)
            reco_df.loc[mask, "VelocityClass"] = "slow_mover"
            reco_df.loc[mask, "PackConfig"] = PACK_CONFIGS["slow_mover"]["label"]
            reco_df.loc[mask, "PackMultiplier"] = PACK_CONFIGS["slow_mover"]["multiplier"]
            reco_df.loc[mask, "JustificationScore"] *= 0.7

        reco_df = self._validate_against_plant_capacity(reco_df, master_df)

        summary = {
            "fast_movers": int((reco_df["VelocityClass"] == "fast_mover").sum()),
            "medium_movers": int((reco_df["VelocityClass"] == "medium_mover").sum()),
            "slow_movers": int((reco_df["VelocityClass"] == "slow_mover").sum()),
            "avg_justification_score": round(reco_df["JustificationScore"].mean(), 1),
            "avg_inventory_reduction_pct": round(
                reco_df["EstInventoryReductionPct"].mean(), 1
            ),
            "capacity_infeasible_count": int((~reco_df["CapacityFeasible"]).sum()),
        }

        self._log(f"Optimization complete - {len(reco_df)} SKU recommendations generated.")
        self._log(f"Summary: {summary}")

        return {
            "agent": self.AGENT_NAME,
            "recommendations_df": reco_df,
            "velocity_profile_df": profile,
            "summary": summary,
            "cluster_labels": self.cluster_labels,
            "log": self.log.copy(),
        }

    def run(self, forecast_result: Dict, master_df: pd.DataFrame, **kwargs) -> Dict:
        """Orchestrator entry point."""
        forecast_df = forecast_result.get("forecast_df", pd.DataFrame())
        unprofitable = kwargs.get("unprofitable_skus", None)
        return self.optimize(forecast_df, master_df, unprofitable_skus=unprofitable)
