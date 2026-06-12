"""
Production Planning Agent
--------------------------
Uses SupplyGraph Production.csv and Factory Issue.csv to map 40 SKUs across
9 plants, calculate optimal batch run sizes minimising factory issue rates,
and output a production schedule with plant utilisation percentages.
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class ProductionPlanningAgent:
    """
    Agent 4 - Production Planning

    Input : master_df (unified), pack recommendations, plant-node metadata
    Output: production schedule dataframe, plant utilisation summary
    """

    AGENT_NAME = "ProductionPlanningAgent"
    TARGET_UTILISATION = 0.82      # target 82% utilisation
    MAX_UTILISATION = 0.95         # cap to avoid overload
    ISSUE_RATE_THRESHOLD = 0.08    # >8% factory issue rate = high-risk

    def __init__(self):
        self.log: list = []

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] [{self.AGENT_NAME}] {msg}"
        self.log.append(entry)
        try:
            print(entry)
        except UnicodeEncodeError:
            print(entry.encode("ascii", errors="replace").decode("ascii"))

    # -- plant capacity model ----------------------------------------------

    def _compute_plant_stats(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate production and factory issue metrics per plant."""
        self._log("Computing plant-level production statistics ...")

        plant_stats = master_df.groupby("PlantId").agg(
            TotalProductionQty=("ProductionQty", "sum"),
            AvgDailyProduction=("ProductionQty", "mean"),
            MaxDailyProduction=("ProductionQty", "max"),
            TotalFactoryIssueQty=("FactoryIssueQty", "sum"),
            AvgFactoryIssueQty=("FactoryIssueQty", "mean"),
            TotalSalesOrderQty=("SalesOrderQty", "sum"),
            AvgDailySales=("SalesOrderQty", "mean"),
            SKU_Count=("SkuId", "nunique"),
            DataDays=("Date", "nunique"),
        ).reset_index()

        # Factory issue rate: SupplyGraph "Factory Issue" qty represents
        # goods dispatched FROM the factory (can exceed same-day production
        # due to buffer drawdown). Cap ratio at 1.0 and convert to a defect
        # proxy as |production - factory_issue| / production.
        raw_ratio = np.where(
            plant_stats["TotalProductionQty"] > 0,
            plant_stats["TotalFactoryIssueQty"] / (plant_stats["TotalProductionQty"] + 1e-9),
            0,
        )
        # Treat over-issue as supply pressure, not quality defect; cap at 1.
        # Issue rate for planning = deviation from 1 (ideal 1:1 issue:produce).
        plant_stats["FactoryIssueRate"] = np.clip(np.abs(raw_ratio - 1.0), 0, 1)

        # Proxy for plant capacity (max daily x 1.1 headroom)
        plant_stats["EstCapacity"] = plant_stats["MaxDailyProduction"] * 1.10

        # Current utilisation
        plant_stats["CurrentUtilisation"] = np.clip(
            plant_stats["AvgDailyProduction"] / (plant_stats["EstCapacity"] + 1e-9),
            0, 1
        )

        plant_stats["RiskFlag"] = (
            plant_stats["FactoryIssueRate"] > self.ISSUE_RATE_THRESHOLD
        )

        return plant_stats

    # -- batch size optimisation -------------------------------------------

    def _optimal_batch_size(
        self,
        avg_demand: float,
        holding_cost_per_unit: float = 0.5,
        setup_cost: float = 500.0,
    ) -> int:
        """Economic Order Quantity (EOQ) as batch size proxy."""
        if avg_demand <= 0 or holding_cost_per_unit <= 0:
            return 100
        eoq = np.sqrt(2 * avg_demand * 365 * setup_cost / holding_cost_per_unit)
        return max(50, int(round(eoq)))

    # -- schedule generation -----------------------------------------------

    def _build_production_schedule(
        self,
        master_df: pd.DataFrame,
        plant_stats: pd.DataFrame,
        reco_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        self._log("Building optimal production schedule ...")

        sku_agg = master_df.groupby(["SkuId", "PlantId"]).agg(
            AvgDailyDemand=("SalesOrderQty", "mean"),
            AvgDailyProduction=("ProductionQty", "mean"),
            AvgFactoryIssue=("FactoryIssueQty", "mean"),
            AvgUnitCost=("UnitCost", "mean"),
            ProductGroup=("ProductGroup", "first"),
            SubGroup=("SubGroup", "first"),
        ).reset_index()

        # Merge with pack recommendations if available
        if reco_df is not None and not reco_df.empty:
            sku_agg = sku_agg.merge(
                reco_df[["SkuId", "VelocityClass", "PackMultiplier", "RecommendedPackQty"]],
                on="SkuId", how="left"
            )
        else:
            sku_agg["VelocityClass"] = "medium_mover"
            sku_agg["PackMultiplier"] = 12
            sku_agg["RecommendedPackQty"] = sku_agg["AvgDailyDemand"] * 12

        sku_agg = sku_agg.merge(
            plant_stats[["PlantId", "EstCapacity", "CurrentUtilisation",
                          "FactoryIssueRate", "RiskFlag"]],
            on="PlantId", how="left"
        )
        sku_agg = sku_agg.fillna(0)

        # Per-SKU optimal batch
        sku_agg["OptimalBatchSize"] = sku_agg.apply(
            lambda r: self._optimal_batch_size(r["AvgDailyDemand"], r["AvgUnitCost"] * 0.22),
            axis=1
        )

        # Target daily production (cap at capacity share)
        plant_sku_count = sku_agg.groupby("PlantId")["SkuId"].count().reset_index()
        plant_sku_count.columns = ["PlantId", "PlantSKUCount"]
        sku_agg = sku_agg.merge(plant_sku_count, on="PlantId", how="left")

        sku_agg["CapacitySharePerSKU"] = sku_agg["EstCapacity"] / sku_agg["PlantSKUCount"].clip(1)
        sku_agg["TargetDailyProduction"] = np.minimum(
            sku_agg["AvgDailyDemand"] * 1.05,     # 5% safety buffer
            sku_agg["CapacitySharePerSKU"] * self.TARGET_UTILISATION
        )

        # Production efficiency score (inverse of factory issue rate)
        sku_agg["ProductionEfficiencyScore"] = np.clip(
            1 - sku_agg["FactoryIssueRate"] * 10, 0, 1
        ) * 100

        # Lead time = days to produce one optimal batch, capped at 30 days
        raw_lt = np.where(
            sku_agg["TargetDailyProduction"] > 0,
            np.ceil(sku_agg["OptimalBatchSize"] / sku_agg["TargetDailyProduction"]),
            14,
        )
        sku_agg["EstLeadTimeDays"] = np.clip(raw_lt, 1, 30).astype(int)

        sku_agg["SchedulePriority"] = pd.cut(
            sku_agg["AvgDailyDemand"],
            bins=[0, 100, 500, np.inf],
            labels=["Low", "Medium", "High"]
        ).astype(str)

        return sku_agg

    # -- plant utilisation summary -----------------------------------------

    def _plant_utilisation_summary(
        self, plant_stats: pd.DataFrame, schedule_df: pd.DataFrame
    ) -> pd.DataFrame:
        self._log("Computing plant utilisation summary ...")

        util = schedule_df.groupby("PlantId").agg(
            TargetTotalDailyProd=("TargetDailyProduction", "sum"),
            SKU_Count=("SkuId", "nunique"),
            AvgBatchSize=("OptimalBatchSize", "mean"),
            AvgEfficiencyScore=("ProductionEfficiencyScore", "mean"),
        ).reset_index()

        util = util.merge(
            plant_stats[["PlantId", "EstCapacity", "CurrentUtilisation",
                          "FactoryIssueRate", "RiskFlag"]],
            on="PlantId", how="left"
        )
        util["TargetUtilisation"] = np.clip(
            util["TargetTotalDailyProd"] / (util["EstCapacity"] + 1e-9),
            0, self.MAX_UTILISATION
        )
        util["UtilisationImprovement_Pct"] = (
            (util["TargetUtilisation"] - util["CurrentUtilisation"]) * 100
        )

        return util

    # -- main entry --------------------------------------------------------

    def plan(
        self,
        master_df: pd.DataFrame,
        pack_result: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        self._log("Agent activated.")

        reco_df = None
        if pack_result:
            reco_df = pack_result.get("recommendations_df", None)

        plant_stats = self._compute_plant_stats(master_df)
        schedule_df = self._build_production_schedule(master_df, plant_stats, reco_df)
        util_summary = self._plant_utilisation_summary(plant_stats, schedule_df)

        high_risk = plant_stats[plant_stats["RiskFlag"]]["PlantId"].tolist()
        if high_risk:
            self._log(f"[WARN] High factory issue rate plants: {high_risk}")

        summary = {
            "plants_analysed": len(plant_stats),
            "skus_scheduled": len(schedule_df),
            "high_risk_plants": high_risk,
            "avg_target_utilisation": round(util_summary["TargetUtilisation"].mean() * 100, 1),
            "avg_factory_issue_rate": round(plant_stats["FactoryIssueRate"].mean() * 100, 2),
            "avg_lead_time_days": round(schedule_df["EstLeadTimeDays"].mean(), 1),
        }

        self._log(f"Planning complete - {summary}")
        return {
            "agent": self.AGENT_NAME,
            "production_schedule_df": schedule_df,
            "plant_stats_df": plant_stats,
            "utilisation_summary_df": util_summary,
            "summary": summary,
            "log": self.log.copy(),
        }

    def run(
        self,
        master_df: pd.DataFrame,
        pack_result: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        return self.plan(master_df, pack_result=pack_result, metadata=metadata)
