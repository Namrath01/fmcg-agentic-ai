"""
Agent Orchestrator
-------------------
Runs all 5 agents in sequence, passes outputs between them,
resolves financial conflicts by looping back to PackSizeOptimizationAgent,
and returns a unified results dictionary.

Pipeline:
  DemandIntelligenceAgent
       v
  PackSizeOptimizationAgent  <--- revision loop from FinancialImpactAgent
       v
  FinancialImpactAgent
       v
  ProductionPlanningAgent
       v
  DispatchOptimizationAgent
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from agents import (
    DemandIntelligenceAgent,
    DispatchOptimizationAgent,
    FinancialImpactAgent,
    PackSizeOptimizationAgent,
    ProductionPlanningAgent,
)
from utils.data_loader import UnifiedDataLoader


class AgentOrchestrator:
    """
    Central orchestration layer for the Enterprise Agentic Supply Chain Optimizer.

    Attributes:
        max_revision_loops : maximum times PackSize -> Financial loop can repeat
        progress_callback  : optional callable(step, total, message) for UI updates
    """

    MAX_REVISION_LOOPS = 2

    def __init__(
        self,
        max_revision_loops: int = MAX_REVISION_LOOPS,
        progress_callback: Optional[Callable] = None,
    ):
        self.max_revision_loops = max_revision_loops
        self.progress_callback = progress_callback
        self.orchestration_log: List[str] = []
        self.results: Dict = {}

    # -- internal logging --------------------------------------------------

    def _log(self, msg: str, step: int = 0, total: int = 5) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] [Orchestrator] {msg}"
        self.orchestration_log.append(entry)
        try:
            print(entry)
        except UnicodeEncodeError:
            print(entry.encode("ascii", errors="replace").decode("ascii"))
        if self.progress_callback:
            try:
                self.progress_callback(step, total, msg)
            except Exception:
                pass

    def _section(self, title: str) -> None:
        line = "-" * 60
        self._log(line)
        self._log(f"  {title}")
        self._log(line)

    # -- data loading ------------------------------------------------------

    def _load_data(self) -> tuple:
        self._log("Loading datasets (SupplyGraph + M5) ...", 0, 5)
        loader = UnifiedDataLoader()
        master_df, metadata = loader.load()
        self._log(
            f"Data loaded - {len(master_df):,} rows, "
            f"{master_df['SkuId'].nunique()} SKUs, "
            f"{master_df['PlantId'].nunique()} plants"
        )
        return master_df, metadata

    # -- agent runners -----------------------------------------------------

    def _run_demand(self, master_df: pd.DataFrame, filters: Dict) -> Dict:
        self._section("STEP 1/5 - Demand Intelligence Agent")
        agent = DemandIntelligenceAgent()

        sku_filter = filters.get("sku_filter", None)
        horizon = filters.get("horizon_days", 30)

        result = agent.run(master_df, sku_filter=sku_filter, horizon_days=horizon)
        self._log(
            f"Demand complete - MAPE={result['metrics']['MAPE']}%, "
            f"RMSE={result['metrics']['RMSE']:.1f}, "
            f"SKUs forecasted={result['metrics']['skus_forecasted']}",
            step=1,
        )
        return result

    def _run_pack_size(
        self,
        demand_result: Dict,
        master_df: pd.DataFrame,
        unprofitable_skus: Optional[List[str]] = None,
    ) -> Dict:
        self._section("STEP 2/5 - Pack Size Optimization Agent")
        agent = PackSizeOptimizationAgent()
        result = agent.run(
            demand_result, master_df, unprofitable_skus=unprofitable_skus
        )
        self._log(
            f"Pack size complete - "
            f"fast={result['summary']['fast_movers']}, "
            f"medium={result['summary']['medium_movers']}, "
            f"slow={result['summary']['slow_movers']}, "
            f"avg_score={result['summary']['avg_justification_score']}",
            step=2,
        )
        return result

    def _run_financial(
        self,
        pack_result: Dict,
        master_df: pd.DataFrame,
        metadata: Dict,
    ) -> Dict:
        self._section("STEP 3/5 - Financial Impact Agent")
        agent = FinancialImpactAgent()
        result = agent.run(pack_result, master_df, metadata)
        s = result["summary"]
        self._log(
            f"Financial complete - "
            f"PBT uplift={s['PBT_Uplift_Pct']}%, "
            f"Revenue uplift={s['Revenue_Uplift_Pct']}%, "
            f"Debtor improvement={s['Debtor_Cycle_Improvement_Pct']}%, "
            f"Unprofitable SKUs={s['Unprofitable_SKU_Count']}",
            step=3,
        )
        return result

    def _run_production(
        self,
        master_df: pd.DataFrame,
        pack_result: Dict,
        metadata: Dict,
    ) -> Dict:
        self._section("STEP 4/5 - Production Planning Agent")
        agent = ProductionPlanningAgent()
        result = agent.run(master_df, pack_result=pack_result, metadata=metadata)
        s = result["summary"]
        self._log(
            f"Production complete - "
            f"plants={s['plants_analysed']}, "
            f"SKUs={s['skus_scheduled']}, "
            f"avg_utilisation={s['avg_target_utilisation']}%, "
            f"avg_lead_time={s['avg_lead_time_days']}d",
            step=4,
        )
        return result

    def _run_dispatch(
        self,
        master_df: pd.DataFrame,
        metadata: Dict,
        production_result: Dict,
    ) -> Dict:
        self._section("STEP 5/5 - Dispatch Optimization Agent")
        agent = DispatchOptimizationAgent()
        result = agent.run(
            master_df, metadata=metadata, production_result=production_result
        )
        s = result["summary"]
        self._log(
            f"Dispatch complete - "
            f"storage_locations={s['storage_locations']}, "
            f"lead_time_reduction={s['avg_lead_time_reduction_pct']}%",
            step=5,
        )
        return result

    # -- conflict resolution -----------------------------------------------

    def _resolve_financial_conflict(
        self,
        financial_result: Dict,
        demand_result: Dict,
        master_df: pd.DataFrame,
        metadata: Dict,
        loop_count: int,
    ) -> tuple:
        """
        If FinancialAgent flags unprofitable SKUs, send them back to
        PackSizeAgent for a conservative pack revision, then re-run Financial.
        """
        unprofitable = financial_result.get("unprofitable_skus", [])
        if not unprofitable or loop_count >= self.max_revision_loops:
            if loop_count >= self.max_revision_loops and unprofitable:
                self._log(
                    f"[CONFLICT RESOLVED] Max revision loops ({self.max_revision_loops}) "
                    f"reached. Accepting {len(unprofitable)} residual unprofitable SKUs."
                )
            return None, None

        self._log(
            f"[CONFLICT DETECTED] {len(unprofitable)} unprofitable SKUs - "
            f"sending back to PackSizeAgent for revision (loop {loop_count + 1}) ..."
        )
        revised_pack = self._run_pack_size(demand_result, master_df, unprofitable_skus=unprofitable)
        revised_financial = self._run_financial(revised_pack, master_df, metadata)
        self._log("[CONFLICT RESOLVED] Pack revision accepted by FinancialAgent.")
        return revised_pack, revised_financial

    # -- main orchestration ------------------------------------------------

    def run(
        self,
        filters: Optional[Dict] = None,
        master_df: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Execute the full 5-agent pipeline with conflict resolution.

        Args:
            filters   : dict with optional sku_filter, horizon_days
            master_df : pre-loaded dataframe (if None, loads from disk)
            metadata  : pre-loaded metadata dict (if None, loads from disk)

        Returns:
            Unified results dict with all agent outputs + orchestration log
        """
        start_time = datetime.now()
        if filters is None:
            filters = {}

        self._section("ENTERPRISE AGENTIC SUPPLY CHAIN OPTIMIZER")
        self._log("Orchestration pipeline starting ...")

        try:
            # -- Data loading ----------------------------------------------
            if master_df is None or metadata is None:
                master_df, metadata = self._load_data()

            # -- Agent 1: Demand Intelligence ------------------------------
            demand_result = self._run_demand(master_df, filters)

            # -- Agent 2: Pack Size Optimization ---------------------------
            pack_result = self._run_pack_size(demand_result, master_df)

            # -- Agent 3: Financial Impact + conflict loop -----------------
            financial_result = self._run_financial(pack_result, master_df, metadata)

            for loop in range(self.max_revision_loops):
                revised_pack, revised_financial = self._resolve_financial_conflict(
                    financial_result, demand_result, master_df, metadata, loop
                )
                if revised_pack is None:
                    break
                pack_result = revised_pack
                financial_result = revised_financial

            # -- Agent 4: Production Planning ------------------------------
            production_result = self._run_production(master_df, pack_result, metadata)

            # -- Agent 5: Dispatch Optimization ----------------------------
            dispatch_result = self._run_dispatch(master_df, metadata, production_result)

            elapsed = (datetime.now() - start_time).total_seconds()
            self._section("PIPELINE COMPLETE")
            self._log(f"All 5 agents completed in {elapsed:.1f}s")

            self.results = {
                "status": "success",
                "elapsed_seconds": round(elapsed, 1),
                "master_df": master_df,
                "metadata": metadata,
                "demand": demand_result,
                "pack_size": pack_result,
                "financial": financial_result,
                "production": production_result,
                "dispatch": dispatch_result,
                "orchestration_log": self.orchestration_log.copy(),
                "pipeline_summary": self._build_pipeline_summary(
                    demand_result, pack_result, financial_result,
                    production_result, dispatch_result
                ),
            }

            self._export_results(self.results)

        except Exception as exc:
            self._log(f"[ERROR] Pipeline failed: {exc}")
            traceback.print_exc()
            self.results = {
                "status": "error",
                "error": str(exc),
                "orchestration_log": self.orchestration_log.copy(),
            }

        return self.results

    def _export_results(self, results: Dict) -> None:
        """Save all agent result dataframes to outputs/ in the project root."""
        root = Path(__file__).resolve().parent.parent
        out_dir = root / "outputs"
        out_dir.mkdir(exist_ok=True)

        exports = {
            "01_demand_forecast":       results["demand"].get("forecast_df"),
            "02_pack_recommendations":  results["pack_size"].get("recommendations_df"),
            "03_sku_velocity_profiles": results["pack_size"].get("velocity_profile_df"),
            "04_financial_impact":      results["financial"].get("financial_table_df"),
            "05_production_schedule":   results["production"].get("production_schedule_df"),
            "06_plant_statistics":      results["production"].get("plant_stats_df"),
            "07_plant_utilisation":     results["production"].get("utilisation_summary_df"),
            "08_dispatch_plan":         results["dispatch"].get("dispatch_plan_df"),
            "09_routing_paths":         results["dispatch"].get("path_df"),
        }

        # Pipeline summary KPIs
        ps = results.get("pipeline_summary", {})
        exports["10_pipeline_summary_kpis"] = pd.DataFrame(
            [{"Metric": k, "Value": v} for k, v in ps.items()]
        )

        # Benchmark comparison
        fin_s = results["financial"].get("summary", {})
        disp_s = results["dispatch"].get("summary", {})
        benchmarks = [
            ("PBT Uplift %",           fin_s.get("PBT_Uplift_Pct", 0),              10, 18),
            ("Revenue Uplift %",       fin_s.get("Revenue_Uplift_Pct", 0),           8, 15),
            ("Inventory Reduction %",  fin_s.get("Inventory_Reduction_Pct", 0),      20, 30),
            ("Lead Time Reduction %",  disp_s.get("avg_lead_time_reduction_pct", 0), 20, 30),
            ("Debtor Improvement %",   fin_s.get("Debtor_Cycle_Improvement_Pct", 0), 25, 35),
        ]
        exports["11_benchmark_comparison"] = pd.DataFrame([
            {
                "Metric": name, "Achieved": val,
                "Target_Min": lo, "Target_Max": hi,
                "Status": "On Target" if lo <= val <= hi else (
                    "Above Target" if val > hi else "Below Target"
                ),
            }
            for name, val, lo, hi in benchmarks
        ])

        # Orchestration log
        exports["12_orchestration_log"] = pd.DataFrame(
            {"Log": results.get("orchestration_log", [])}
        )

        saved, failed = [], []
        for name, df in exports.items():
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            try:
                path = out_dir / f"{name}.csv"
                df.to_csv(path, index=False, encoding="utf-8-sig")
                saved.append(f"{name}.csv ({len(df):,} rows)")
            except Exception as exc:
                failed.append(f"{name}: {exc}")

        self._log(f"Exported {len(saved)} CSVs to {out_dir}")
        for s in saved:
            self._log(f"  -> {s}")
        for f in failed:
            self._log(f"  [WARN] Export failed: {f}")

    def _build_pipeline_summary(
        self, demand, pack, financial, production, dispatch
    ) -> Dict:
        """Consolidate key metrics from all agents into one summary dict."""
        fin_s = financial.get("summary", {})
        prod_s = production.get("summary", {})
        disp_s = dispatch.get("summary", {})
        pack_s = pack.get("summary", {})
        demand_m = demand.get("metrics", {})

        return {
            # Demand
            "demand_mape_pct": demand_m.get("MAPE", 0),
            "demand_rmse": demand_m.get("RMSE", 0),
            "skus_forecasted": demand_m.get("skus_forecasted", 0),
            # Pack
            "fast_movers": pack_s.get("fast_movers", 0),
            "medium_movers": pack_s.get("medium_movers", 0),
            "slow_movers": pack_s.get("slow_movers", 0),
            "avg_inventory_reduction_pct": pack_s.get("avg_inventory_reduction_pct", 0),
            # Financial
            "pbt_uplift_pct": fin_s.get("PBT_Uplift_Pct", 0),
            "revenue_uplift_pct": fin_s.get("Revenue_Uplift_Pct", 0),
            "inventory_reduction_pct": fin_s.get("Inventory_Reduction_Pct", 0),
            "debtor_cycle_improvement_pct": fin_s.get("Debtor_Cycle_Improvement_Pct", 0),
            # Production
            "plants_analysed": prod_s.get("plants_analysed", 0),
            "avg_plant_utilisation_pct": prod_s.get("avg_target_utilisation", 0),
            "avg_lead_time_days": prod_s.get("avg_lead_time_days", 0),
            # Dispatch
            "avg_lead_time_reduction_pct": disp_s.get("avg_lead_time_reduction_pct", 0),
            "storage_locations": disp_s.get("storage_locations", 0),
        }


if __name__ == "__main__":
    orch = AgentOrchestrator()
    results = orch.run()
    if results["status"] == "success":
        print("\nPipeline Summary:")
        for k, v in results["pipeline_summary"].items():
            print(f"  {k}: {v}")
    else:
        print(f"Pipeline failed: {results['error']}")
        sys.exit(1)
