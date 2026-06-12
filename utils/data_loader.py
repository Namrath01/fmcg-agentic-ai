"""
Data loader for the Enterprise Agentic Supply Chain Optimizer.

Handles SupplyGraph (Wasi et al., AAAI 2024) and M5 (Makridakis et al.) datasets,
melts wide-format temporal CSVs into long format, and maps columns to a unified
SAP OData-aligned schema. Falls back to synthetic data if files are missing.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
SUPPLYGRAPH_DIR = BASE_DIR / "data" / "supplygraph"
M5_DIR = BASE_DIR / "data" / "m5"

# -- Fallback paths (original locations before user moves data) --------------
_ALT_SG = BASE_DIR / "SupplyGraph" / "Raw Dataset" / "Homogenoeus"
_ALT_M5 = BASE_DIR / "M5"


def _sg_root() -> Path:
    """Return whichever SupplyGraph root exists."""
    canonical = SUPPLYGRAPH_DIR / "Temporal Data"
    if canonical.exists():
        return SUPPLYGRAPH_DIR
    if (_ALT_SG / "Temporal Data").exists():
        return _ALT_SG
    return SUPPLYGRAPH_DIR  # will trigger synthetic fallback


def _m5_root() -> Path:
    if (M5_DIR / "calendar.csv").exists():
        return M5_DIR
    if (_ALT_M5 / "calendar.csv").exists():
        return _ALT_M5
    return M5_DIR


# -- Unified SAP OData column schema -----------------------------------------
UNIFIED_SCHEMA = [
    "Date", "SkuId", "ProductGroup", "SubGroup",
    "PlantId", "StorageLocationId",
    "OnHandQty", "UnitCost", "LeadTimeDays",
    "SalesOrderQty", "ProductionQty", "DeliveryQty", "FactoryIssueQty",
    "Promotional_Flag", "Price",
]


# ===========================================================================
# SupplyGraph Loader
# ===========================================================================

class SupplyGraphLoader:
    """Load and merge all SupplyGraph homogeneous datasets."""

    def __init__(self):
        self.root = _sg_root()
        self.temporal_unit = self.root / "Temporal Data" / "Unit"
        self.temporal_weight = self.root / "Temporal Data" / "Weight"
        self.nodes_dir = self.root / "Nodes"
        self.edges_dir = self.root / "Edges"

    # -- helpers ----------------------------------------------------------

    def _melt_temporal(self, filepath: Path, value_name: str) -> Optional[pd.DataFrame]:
        """Read a wide-format CSV (rows=dates, cols=SKUs) and melt to long."""
        if not filepath.exists():
            return None
        try:
            df = pd.read_csv(filepath)
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.rename(columns={date_col: "Date"})
            melted = df.melt(id_vars="Date", var_name="SkuId", value_name=value_name)
            melted[value_name] = pd.to_numeric(melted[value_name], errors="coerce").fillna(0)
            return melted
        except Exception as exc:
            print(f"[WARN] Could not load {filepath.name}: {exc}")
            return None

    def _find_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """Case-insensitive file search within a directory."""
        if not directory.exists():
            return None
        for p in directory.iterdir():
            if pattern.lower() in p.name.lower():
                return p
        return None

    # -- temporal loaders -------------------------------------------------

    def load_sales_order(self) -> Optional[pd.DataFrame]:
        f = self._find_file(self.temporal_unit, "sales order")
        return self._melt_temporal(f, "SalesOrderQty") if f else None

    def load_production(self) -> Optional[pd.DataFrame]:
        f = self._find_file(self.temporal_unit, "production")
        return self._melt_temporal(f, "ProductionQty") if f else None

    def load_delivery(self) -> Optional[pd.DataFrame]:
        f = self._find_file(self.temporal_unit, "delivery")
        return self._melt_temporal(f, "DeliveryQty") if f else None

    def load_factory_issue(self) -> Optional[pd.DataFrame]:
        f = self._find_file(self.temporal_unit, "factory issue")
        return self._melt_temporal(f, "FactoryIssueQty") if f else None

    # -- node / edge loaders ----------------------------------------------

    def load_nodes(self) -> pd.DataFrame:
        f = self._find_file(self.nodes_dir, "nodes.csv")
        if f and f.exists():
            return pd.read_csv(f).rename(columns={"Node": "SkuId"})
        return pd.DataFrame(columns=["SkuId"])

    def load_node_plant_storage(self) -> pd.DataFrame:
        f = self._find_file(self.nodes_dir, "plant")
        if f and f.exists():
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            return df.rename(columns={
                "Node": "SkuId",
                "Plant": "PlantId",
                "Storage Location": "StorageLocationId",
            })
        return pd.DataFrame(columns=["SkuId", "PlantId", "StorageLocationId"])

    def load_node_product_group(self) -> pd.DataFrame:
        f = self._find_file(self.nodes_dir, "product group")
        if f and f.exists():
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            return df.rename(columns={
                "Node": "SkuId",
                "Group": "ProductGroup",
                "Sub-Group": "SubGroup",
            })
        return pd.DataFrame(columns=["SkuId", "ProductGroup", "SubGroup"])

    def load_edges_storage(self) -> pd.DataFrame:
        f = self._find_file(self.edges_dir, "storage location")
        if f and f.exists():
            return pd.read_csv(f)
        return pd.DataFrame(columns=["Storage Location", "node1", "node2"])

    def load_edges_plant(self) -> pd.DataFrame:
        f = self._find_file(self.edges_dir, "plant")
        if f and f.exists():
            return pd.read_csv(f)
        return pd.DataFrame(columns=["Plant", "node1", "node2"])

    def load_edges_product_group(self) -> pd.DataFrame:
        f = self._find_file(self.edges_dir, "product group")
        if f and f.exists():
            return pd.read_csv(f)
        return pd.DataFrame()

    def load_edges_product_subgroup(self) -> pd.DataFrame:
        f = self._find_file(self.edges_dir, "sub-group")
        if f and f.exists():
            return pd.read_csv(f)
        return pd.DataFrame()

    # -- master merge -----------------------------------------------------

    def build_master(self) -> pd.DataFrame:
        """Merge all temporal and structural tables into one unified dataframe."""
        so = self.load_sales_order()
        pr = self.load_production()
        dl = self.load_delivery()
        fi = self.load_factory_issue()

        if so is None:
            print("[INFO] Sales Order CSV not found - using synthetic fallback.")
            return _generate_synthetic()

        base = so.copy()
        for df, col in [(pr, "ProductionQty"), (dl, "DeliveryQty"), (fi, "FactoryIssueQty")]:
            if df is not None:
                base = base.merge(df, on=["Date", "SkuId"], how="left")
            else:
                base[col] = 0

        # Structural metadata
        plant_storage = self.load_node_plant_storage()
        product_group = self.load_node_product_group()

        base = base.merge(plant_storage, on="SkuId", how="left")
        base = base.merge(product_group, on="SkuId", how="left")

        # Fill remaining unified schema columns with defaults
        # Seeded RNG ensures UnitCost and Price are identical across every run
        rng = np.random.default_rng(42)
        base["OnHandQty"] = base.get("OnHandQty", 0)
        base["UnitCost"] = rng.uniform(5, 50, len(base))
        base["LeadTimeDays"] = rng.integers(1, 14, len(base))
        base["Promotional_Flag"] = 0
        base["Price"] = base["UnitCost"] * rng.uniform(1.2, 2.5, len(base))

        # Fill NaN structural columns
        if "PlantId" not in base.columns:
            base["PlantId"] = "P_UNKNOWN"
        if "StorageLocationId" not in base.columns:
            base["StorageLocationId"] = "SL_UNKNOWN"
        if "ProductGroup" not in base.columns:
            base["ProductGroup"] = "UNK"
        if "SubGroup" not in base.columns:
            base["SubGroup"] = "UNK"

        for col in ["PlantId", "StorageLocationId", "ProductGroup", "SubGroup"]:
            base[col] = base[col].fillna("UNKNOWN").astype(str)

        base = base.fillna(0)
        return base[UNIFIED_SCHEMA]


# ===========================================================================
# M5 Loader
# ===========================================================================

class M5Loader:
    """Load Walmart M5 dataset components."""

    def __init__(self):
        self.root = _m5_root()

    def load_calendar(self) -> pd.DataFrame:
        p = self.root / "calendar.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p, parse_dates=["date"])
        return df

    def load_sell_prices(self) -> pd.DataFrame:
        p = self.root / "sell_prices.csv"
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p)

    def load_sales_train(self, evaluation: bool = False) -> pd.DataFrame:
        fname = "sales_train_evaluation.csv" if evaluation else "sales_train_validation.csv"
        p = self.root / fname
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p)

    def get_snap_flags(self) -> pd.DataFrame:
        """Return per-date SNAP flag (any state active = 1)."""
        cal = self.load_calendar()
        if cal.empty:
            return pd.DataFrame()
        snap_cols = [c for c in cal.columns if c.startswith("snap_")]
        if snap_cols:
            cal["snap_flag"] = cal[snap_cols].max(axis=1)
        else:
            cal["snap_flag"] = 0
        return cal[["date", "snap_flag", "event_name_1", "event_type_1",
                     "weekday", "wday", "month", "year"]].rename(columns={"date": "Date"})

    def get_price_lookup(self) -> pd.DataFrame:
        """Return averaged sell price per item_id across all stores and weeks."""
        prices = self.load_sell_prices()
        if prices.empty:
            return pd.DataFrame()
        return prices.groupby("item_id")["sell_price"].mean().reset_index().rename(
            columns={"item_id": "SkuId", "sell_price": "M5_AvgPrice"}
        )


# ===========================================================================
# Unified Loader
# ===========================================================================

class UnifiedDataLoader:
    """Combine SupplyGraph master table with M5 calendar signals."""

    def __init__(self):
        self.sg = SupplyGraphLoader()
        self.m5 = M5Loader()

    def load(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Returns:
            master_df: unified long-format dataframe aligned to UNIFIED_SCHEMA
            metadata:  dict with raw edge/node tables and M5 components
        """
        master = self.sg.build_master()

        # Attach M5 SNAP / event calendar
        snap = self.m5.get_snap_flags()
        if not snap.empty:
            snap["Date"] = pd.to_datetime(snap["Date"], errors="coerce")
            master["Date"] = pd.to_datetime(master["Date"], errors="coerce")
            master = master.merge(snap, on="Date", how="left")
            master["Promotional_Flag"] = master["snap_flag"].fillna(0).astype(int)
            master.drop(columns=["snap_flag"], inplace=True, errors="ignore")

        metadata = {
            "edges_storage": self.sg.load_edges_storage(),
            "edges_plant": self.sg.load_edges_plant(),
            "edges_product_group": self.sg.load_edges_product_group(),
            "edges_product_subgroup": self.sg.load_edges_product_subgroup(),
            "node_plant_storage": self.sg.load_node_plant_storage(),
            "node_product_group": self.sg.load_node_product_group(),
            "m5_calendar": self.m5.load_calendar(),
            "m5_prices": self.m5.get_price_lookup(),
        }
        return master, metadata


# ===========================================================================
# Synthetic Fallback (500 rows, all 40 SKUs represented)
# ===========================================================================

def _generate_synthetic(n_rows: int = 500) -> pd.DataFrame:
    """Generate realistic-looking synthetic supply-chain data."""
    rng = np.random.default_rng(42)
    skus = [f"SKU_{i:03d}" for i in range(1, 41)]
    product_groups = ["S", "P", "E", "A", "T"]
    sub_groups = ["SOS", "POV", "EEA", "ATW", "TAN"]
    plants = [str(p) for p in [2120, 2130, 2140, 2150, 2160, 2170, 2180, 2190, 2200]]
    storage_locs = [str(s) for s in range(100, 1400, 100)]

    dates = pd.date_range("2023-01-01", periods=n_rows // len(skus) + 1, freq="D")
    records = []
    for sku_idx, sku in enumerate(skus):
        pg = product_groups[sku_idx % len(product_groups)]
        sg = sub_groups[sku_idx % len(sub_groups)]
        plant = plants[sku_idx % len(plants)]
        sloc = storage_locs[sku_idx % len(storage_locs)]
        base_demand = rng.uniform(100, 2000)
        for date in dates[:n_rows // len(skus)]:
            sales = max(0, base_demand + rng.normal(0, base_demand * 0.2))
            records.append({
                "Date": date,
                "SkuId": sku,
                "ProductGroup": pg,
                "SubGroup": sg,
                "PlantId": plant,
                "StorageLocationId": sloc,
                "OnHandQty": rng.uniform(0, sales * 5),
                "UnitCost": rng.uniform(5, 50),
                "LeadTimeDays": int(rng.integers(1, 14)),
                "SalesOrderQty": sales,
                "ProductionQty": sales * rng.uniform(0.8, 1.3),
                "DeliveryQty": sales * rng.uniform(0.7, 1.0),
                "FactoryIssueQty": sales * rng.uniform(0.9, 1.1),
                "Promotional_Flag": int(rng.integers(0, 2)),
                "Price": rng.uniform(10, 100),
            })

    df = pd.DataFrame(records)
    return df[UNIFIED_SCHEMA].head(n_rows)


# ===========================================================================
# Schema Validation
# ===========================================================================

def validate_schema(df: pd.DataFrame) -> None:
    """Print a full schema validation report for the unified dataframe."""
    print("\n" + "=" * 60)
    print("  UNIFIED SAP OData SCHEMA VALIDATION REPORT")
    print("=" * 60)
    print(f"  Rows       : {len(df):,}")
    print(f"  Columns    : {df.shape[1]}")
    print(f"  Date range : {df['Date'].min()} -> {df['Date'].max()}")
    print(f"  Unique SKUs: {df['SkuId'].nunique()}")
    print(f"  Plants     : {df['PlantId'].nunique()}")
    print(f"  Storage loc: {df['StorageLocationId'].nunique()}")
    print(f"  Prod groups: {df['ProductGroup'].nunique()}")
    print()
    missing = [c for c in UNIFIED_SCHEMA if c not in df.columns]
    extra = [c for c in df.columns if c not in UNIFIED_SCHEMA]
    print(f"  Missing schema columns : {missing if missing else 'None'}")
    print(f"  Extra columns          : {extra if extra else 'None'}")
    print()
    print(f"  {'Column':<22} {'dtype':<12} {'nulls':>6}  {'min':>10}  {'max':>10}")
    print("  " + "-" * 64)
    for col in UNIFIED_SCHEMA:
        if col not in df.columns:
            print(f"  {col:<22} {'MISSING':<12}")
            continue
        dtype = str(df[col].dtype)
        nulls = df[col].isna().sum()
        try:
            mn = f"{df[col].min():.2f}"
            mx = f"{df[col].max():.2f}"
        except (TypeError, ValueError):
            mn = str(df[col].min())[:10]
            mx = str(df[col].max())[:10]
        print(f"  {col:<22} {dtype:<12} {nulls:>6}  {mn:>10}  {mx:>10}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    loader = UnifiedDataLoader()
    df, meta = loader.load()
    validate_schema(df)
    print(f"Metadata keys: {list(meta.keys())}")
    print(f"Edges (Storage): {meta['edges_storage'].shape}")
