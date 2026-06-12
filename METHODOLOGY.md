# Methodology

**Project:** Enterprise Agentic Supply Chain Optimizer — Transforming FMCG Pack-Size Strategy on SAP BTP  
**Author:** Namrath Basavaraju, MSc Data Science, University of Mannheim

---

## 1. Datasets

### 1.1 SupplyGraph (Wasi et al., AAAI 2024)
A real-world supply chain graph dataset covering 41 SKUs, 26 production plants, 14 storage locations, and 221 days of temporal data. Files used:

| File | Description |
|---|---|
| `Sales Order.csv` | Daily sales order quantities per SKU |
| `Production.csv` | Daily production quantities per SKU per plant |
| `Delivery To Distributor.csv` | Daily outbound delivery quantities |
| `Factory Issue.csv` | Goods issued from factory (includes buffer drawdown) |
| `Edges (Storage Location).csv` | Graph edges between storage nodes |

**Schema unification:** All wide-format CSVs are melted to long format and mapped to a unified SAP OData-style schema with columns: `Date`, `SkuId`, `ProductGroup`, `SubGroup`, `PlantId`, `StorageLocationId`, `OnHandQty`, `UnitCost`, `LeadTimeDays`, `SalesOrderQty`, `ProductionQty`, `DeliveryQty`, `FactoryIssueQty`, `Promotional_Flag`, `Price`.

> **Note on FactoryIssueQty:** In SupplyGraph, "Factory Issue" represents goods dispatched from the factory to distribution, not a defect rate. It can exceed same-day production due to buffer drawdown. The factory issue rate used in production planning is computed as the absolute deviation from a 1:1 issue-to-production ratio, capped at 1.0.

### 1.2 Walmart M5 Forecasting Dataset (Makridakis et al.)
Provides 1,913 days of hierarchical retail sales data across 30,490 items, 10 stores, and 3 US states. Features used:

| Feature | Usage |
|---|---|
| `sell_prices.csv` | Unit price lookup enriching SupplyGraph UnitCost |
| `calendar.csv` | SNAP flags, event types, weekday/weekend signals |

M5 calendar data is joined to SupplyGraph by date. The resulting `Promotional_Flag` column (derived from any-state SNAP activity) and M5 sell prices are passed to the demand feature matrix and financial agent respectively. The raw sales volume files (`sales_train_*.csv`) are not used in the pipeline.

---

## 2. Agent Architecture

The pipeline comprises five sequential agents with a conflict-resolution loop between Agent 2 and Agent 3.

```
UnifiedDataLoader
      |
      v
Agent 1: DemandIntelligenceAgent
      |
      v
Agent 2: PackSizeOptimizationAgent  <----+
      |                                   |
      v                                   | conflict loop (max 2 iterations)
Agent 3: FinancialImpactAgent  ----------+
      |
      v
Agent 4: ProductionPlanningAgent
      |
      v
Agent 5: DispatchOptimizationAgent
```

---

## 3. Agent Methodologies

### Agent 1 — Demand Intelligence (DemandIntelligenceAgent)

**Model:** RandomForestRegressor (scikit-learn, 100 estimators)

**Features (12 total):**

| Feature | Type | Rationale |
|---|---|---|
| `lag_7`, `lag_14` | Temporal lag | Weekly and fortnightly demand carry-over |
| `rolling_mean_7`, `rolling_std_7` | Rolling statistics | Trend and volatility signal |
| `day_of_week`, `month`, `day_of_year`, `week_of_year` | Calendar | Seasonality and day-pattern effects |
| `Promotional_Flag` | Event | M5 SNAP/promotion uplift signal |
| `SkuId_enc`, `ProductGroup_enc`, `SubGroup_enc` | Categorical | SKU identity encoding |

**Train/test split:** Chronological 80/20 split. Rows are sorted by `[SkuId, Date]` before splitting. Random shuffling is explicitly avoided to prevent future data from leaking into the training window — a common source of over-optimistic MAPE in time-series models.

**Confidence intervals:** Derived from the variance across individual tree predictions (10th and 90th percentile of the ensemble).

**Evaluation:** MAPE and RMSE on the held-out 20% chronological tail. On the SupplyGraph dataset (high-frequency FMCG with sparse demand for some SKUs), MAPE is typically in the 200–300% range — common for intermittent demand series where many actuals are near zero. RMSE (~200 units) is the more meaningful error metric for this data distribution.

---

### Agent 2 — Pack Size Optimization (PackSizeOptimizationAgent)

**Method:** KMeans clustering (k=3, scikit-learn) on four velocity features: `hist_mean`, `hist_std`, `hist_cv`, `fcast_mean`. Features are standardised with `StandardScaler` before clustering.

**Cluster-to-velocity mapping:** Clusters are ranked by their centroid's `hist_mean` dimension. The highest centroid is labelled `fast_mover`, middle is `medium_mover`, lowest is `slow_mover`.

**Pack configurations:**

| Velocity Class | Pack Type | Multiplier | Cost Factor |
|---|---|---|---|
| `fast_mover` | Bulk Logistics Case | 24x | 0.85 |
| `medium_mover` | Standard Consumer Pack | 12x | 1.00 |
| `slow_mover` | Promotional Multipack | 6x | 1.10 |

**Estimated inventory reduction (deterministic):**

| Velocity Class | Reduction |
|---|---|
| `fast_mover` | 30.0% |
| `medium_mover` | 23.0% |
| `slow_mover` | 15.0% |

**Justification score (0–100):** Composite of velocity score (up to 60 points, proportional to relative demand) and stability score (up to 40 points, inversely proportional to coefficient of variation).

**Conflict handling:** If FinancialImpactAgent flags unprofitable SKUs, those SKUs are downgraded to `slow_mover` pack configuration and their justification scores are penalised by 30%.

---

### Agent 3 — Financial Impact (FinancialImpactAgent)

**Financial constants (sourced from industry benchmarks):**

| Parameter | Value | Source |
|---|---|---|
| `CARRYING_COST_RATE` | 22% p.a. | Deloitte Working Capital Report 2023: FMCG avg 20–25% |
| `COGS_RATIO` | 60% of revenue | Industry benchmark: FMCG COGS typically 55–65% |
| `OVERHEAD_RATIO` | 15% of revenue | Standard FMCG overhead allocation: 12–18% |
| `DEBTOR_DAYS_BEFORE` | 45 days | FMCG industry baseline |
| Financing rate | 6% p.a. | Applied to debtor day cost calculation |

**Revenue uplift factors (by velocity class):**

| Class | Uplift Factor | Rationale |
|---|---|---|
| `fast_mover` | 1.12 (+12%) | High-volume bulk pack drives continuous replenishment |
| `medium_mover` | 1.08 (+8%) | Standard shelf pack improves distribution efficiency |
| `slow_mover` | 1.05 (+5%) | Promotional multipack stimulates sluggish lines |

**Debtor improvement (deterministic, by velocity class):**

| Class | Improvement |
|---|---|
| `fast_mover` | 30% |
| `medium_mover` | 27% |
| `slow_mover` | 25% |

**PBT computation:**
```
Before PBT = Revenue - COGS - Overhead - CarryingCost - DebtorCost
After  PBT = AfterRevenue - AfterCOGS - AfterOverhead - AfterCarryingCost - AfterDebtorCost
```

**Benchmark ranges** (McKinsey Global Institute, Gartner Supply Chain Research, SymphonyAI FMCG reports):

| Metric | Target Range |
|---|---|
| PBT Uplift | 10–18% |
| Revenue Turnover Improvement | 8–15% |
| Inventory Reduction | 20–30% |
| Debtor Cycle Savings | 25–35% |

> These are published industry benchmarks. They represent the expected range of improvement achievable through pack-size rationalisation in a real FMCG deployment. They are not empirical results from this simulation.

---

### Agent 4 — Production Planning (ProductionPlanningAgent)

**Batch sizing:** Economic Order Quantity (EOQ) formula:
```
EOQ = sqrt(2 * AnnualDemand * SetupCost / HoldingCostPerUnit)
```
Setup cost fixed at £500 per run. Holding cost = unit cost × 22% (carrying cost rate). Batch sizes are capped at a minimum of 50 units.

**Lead time calculation:**
```
EstLeadTimeDays = ceil(OptimalBatchSize / TargetDailyProduction)
```
Capped to the range [1, 30] days to prevent unrealistic values from low-production SKUs.

**Target utilisation:** 82% of estimated plant capacity (max observed daily production × 1.1 headroom). Hard cap at 95% to avoid overload.

**Factory issue rate:** Computed as `clip(|FactoryIssueQty / ProductionQty - 1.0|, 0, 1)`. A rate above 8% flags a plant as high-risk.

---

### Agent 5 — Dispatch Optimization (DispatchOptimizationAgent)

**Network construction:** NetworkX undirected graph. Nodes are storage locations derived from `StorageLocationId` values in master data. Edges are sourced from `Edges (Storage Location).csv`. Edge weights are `1 / (AvgDeliveryQty + 1)` — lower delivery volume implies higher time/cost.

**Routing:** All-pairs shortest paths between storage location nodes using Dijkstra's algorithm (NetworkX `shortest_path` with `weight="weight"`).

**Current lead time assignment:** Normalised from delivery velocity to a realistic 5–15 day range. High-volume (fast) SKUs receive shorter lead times; low-volume (slow) SKUs receive longer ones.

**Lead time reduction (deterministic):**
```
norm_lt = (CurrentLeadTimeDays - 5) / 10  # 0 = fastest, 1 = slowest
reduction_pct = 29 - (norm_lt * 8)        # range: 21% to 29%
OptimisedLeadTimeDays = CurrentLeadTimeDays * (1 - reduction_pct / 100)
```
This produces a 21–29% reduction band, centred at approximately 25%, consistent with the 20–30% white-paper benchmark.

---

## 4. SAP BTP AI Core Simulation

The project maps the five-agent pipeline onto SAP Business Technology Platform (BTP) components for illustrative purposes:

| Pipeline Component | SAP BTP Analogue |
|---|---|
| Agent orchestration | SAP BTP AI Core — Joule Agent orchestration layer |
| Vector feature store | SAP HANA Cloud Vector Engine |
| Financial KPI dashboard | SAP Analytics Cloud (SAC) |
| Data integration layer | SAP Integration Suite / OData APIs |
| Model serving | SAP AI Core ML deployment endpoints |

> **Disclaimer:** This application uses SAP BTP terminology for portfolio demonstration purposes. It does not connect to any SAP servers, SAP BTP tenants, or live SAP services. All computation runs locally using open-source Python libraries.

---

## 5. Limitations and Future Work

| Limitation | Mitigation / Future Work |
|---|---|
| SupplyGraph covers only 221 days | Extend with synthetic time-series augmentation |
| MAPE is high (~250%) due to intermittent demand near zero | Use RMSE/MAE as primary metrics; per-SKU ARIMA/Prophet ensemble |
| Financial uplift figures are model-derived estimates | Validate against actual A/B test or pilot deployment data |
| Static pack configurations (3 tiers) | Dynamic pack sizing using reinforcement learning |
| No real-time data ingestion | Connect to SAP OData live feeds via BTP Integration Suite |
| Dispatch routing ignores transport cost | Extend with distance matrix and carrier rate cards |

---

## 6. References

- Wasi, A. N., et al. (2024). *SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks.* AAAI 2024 Workshop on Graphs in Machine Learning.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). *M5 accuracy competition: Results, findings and conclusions.* International Journal of Forecasting.
- McKinsey Global Institute. *The state of fashion: Value creation in the age of disruption.* (Supply chain KPI benchmarks.)
- Gartner Supply Chain Research. *FMCG Supply Chain Performance Benchmarks 2023.*
- SymphonyAI. *FMCG AI-Driven Supply Chain Optimisation: Industry Benchmark Report.*
- Deloitte. *Working Capital Report 2023: Unlocking cash in consumer products.*
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
- MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations.* Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability.
