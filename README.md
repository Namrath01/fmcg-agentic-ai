# Enterprise Agentic Supply Chain Optimizer
### Transforming FMCG Pack-Size Strategy on SAP BTP

**Author:** Namrath Basavaraju · MSc Data Science, University of Mannheim  
**Platform:** SAP BTP AI Core · SAP HANA Cloud · Joule Agents · SAP Analytics Cloud

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-3.2+-green)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple?logo=plotly&logoColor=white)
![SAP BTP](https://img.shields.io/badge/SAP_BTP-AI_Core-0070f3)

---

## Project Overview

This project demonstrates an **enterprise-grade agentic AI system** that optimises FMCG (Fast-Moving Consumer Goods) pack-size strategy using a multi-agent pipeline built on two real-world datasets:

- **SupplyGraph** (Wasi et al., AAAI 2024) — real FMCG supply chain data: 41 SKUs, 26 plants, 14 storage locations, 221 days of operational telemetry
- **M5 Forecasting** (Makridakis et al.) — Walmart demand data for 30,490 SKUs with calendar events and SNAP signals

The system is mapped to the **SAP BTP AI Core** architecture and demonstrates how Joule Agents, HANA Cloud Vector Engine, and SAP Analytics Cloud could deploy this in enterprise production.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE ERP CONTEXT (SAP BTP)                  │
│                                                                       │
│   SupplyGraph CSVs          M5 Forecasting Data                      │
│   (41 SKUs · 26 Plants)     (Calendar · Prices · SNAP)               │
│           │                         │                                 │
│           └──────────┬──────────────┘                                 │
│                      ▼                                                │
│              ┌───────────────┐                                        │
│              │  Data Loader  │  ← SAP HANA Cloud (persistence layer) │
│              │  (utils/)     │                                        │
│              └───────┬───────┘                                        │
│                      │                                                │
│              ┌───────▼───────────────────────────────────────┐       │
│              │           AGENT ORCHESTRATOR                   │       │
│              │         (orchestrator/orchestrator.py)         │       │
│              │                                                │       │
│  ┌───────────┼──────────────────────────────────────────┐    │       │
│  │           │                                           │    │       │
│  ▼           ▼           ▼           ▼           ▼       │    │       │
│ [1]         [2]         [3]         [4]         [5]      │    │       │
│ Demand    Pack Size   Financial  Production  Dispatch    │    │       │
│ Intel.    Optim.      Impact     Planning    Optim.      │    │       │
│  │          │          │           │           │         │    │       │
│  │     ◄────┤◄─ conflict resolution loop ──►   │         │    │       │
│  └──────────┴──────────┴───────────┴───────────┘         │    │       │
│              │                                            │    │       │
│              └────────────────────────────────────────────┘    │       │
│                      │                                                │
│              ┌───────▼───────┐                                        │
│              │  Streamlit    │  ← SAP Analytics Cloud (visualisation)│
│              │  Dashboard    │                                        │
│              │  (app.py)     │                                        │
│              └───────────────┘                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Five-Agent Pipeline

| # | Agent | Technology | SAP BTP Mapping |
|---|-------|-----------|-----------------|
| 1 | **Demand Intelligence** | RandomForestRegressor + M5 SNAP signals | SAP BTP AI Core (Model Serving) |
| 2 | **Pack Size Optimization** | KMeans clustering (3 velocity classes) | Joule Agents (decision layer) |
| 3 | **Financial Impact** | Revenue simulation + PBT modelling | SAP Analytics Cloud (Finance) |
| 4 | **Production Planning** | EOQ batch sizing + plant utilisation | SAP IBP (Integrated Business Planning) |
| 5 | **Dispatch Optimization** | NetworkX shortest-path routing | SAP TM (Transportation Management) |

---

## White-Paper Benchmark Targets

| KPI | Target Range | Source |
|-----|-------------|--------|
| PBT Uplift | **10–18%** | FMCG Pack-Size White Paper |
| Revenue Turnover Improvement | **8–15%** | Industry Benchmarks |
| Debtor Cycle Savings | **25–35%** | Working Capital Studies |
| Inventory Reduction | **20–30%** | Supply Chain Optimization |
| Lead Time Reduction | **20–30%** | Logistics Excellence Reports |

---

## Enterprise Platform Mapping

### SAP BTP AI Core
- Hosts and serves the RandomForest demand forecasting model (Agent 1)
- Manages model versioning and A/B testing between pack-size strategies
- Handles inference pipelines triggered by SAP S/4HANA sales order events

### SAP HANA Cloud Vector Engine
- Stores SKU embedding vectors for semantic similarity matching
- Enables fast nearest-neighbour lookups for comparable SKU demand profiles
- Powers the SupplyGraph topology queries

### Joule Agents (SAP AI)
- Orchestrates the 5-agent pipeline via natural language instructions
- Resolves financial conflicts between PackSize and Financial agents
- Surfaces recommendations into SAP Fiori / SAP Build Work Zone

### SAP Analytics Cloud
- Renders the 5-tab enterprise dashboard (equivalent of app.py)
- Provides drill-down into plant utilisation and dispatch network views
- Exports financial impact reports to SAP Financial Consolidation

---

## Datasets

### SupplyGraph
```
Wasi, N., Ahmed, S., & Anwaar, M. (2024).
SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks.
Proceedings of the AAAI Workshop on AI for Time Series Analysis (AI4TS).
```

### M5 Forecasting
```
Makridakis, S., Spiliotis, E., & Assimakopoulos, V.
M5 Competition: Uncertainty Edition.
Kaggle, 2020. https://www.kaggle.com/c/m5-forecasting-uncertainty
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Namrath01/fmcg-agentic-ai.git
cd fmcg-agentic-ai
```

### 2. Set up data folders
Move your dataset files:
```
data/
├── supplygraph/
│   ├── Temporal Data/
│   │   ├── Unit/
│   │   │   ├── Sales Order.csv
│   │   │   ├── Production .csv
│   │   │   ├── Delivery To distributor.csv
│   │   │   └── Factory Issue.csv
│   │   └── Weight/  (same structure)
│   ├── Nodes/
│   │   ├── Nodes.csv
│   │   ├── NodesIndex.csv
│   │   ├── Nodes Type (Plant & Storage).csv
│   │   └── Node Types (Product Group and Subgroup).csv
│   └── Edges/
│       ├── Edges (Plant).csv
│       ├── Edges (Storage Location).csv
│       ├── Edges (Product Group).csv
│       └── Edges (Product Sub-Group).csv
└── m5/
    ├── calendar.csv
    ├── sales_train_validation.csv
    ├── sell_prices.csv
    └── sales_train_evaluation.csv
```

> **Note:** If data is not moved yet, the system auto-detects the original `SupplyGraph/` and `M5/` folders and also includes a 500-row synthetic fallback.

### 3. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit dashboard
```bash
streamlit run app.py
```

### 6. (Optional) Run the EDA notebook
```bash
jupyter notebook notebooks/eda.ipynb
```

### 7. (Optional) Run the pipeline from CLI
```bash
python orchestrator/orchestrator.py
```

---

## Project Structure

```
fmcg-agentic-ai/
├── data/
│   ├── supplygraph/          ← SupplyGraph dataset (move here)
│   └── m5/                   ← M5 dataset (move here)
├── agents/
│   ├── __init__.py
│   ├── demand_intelligence.py      ← Agent 1: RF demand forecast
│   ├── pack_size_optimization.py   ← Agent 2: KMeans velocity clustering
│   ├── financial_impact.py         ← Agent 3: PBT & revenue modelling
│   ├── production_planning.py      ← Agent 4: EOQ batch planning
│   └── dispatch_optimization.py    ← Agent 5: NetworkX routing
├── orchestrator/
│   ├── __init__.py
│   └── orchestrator.py             ← 5-agent pipeline + conflict resolution
├── utils/
│   ├── __init__.py
│   └── data_loader.py              ← SupplyGraph + M5 data loading
├── notebooks/
│   └── eda.ipynb                   ← Full EDA across both datasets
├── app.py                          ← Streamlit 5-tab enterprise dashboard
├── METHODOLOGY.md                  ← Detailed methodology, datasets, agent design
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Dashboard | Streamlit 1.32+ |
| ML | scikit-learn (RandomForest, KMeans) |
| Graphs | NetworkX 3.2+ |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Data | Pandas, NumPy, OpenPyXL |
| Notebooks | Jupyter |

---

*Built as an enterprise portfolio project demonstrating agentic AI capabilities for SAP BTP supply chain optimisation.*  
*© 2026 Namrath Basavaraju · University of Mannheim*
