# ML-Guided Dynamic Pricing & Inventory Allocation
## Full Project Build Pipeline (use as a prompt for Claude Code / Cursor / etc.)

---

## 0. Project Framing (map to JD accountabilities before writing any code)

**Business problem statement (write this first, in your README):**
"A multi-location retailer wants to set weekly prices and allocate limited inventory
across stores/warehouses to maximize total revenue, subject to demand uncertainty,
capacity constraints, and business rules (max price change per week, minimum stock
per location). This mirrors real Revenue Management problems solved by OR teams at
airlines, retailers, and logistics companies."

This framing matters because it mirrors the JD's own opening: "partner with business
and technology leaders... to drive incremental revenue, reduce costs, and boost
customer engagement by leveraging state-of-the-art Operations Research techniques."

Map every pipeline stage below to a JD accountability:
| Pipeline Stage | JD Accountability |
|---|---|
| Data cleaning, EDA, business framing | #1 – partnering to identify opportunities, gathering/analyzing data |
| Forecasting model + MILP + GA | #2 – delivering high quality OR solutions, documented, reusable |
| Final report/dashboard/README | #3 – structuring communication of insights, clear presentation |
| Comparing exact vs heuristic, sensitivity checks | #4 – technical coordination/review, escalation of risks/limitations |

---

## 1. Dataset

**Primary:** Kaggle "Retail Price Optimization" dataset (product price, cost,
competitor price, quantity sold, category, date).

**Trim to:**
- Top 15–20 SKUs by transaction volume
- 2–3 "locations" (if the dataset has no location field, synthetically split
  by category or simulate 2-3 warehouses with different capacity limits —
  state this clearly as a documented assumption in your README)
- 6–12 months of weekly-aggregated data

**Optional enrichment:** MOSPI CPI (Consumer Price Index) data to sanity-check
your elasticity assumptions against real-world inflation-adjusted price sensitivity,
and to justify constraint bounds (e.g., "price changes capped at X% per week,
consistent with observed CPI volatility for this category").

---

## 2. Repository Structure

```
pricing-allocation-or/
├── data/
│   ├── raw/                     # original Kaggle CSV
│   ├── processed/                # cleaned, weekly-aggregated, SKU x location
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_demand_model.ipynb
│   ├── 03_milp_formulation.ipynb
│   ├── 04_metaheuristic_ga.ipynb
│   ├── 05_benchmark_comparison.ipynb
├── src/
│   ├── data_prep.py
│   ├── demand_model.py          # elasticity / forecasting model
│   ├── milp_solver.py           # OR-Tools / Gurobi exact formulation
│   ├── ga_solver.py             # genetic algorithm heuristic
│   ├── evaluate.py              # revenue lift, optimality gap, solve time
├── dashboard/
│   └── app.py                   # Streamlit demo
├── docs/
│   ├── formulation.md           # full math writeup (decision vars, objective, constraints)
│   ├── results_summary.md       # business-facing 1-pager (JD accountability #3)
├── README.md
└── requirements.txt
```

---

## 3. Stage 1 — Data Preparation & EDA

- Load raw CSV, clean nulls/outliers, aggregate to weekly SKU × location granularity.
- Compute historical price elasticity per SKU (simple log-log regression:
  `log(quantity) ~ log(price)`) as a sanity-check baseline before the ML model.
- Visualize demand vs. price scatter per SKU, seasonality trends, competitor
  price gaps.
- Document every cleaning/assumption decision in `docs/formulation.md` —
  this is what a stakeholder review would ask about (JD accountability #4).

**Deliverable:** `01_eda.ipynb` + a short written summary of key patterns found.

---

## 4. Stage 2 — Demand Forecasting Model (the "AI/ML modeling" JD bullet)

Build a model that predicts demand as a function of price, competitor price,
seasonality, and location:

- **Baseline:** Linear/log-log regression (interpretable elasticity coefficient
  per SKU — useful to report as a sanity check to non-technical stakeholders).
- **Main model:** Gradient boosting (XGBoost/LightGBM) or a small feed-forward
  network, predicting `quantity_sold` given `(price, competitor_price, week,
  location, category)`.
- Validate with time-based train/test split (not random split — demand
  forecasting must respect time order).
- Output: a **demand function per SKU per location**, `d_i(p_i)`, that the
  optimization layer will consume. This is the "predict-then-optimize" bridge
  between ML and OR — call this out explicitly in your writeup, it's the
  single most important technical decision in the project.

**Deliverable:** `demand_model.py`, with a function `predict_demand(sku, location,
price, week) -> expected_quantity`.

---

## 5. Stage 3 — Exact OR Formulation (core MILP — the JD's #1 requirement)

### Decision Variables
- `p_{i,l,t}` — price of SKU *i* at location *l* in week *t* (continuous or
  discretized into a small price grid, e.g., 5 price points per SKU, to keep
  it MILP rather than nonlinear)
- `x_{i,l,t}` — units of SKU *i* allocated to location *l* in week *t* (integer)
- `y_{i,l,t}` — binary indicator: 1 if a given discrete price level is chosen
  for SKU *i* at location *l* in week *t* (needed to linearize price choice
  from a discrete grid — this is the standard trick to keep pricing MILP-solvable)

### Objective
Maximize total expected revenue minus cost:

```
max  Σ_i Σ_l Σ_t [ p_{i,l,t} * d_i(p_{i,l,t}) - c_i * x_{i,l,t} ]
```

where `d_i(p)` comes from your ML demand model (approximate with the
discretized price-grid values so the objective stays linear per price choice).

### Constraints
1. **Capacity:** `Σ_i x_{i,l,t} ≤ Capacity_l` (each location's shelf/warehouse
   limit)
2. **Demand feasibility:** `x_{i,l,t} ≤ d_i(p_{i,l,t})` (can't sell more than
   expected demand at that price)
3. **Price stability:** `|p_{i,l,t} - p_{i,l,t-1}| ≤ ΔMax` (business rule: no
   more than X% price change week over week)
4. **Single price choice:** `Σ_k y_{i,l,t,k} = 1` (exactly one discrete price
   level chosen per SKU/location/week)
5. **Minimum service level:** `x_{i,l,t} ≥ MinStock_l` (optional — ensures no
   location goes fully out of stock, a realistic business constraint)

### Solver
- **OR-Tools CP-SAT** (free, no license needed, very good at this problem
  size) as your primary solver.
- **Gurobi** (free academic license) as a secondary benchmark — mentioning
  both in your writeup directly echoes the JD's "solvers like Gurobi, CPLEX"
  bullet.

**Deliverable:** `milp_solver.py` with a clean `solve_pricing_milp(demand_model,
capacity, price_grid) -> solution` function, plus `docs/formulation.md` with
the full math written out (this document alone is a strong interview artifact).

---

## 6. Stage 4 — Metaheuristic for Scale (the "heuristics for solving OR
formulations efficiently" JD bullet)

- As you scale up SKUs × locations × weeks, the MILP will slow down or hit
  optimality gaps. Implement a **Genetic Algorithm** (or Simulated Annealing)
  that searches over price/allocation combinations directly:
  - Chromosome = vector of price choices per SKU/location/week
  - Fitness = same revenue objective as the MILP
  - Repair operator to enforce capacity/price-stability constraints
- Benchmark GA against the exact MILP on small instances (where MILP can
  still solve to optimality) to report **optimality gap** and **solve time**.
- Scale both up and show the crossover point where GA becomes necessary —
  this comparison chart is one of the most compelling things you can show
  in an interview.

**Deliverable:** `ga_solver.py`, `05_benchmark_comparison.ipynb` with a
solve-time-vs-instance-size chart and an optimality-gap table.

---

## 7. Stage 5 — Evaluation

Report these metrics clearly (this is your "results" story):
- **Revenue lift** vs. a naive baseline (e.g., static historical pricing)
- **Optimality gap** of GA vs. exact MILP on comparable instances
- **Solve time** scaling curve (MILP vs. GA as problem size grows)
- **Constraint sensitivity**: how much revenue changes if you loosen/tighten
  the price-stability constraint — this kind of sensitivity analysis is
  exactly what a stakeholder review meeting would ask for (JD accountability #4)

---

## 8. Stage 6 — Demo & Communication (JD accountability #3)

Build a lightweight **Streamlit dashboard** (`dashboard/app.py`) showing:
- Input: pick a SKU/location/time range
- Output: forecasted demand curve, optimized price recommendation, allocation
  across locations, revenue lift vs. baseline, and a toggle to compare
  MILP vs. GA solutions

Also write a **one-page business summary** (`docs/results_summary.md`) — no
math, just: problem, approach, key result, and limitations. This is the
document you'd actually hand to a "team leader/manager/stakeholder," per
the JD's own wording.

---

## 9. Suggested Tech Stack

- **Data/ML:** pandas, scikit-learn, XGBoost/LightGBM
- **OR:** OR-Tools (CP-SAT), Gurobi (academic license), PuLP as a lightweight
  fallback
- **Metaheuristic:** DEAP (Python GA library) or a hand-rolled GA for full
  control/transparency
- **Demo:** Streamlit
- **Docs:** Markdown + one diagram (pipeline flowchart) in the README

---

## 10. README Checklist (what a reviewer will actually look for)

- [ ] Clear business problem statement (no jargon in first paragraph)
- [ ] Data source + assumptions documented
- [ ] ML model performance (accuracy/MAE on demand forecast)
- [ ] Full MILP formulation written in math notation
- [ ] Exact solver results + GA benchmark comparison (chart)
- [ ] Revenue lift number vs. baseline (headline result)
- [ ] Limitations section (be honest — this signals seniority, not weakness)
- [ ] Link to live Streamlit demo or a demo video/GIF

---

## Prompt block (copy-paste this into Claude Code / Cursor to start building)

```
Build a Python project called "pricing-allocation-or" implementing the pipeline
described above: (1) load and clean the Kaggle Retail Price Optimization dataset,
aggregate to weekly SKU x location level with 2-3 simulated locations, (2) train
a demand forecasting model (XGBoost) predicting quantity sold from price,
competitor price, and seasonality, (3) formulate and solve a MILP in OR-Tools
CP-SAT that jointly optimizes discrete pricing and inventory allocation across
locations to maximize revenue subject to capacity, price-stability, and minimum-
stock constraints, (4) implement a Genetic Algorithm alternative and benchmark
it against the exact MILP on solve time and optimality gap across increasing
problem sizes, (5) build a Streamlit dashboard visualizing demand curves,
optimized prices/allocations, and revenue lift vs. a static-pricing baseline,
and (6) write a docs/formulation.md with the full mathematical formulation
(decision variables, objective, constraints) and a docs/results_summary.md
one-page business summary. Set up the repo structure as: data/, notebooks/,
src/, dashboard/, docs/. Use OR-Tools as the primary solver and structure
src/milp_solver.py with a clean solve_pricing_milp() function.
```
