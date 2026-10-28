# Dynamic Pricing & Inventory Allocation — Mathematical Formulation

This document lays out the full mathematical formulation for the joint pricing
and inventory allocation optimization problem, matching the JD's requirement of
"state-of-the-art Operations Research techniques."

---

## 1. Problem Definition

A multi-location retailer sets weekly prices and allocates limited inventory
across stores/warehouses to **maximize total revenue**, subject to demand
uncertainty, capacity constraints, and business rules.

**Indices:**
- \( i \in \mathcal{I} \) — SKUs
- \( l \in \mathcal{L} \) — locations (stores/warehouses)
- \( t \in \mathcal{T} \) — time periods (weeks)
- \( k \in \mathcal{K}_i \) — discrete price levels for SKU \( i \)

**Sets:**
- \( \mathcal{I} \) — set of SKUs
- \( \mathcal{L} \) — set of locations
- \( \mathcal{T} \) — set of time periods
- \( \mathcal{K}_{i,l} \) — set of price levels available for SKU \( i \) at location \( l \)

---

## 2. Data & Parameters

| Notation | Description | Source |
|---|---|---|
| \( C_l \) | Weekly capacity (units) of location \( l \) | Business input |
| \( c_i \) | Unit cost of SKU \( i \) | Cost data |
| \( d_i(p) \) | Expected demand for SKU \( i \) at price \( p \) | ML model |
| \( p^G_{i,l,k} \) | The \( k \)-th price grid level for SKU \( i \) at location \( l \) | Price grid |
| \( \Delta_{max} \) | Maximum allowed price change week-over-week | Business rule |
| \( S_l \) | Minimum stock level for location \( l \) | Business rule |
| \( q_{i,l,k,t} \) | Expected demand when price \( p^G_{i,l,k} \) is chosen | ML model output |

**Price grid:** Each SKU \( i \) at location \( l \) has a discretized price grid
\( \mathcal{K}_{i,l} \) with \( n_{price} \) levels. Discretization keeps the
problem linear (MILP) rather than nonlinear, matching standard Revenue
Management practice.

---

## 3. Decision Variables

**Binary price selection:**
\[
y_{i,l,k,t} = \begin{cases}
1 & \text{if price level } k \text{ is chosen for SKU } i \text{ at location } l \text{ in week } t\\
0 & \text{otherwise}
\end{cases}
\]

**Inventory allocation:**
\[
x_{i,l,t} \in \mathbb{Z}_{\ge 0}
\]
Units of SKU \( i \) allocated to location \( l \) in week \( t \).

**Effective price:**
\[
p_{i,l,t} = \sum_{k \in \mathcal{K}_{i,l}} p^G_{i,l,k} \cdot y_{i,l,k,t}
\]

---

## 4. Objective Function

Maximize total expected profit over the planning horizon:

\[
\max \sum_{i \in \mathcal{I}} \sum_{l \in \mathcal{L}} \sum_{t \in \mathcal{T}}
\left[ p_{i,l,t} \cdot \min(x_{i,l,t}, d_i(p_{i,l,t})) - c_i \cdot x_{i,l,t} \right]
\]

Since we enforce \( x_{i,l,t} \le d_i(p_{i,l,t}) \) (demand feasibility constraint),
the objective simplifies to:

\[
\max \sum_{i \in \mathcal{I}} \sum_{l \in \mathcal{L}} \sum_{t \in \mathcal{T}}
\left( p_{i,l,t} \cdot x_{i,l,t} - c_i \cdot x_{i,l,t} \right)
\]

**Linearization via price grid:**
\[
p_{i,l,t} \cdot x_{i,l,t} = \sum_{k \in \mathcal{K}_{i,l}} p^G_{i,l,k} \cdot x_{i,l,t} \cdot y_{i,l,k,t}
\]

This bilinear term is linearized with a standard McCormick relaxation / indicator
constraint approach using auxiliary variables \( z_{i,l,k,t} = x_{i,l,t} \cdot y_{i,l,k,t} \):
\[
z_{i,l,k,t} \le M \cdot y_{i,l,k,t}
\]
\[
z_{i,l,k,t} \le x_{i,l,t}
\]
\[
z_{i,l,k,t} \ge x_{i,l,t} - M(1 - y_{i,l,k,t})
\]
for a sufficiently large \( M \).

---

## 5. Constraints

### 5.1 Capacity Constraints
\[
\sum_{i \in \mathcal{I}} x_{i,l,t} \le C_l, \quad \forall l \in \mathcal{L}, t \in \mathcal{T}
\]

### 5.2 Demand Feasibility
\[
x_{i,l,t} \le q_{i,l,k,t}, \quad \forall i, l, t, k
\]
(with enforcement only when \( y_{i,l,k,t} = 1 \))
\[
x_{i,l,t} \le \sum_{k} q_{i,l,k,t} \cdot y_{i,l,k,t}
\]

### 5.3 Single Price Choice
\[
\sum_{k \in \mathcal{K}_{i,l}} y_{i,l,k,t} = 1, \quad \forall i, l, t
\]

### 5.4 Price Stability (Week-over-Week)
\[
\left| p_{i,l,t} - p_{i,l,t-1} \right| \le \Delta_{max}, \quad \forall i, l, t \ge 2
\]

**Linearization of absolute value:**
\[
p_{i,l,t} - p_{i,l,t-1} \le \Delta_{max}
\]
\[
p_{i,l,t-1} - p_{i,l,t} \le \Delta_{max}
\]

For discrete price grids, this is enforced pairwise:
\[
y_{i,l,k_1,t} = 1 \land y_{i,l,k_2,t-1} = 1 \implies \left| p^G_{i,l,k_1} - p^G_{i,l,k_2} \right| \le \Delta_{max}
\]

### 5.5 Minimum Service Level (Optional)
\[
x_{i,l,t} \ge S_l \cdot \max(0, \mathbb{1}_{y_{i,l,k,t}=1 \text{ for some } k})
\]
enforced as:
\[
x_{i,l,t} \ge S_l \cdot \sum_k y_{i,l,k,t} \cdot 1_{q_{i,l,k,t} > 0}
\]

### 5.6 Integrality
\[
y_{i,l,k,t} \in \{0, 1\}, \quad x_{i,l,t} \in \mathbb{Z}_{\ge 0}
\]

---

## 6. Pred-then-Optimize Bridge

The demand function \( d_i(p) \) is learned via ML (gradient boosting / neural
network) and then **discretized to the price grid** for the MILP. This is the
most important technical decision:

- **Benefit:** Keeps the problem linear and solvable by CP-SAT/Gurobi/CPLEX at scale
- **Trade-off:** Loses the continuous price precision of the true demand curve
- **Mitigation:** Use a fine enough grid (e.g., 8–20 points per SKU) to keep
  discretization error negligible

---

## 7. Solver Choices

| Solver | Type | Pros | Used For |
|---|---|---|---|
| OR-Tools CP-SAT | Exact MILP | Free, fast, no license | Primary solver |
| Gurobi | Exact MILP | Industry standard, faster | Benchmark |
| PuLP | Exact LP/MILP | Simple interface | Lightweight fallback |
| Custom GA (DEAP) | Heuristic | Scales to large instances | Large problems |
| Simulated Annealing | Heuristic | Simpler, robust | Alternative heuristic |

---

## 8. Genetic Algorithm Formulation

For problem instances where MILP becomes intractable (large \( \mathcal{I} \times
\mathcal{L} \times \mathcal{T} \)):

- **Chromosome:** vector \( \mathbf{c} \) where \( c_{i,l,t} \in \{0, 1, \ldots, |\mathcal{K}^G_{i,l}| - 1\} \) indexes the chosen price level
- **Fitness function:**
\[
f(\mathbf{c}) = \sum_{i,l,t} \left( p^G_{i,l,c_{i,l,t}} - c_i \right) \cdot \min\left( d_i(p^G_{i,l,c_{i,l,t}}), C_l \right)
\]
- **Repair operators:** enforce capacity and price stability post-mutation
- **Selection:** tournament selection with elitism
- **Crossover:** uniform or single-point with crossover probability
- **Mutation:** randomly reassign price level with mutation probability

---

## 9. Evaluation Metrics

**Revenue lift:**
\[
\text{Lift}\% = \frac{\text{Optimal revenue} - \text{Baseline revenue}}{\text{Baseline revenue}} \times 100
\]

**Optimality gap:**
\[
\text{Gap}\% = \frac{\text{MILP objective} - \text{GA objective}}{\text{MILP objective}} \times 100
\]

**Solve time scaling:** Measured across increasing product of
\( |\mathcal{I}| \times |\mathcal{L}| \times |\mathcal{T}| \).

---

## 10. Assumptions & Limitations

1. **Price grid discretization** — prices are restricted to a fixed grid, which
   approximates the continuous decision space.
2. **Demand model predictive accuracy** — the MILP optimizes against *expected*
   demand; forecast errors propagate to suboptimal allocations.
3. **No cross-SKU substitution effects** — demand for each SKU is assumed
   independent, which may overestimate aggregate demand for substitutes.
4. **Static capacities** — capacities are weekly and fixed, ignoring lead times.
5. **Synthetic locations** — dataset has no location field; locations are
   simulated. Real deployment requires actual store/warehouse data.