"""
MILP Solver Module

Formulates and solves the Dynamic Pricing & Inventory Allocation problem
as a Mixed-Integer Linear Program using OR-Tools CP-SAT.

Decision Variables:
  p_{i,l,t,k} - binary: 1 if SKU i at location l in week t uses price level k
  x_{i,l,t}   - integer: units of SKU i allocated to location l in week t
  y_{i,l,t}   - binary: 1 if SKU i at location l in week t has any sale

Objective: Maximize total expected revenue minus cost

Constraints:
  1. Capacity: total units per location <= capacity
  2. Demand feasibility: allocation <= expected demand at chosen price
  3. Price stability: |price_t - price_{t-1}| <= max_change
  4. Single price choice: exactly one price level per SKU/location/week
  5. Minimum service level: allocation >= minimum stock (optional)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

from ortools.sat.python import cp_model


SOLVER_TIMEOUT_SECONDS = 120


class PricingMILP:
    """MILP formulation for joint pricing and inventory allocation optimization."""

    def __init__(self, n_skus: int, n_locations: int, n_weeks: int,
                 price_grids: Dict[Tuple[str, str], np.ndarray],
                 demand_grids: Dict[Tuple[str, str, int], Dict[float, float]],
                 costs: Dict[str, float],
                 capacities: Dict[str, int],
                 max_price_change: float = 15.0,
                 min_stock: int = 5,
                 demand_multiplier: float = 1.0):
        """
        Args:
            n_skus: number of SKUs
            n_locations: number of locations
            n_weeks: number of weeks
            price_grids: {(sku, location): array of price levels}
            demand_grids: {(sku, location, week): {price: expected_demand}}
            costs: {sku: unit_cost}
            capacities: {location: weekly_capacity}
            max_price_change: maximum allowed price change week-over-week
            min_stock: minimum units to allocate per SKU/location/week
            demand_multiplier: multiplier on demand (allows relaxation)
        """
        self.n_skus = n_skus
        self.n_locations = n_locations
        self.n_weeks = n_weeks
        self.price_grids = price_grids
        self.demand_grids = demand_grids
        self.costs = costs
        self.capacities = capacities
        self.max_price_change = max_price_change
        self.min_stock = min_stock
        self.demand_multiplier = demand_multiplier

        self.sku_list = sorted(set(k[0] for k in price_grids.keys()))
        self.location_list = sorted(set(k[1] for k in price_grids.keys()))
        self.week_list = sorted(set(k[2] for k in demand_grids.keys()))

        self.sku_idx = {s: i for i, s in enumerate(self.sku_list)}
        self.loc_idx = {l: i for i, l in enumerate(self.location_list)}
        self.week_idx = {w: i for i, w in enumerate(self.week_list)}

        self.model = None
        self.solution = None

    def build(self) -> cp_model.CpModel:
        """Build the MILP model."""
        model = cp_model.CpModel()

        self.price_vars = {}
        self.alloc_vars = {}
        self.demand_limit_vars = {}

        for sku in self.sku_list:
            for loc in self.location_list:
                grid = self.price_grids.get((sku, loc), np.array([10.0]))
                n_prices = len(grid)

                for week in self.week_list:
                    price_key = (sku, loc, week)
                    alloc_key = (sku, loc, week)

                    self.price_vars[price_key] = [
                        model.NewBoolVar(f"p_{sku}_{loc}_{week}_k{k}")
                        for k in range(n_prices)
                    ]

                    max_demand = 0
                    if (sku, loc, week) in self.demand_grids:
                        demands = self.demand_grids[(sku, loc, week)]
                        max_demand = max(demands.values()) if demands else 0

                    max_alloc = int(max(self.demand_multiplier * max_demand, 1))
                    self.alloc_vars[alloc_key] = model.NewIntVar(
                        0, max_alloc, f"x_{sku}_{loc}_{week}"
                    )

        self._add_single_price_constraint(model)
        self._add_capacity_constraint(model)
        self._add_demand_feasibility_constraint(model)
        self._add_price_stability_constraint(model)
        self._add_minimum_stock_constraint(model)

        self._set_objective(model)

        self.model = model
        return model

    def _add_single_price_constraint(self, model: cp_model.CpModel):
        """Exactly one price level per SKU/location/week."""
        for sku in self.sku_list:
            for loc in self.location_list:
                for week in self.week_list:
                    price_key = (sku, loc, week)
                    if price_key in self.price_vars:
                        model.Add(
                            sum(self.price_vars[price_key]) == 1
                        )

    def _add_capacity_constraint(self, model: cp_model.CpModel):
        """Total allocation per location per week <= capacity."""
        for loc in self.location_list:
            capacity = self.capacities.get(loc, 1000)
            for week in self.week_list:
                allocs = []
                for sku in self.sku_list:
                    key = (sku, loc, week)
                    if key in self.alloc_vars:
                        allocs.append(self.alloc_vars[key])
                if allocs:
                    model.Add(sum(allocs) <= capacity)

    def _add_demand_feasibility_constraint(self, model: cp_model.CpModel):
        """Allocation <= expected demand at chosen price level."""
        for sku in self.sku_list:
            for loc in self.location_list:
                grid = self.price_grids.get((sku, loc), np.array([10.0]))
                for week in self.week_list:
                    key = (sku, loc, week)
                    if key not in self.price_vars:
                        continue

                    if key in self.demand_grids:
                        demands = self.demand_grids[key]
                        for k, price in enumerate(grid):
                            demand_at_price = demands.get(price, 0)
                            max_alloc = int(self.demand_multiplier * demand_at_price)
                            model.Add(
                                self.alloc_vars[key] <= max_alloc
                            ).OnlyEnforceIf(self.price_vars[key][k])

    def _add_price_stability_constraint(self, model: cp_model.CpModel):
        """Price change week-over-week <= max_price_change (linearized)."""
        for sku in self.sku_list:
            for loc in self.location_list:
                grid = self.price_grids.get((sku, loc), np.array([10.0]))

                for idx in range(1, len(self.week_list)):
                    prev_week = self.week_list[idx - 1]
                    curr_week = self.week_list[idx]

                    prev_key = (sku, loc, prev_week)
                    curr_key = (sku, loc, curr_week)

                    if prev_key not in self.price_vars or curr_key not in self.price_vars:
                        continue

                    prev_grid = self.price_grids.get((sku, loc), grid)
                    curr_grid = self.price_grids.get((sku, loc), grid)

                    for k_prev, p_prev in enumerate(prev_grid):
                        for k_curr, p_curr in enumerate(curr_grid):
                            price_diff = abs(p_curr - p_prev)
                            if price_diff > self.max_price_change:
                                model.AddBoolOr([
                                    self.price_vars[prev_key][k_prev].Not(),
                                    self.price_vars[curr_key][k_curr].Not(),
                                ])

    def _add_minimum_stock_constraint(self, model: cp_model.CpModel):
        """Minimum service level: allocation >= min_stock for active SKU/location/week."""
        if self.min_stock <= 0:
            return

        for sku in self.sku_list:
            for loc in self.location_list:
                for week in self.week_list:
                    key = (sku, loc, week)
                    if key in self.alloc_vars and key in self.price_vars:
                        model.Add(
                            self.alloc_vars[key] >= self.min_stock
                        ).OnlyEnforceIf(self.price_vars[key])

    def _set_objective(self, model: cp_model.CpModel):
        """Maximize total expected revenue minus cost."""
        profit_terms = []

        for sku in self.sku_list:
            cost = self.costs.get(sku, 5.0)
            for loc in self.location_list:
                grid = self.price_grids.get((sku, loc), np.array([10.0]))
                for week in self.week_list:
                    key = (sku, loc, week)
                    if key not in self.price_vars or key not in self.alloc_vars:
                        continue

                    if key in self.demand_grids:
                        demands = self.demand_grids[key]
                        for k, price in enumerate(grid):
                            demand = demands.get(price, 0)
                            revenue_per_unit = int(price * 100)
                            cost_per_unit = int(cost * 100)

                            revenue_profit = revenue_per_unit - cost_per_unit
                            scaled_demand = int(self.demand_multiplier * demand * 100)

                            profit_terms.append(
                                (revenue_profit, scaled_demand,
                                 self.price_vars[key][k], self.alloc_vars[key])
                            )

        objective_vars = []
        for revenue_profit, scaled_demand, price_var, alloc_var in profit_terms:
            profit_var = model.NewIntVar(0, revenue_profit * scaled_demand,
                                          f"profit_{len(objective_vars)}")
            model.AddMultiplicationEquality(
                profit_var,
                [price_var, alloc_var]
            )
            model.Add(
                profit_var <= revenue_profit * alloc_var
            )
            objective_vars.append(profit_var)

        model.Maximize(sum(objective_vars))

    def solve(self, timeout_seconds: int = SOLVER_TIMEOUT_SECONDS) -> Dict:
        """Solve the MILP and return results."""
        if self.model is None:
            self.build()

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = timeout_seconds
        solver.parameters.num_workers = 4

        start_time = time.time()
        status = solver.Solve(self.model)
        solve_time = time.time() - start_time

        status_names = {0: "OPTIMAL", 1: "FEASIBLE", 2: "INFEASIBLE",
                        3: "MODEL_INVALID", 4: "UNKNOWN"}
        status_name = status_names.get(status, f"STATUS_{status}")

        result = {
            "status": status,
            "status_name": status_name,
            "objective_value": solver.ObjectiveValue() if status == cp_model.OPTIMAL else 0,
            "solve_time_seconds": solve_time,
            "num_variables": len(list(self.model.Proto().variables)),
            "num_constraints": len(list(self.model.Proto().constraints)),
            "assignments": {},
        }

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for sku in self.sku_list:
                for loc in self.location_list:
                    for week in self.week_list:
                        key = (sku, loc, week)
                        if key not in self.price_vars:
                            continue

                        grid = self.price_grids.get((sku, loc), np.array([10.0]))
                        chosen_price = None
                        for k, price in enumerate(grid):
                            if solver.Value(self.price_vars[key][k]) == 1:
                                chosen_price = price
                                break

                        allocation = solver.Value(self.alloc_vars[key])

                        if chosen_price is not None:
                            result["assignments"][key] = {
                                "price": chosen_price,
                                "allocation": allocation,
                                "revenue": chosen_price * allocation,
                            }

        self.solution = result
        return result


def create_milp_from_data(weekly_df: pd.DataFrame,
                          demand_model,
                          capacities: Dict[str, int],
                          n_price_levels: int = 8,
                          max_price_change: float = 15.0,
                          min_stock: int = 5) -> PricingMILP:
    """Create a PricingMILP instance from processed data and a demand model.

    This is the main entry point for building the optimization model.
    """
    sku_list = weekly_df["product_id"].unique().tolist()
    location_list = weekly_df["location"].unique().tolist()
    week_list = sorted(weekly_df["week"].unique().tolist())

    price_grids = {}
    demand_grids = {}
    costs = {}

    for sku in sku_list:
        sku_data = weekly_df[weekly_df["product_id"] == sku]
        mean_price = sku_data["price"].mean()
        std_price = sku_data["price"].std()
        if std_price == 0 or np.isnan(std_price):
            std_price = mean_price * 0.1

        low = max(mean_price - 2 * std_price, mean_price * 0.3)
        high = mean_price + 2 * std_price
        grid = np.linspace(low, high, n_price_levels)
        grid = np.round(grid, 2)

        cost = sku_data["cost"].mean() if "cost" in sku_data.columns else mean_price * 0.5
        costs[sku] = cost

        for loc in location_list:
            price_grids[(sku, loc)] = grid

            loc_data = weekly_df[
                (weekly_df["product_id"] == sku) & (weekly_df["location"] == loc)
            ]
            mean_comp = loc_data["competitor_price"].mean() if "competitor_price" in loc_data.columns else mean_price

            for week in week_list:
                demand_fn = demand_model.create_demand_function(
                    competitor_price=mean_comp,
                    week=week,
                    category=sku_data["category"].iloc[0] if "category" in sku_data.columns else "default",
                    location=loc,
                )

                demands = {}
                for price in grid:
                    demands[price] = demand_fn(price)
                demand_grids[(sku, loc, week)] = demands

    if not capacities:
        capacities = {loc: 800 for loc in location_list}

    milp = PricingMILP(
        n_skus=len(sku_list),
        n_locations=len(location_list),
        n_weeks=len(week_list),
        price_grids=price_grids,
        demand_grids=demand_grids,
        costs=costs,
        capacities=capacities,
        max_price_change=max_price_change,
        min_stock=min_stock,
    )

    return milp


def solve_pricing_milp(weekly_df: pd.DataFrame,
                       demand_model,
                       capacities: Optional[Dict[str, int]] = None,
                       n_price_levels: int = 8,
                       max_price_change: float = 15.0,
                       min_stock: int = 5,
                       timeout_seconds: int = 120) -> Dict:
    """High-level function to formulate and solve the pricing MILP.

    Args:
        weekly_df: processed weekly data
        demand_model: fitted DemandModel instance
        capacities: {location: capacity} or None for defaults
        n_price_levels: number of discrete price points per SKU
        max_price_change: max week-over-week price change allowed
        min_stock: minimum allocation per SKU/location/week
        timeout_seconds: solver time limit

    Returns:
        Dictionary with solution status, assignments, and metrics
    """
    if capacities is None:
        location_list = weekly_df["location"].unique().tolist()
        capacities = {loc: 800 for loc in location_list}

    print("=" * 60)
    print("MILP FORMULATION & SOLVE")
    print("=" * 60)

    print("\nBuilding model...")
    milp = create_milp_from_data(
        weekly_df, demand_model, capacities,
        n_price_levels=n_price_levels,
        max_price_change=max_price_change,
        min_stock=min_stock,
    )

    print("Building constraints...")
    milp.build()

    print(f"\nSolving (timeout={timeout_seconds}s)...")
    result = milp.solve(timeout_seconds=timeout_seconds)

    print(f"\nStatus: {result['status_name']}")
    print(f"Solve time: {result['solve_time_seconds']:.2f}s")

    if result["assignments"]:
        total_revenue = sum(a["revenue"] for a in result["assignments"].values())
        total_allocation = sum(a["allocation"] for a in result["assignments"].values())
        print(f"Total revenue: ${total_revenue:,.2f}")
        print(f"Total allocation: {total_allocation:,} units")

    print("=" * 60)

    return result


if __name__ == "__main__":
    print("MILP Solver Module - use solve_pricing_milp() for the main entry point")
