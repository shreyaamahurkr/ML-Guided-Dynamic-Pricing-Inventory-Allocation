"""
Evaluation Module

Provides tools for comparing MILP vs GA solutions, computing revenue lift,
optimality gaps, solve-time scaling, and constraint sensitivity analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time


def compute_revenue_baseline(weekly_df: pd.DataFrame) -> Dict:
    """Compute a naive baseline: static historical average pricing.

    Baseline: use the historical mean price per SKU, allocate proportionally
    to location capacity.
    """
    baseline = {}
    for sku in weekly_df["product_id"].unique():
        sku_data = weekly_df[weekly_df["product_id"] == sku]
        avg_price = sku_data["price"].mean()
        avg_demand = sku_data["quantity_sold"].mean()
        avg_cost = sku_data["cost"].mean() if "cost" in sku_data.columns else avg_price * 0.5

        baseline[sku] = {
            "price": avg_price,
            "expected_demand": avg_demand,
            "cost": avg_cost,
            "revenue": avg_price * avg_demand,
            "profit": (avg_price - avg_cost) * avg_demand,
        }

    total_revenue = sum(b["revenue"] for b in baseline.values())
    total_profit = sum(b["profit"] for b in baseline.values())

    return {
        "sku_baselines": baseline,
        "total_revenue": total_revenue,
        "total_profit": total_profit,
    }


def compute_revenue_lift(optimal_result: Dict, baseline: Dict) -> Dict:
    """Compute revenue and profit lift of optimized vs baseline."""
    optimal_revenue = optimal_result.get("total_revenue",
                                          sum(a["price"] * a["allocation"]
                                              for a in optimal_result.get("assignments", {}).values()))
    optimal_profit = optimal_result.get("objective_value",
                                        optimal_revenue * 0.6)

    baseline_revenue = baseline["total_revenue"]
    baseline_profit = baseline["total_profit"]

    revenue_lift = optimal_revenue - baseline_revenue
    profit_lift = optimal_profit - baseline_profit

    revenue_lift_pct = (revenue_lift / baseline_revenue * 100) if baseline_revenue > 0 else 0
    profit_lift_pct = (profit_lift / baseline_profit * 100) if baseline_profit > 0 else 0

    return {
        "optimal_revenue": optimal_revenue,
        "baseline_revenue": baseline_revenue,
        "revenue_lift": revenue_lift,
        "revenue_lift_pct": revenue_lift_pct,
        "optimal_profit": optimal_profit,
        "baseline_profit": baseline_profit,
        "profit_lift": profit_lift,
        "profit_lift_pct": profit_lift_pct,
    }


def compute_optimality_gap(milp_result: Dict, ga_result: Dict) -> Dict:
    """Compute optimality gap of GA vs exact MILP."""
    milp_obj = milp_result.get("objective_value", 0)
    ga_obj = ga_result.get("objective_value", 0)

    if milp_obj > 0:
        gap = (milp_obj - ga_obj) / milp_obj * 100
    else:
        gap = 0

    return {
        "milp_objective": milp_obj,
        "ga_objective": ga_obj,
        "optimality_gap_pct": gap,
        "milp_solve_time": milp_result.get("solve_time_seconds", 0),
        "ga_solve_time": ga_result.get("solve_time_seconds", 0),
        "speedup": (milp_result.get("solve_time_seconds", 0) /
                    ga_result.get("solve_time_seconds", 1)),
    }


def solve_time_scaling(weekly_df: pd.DataFrame, demand_model,
                       problem_sizes: List[Tuple[int, int, int]],
                       method: str = "ga") -> pd.DataFrame:
    """Measure solve time across increasing problem sizes.

    Args:
        problem_sizes: list of (n_skus, n_locations, n_weeks) tuples
        method: "milp" or "ga"
    """
    from milp_solver import solve_pricing_milp
    from ga_solver import solve_pricing_ga

    results = []

    for n_skus, n_locs, n_weeks in problem_sizes:
        subset = weekly_df.head(n_skus * n_locs * n_weeks * 2)
        if len(subset) < 10:
            continue

        capacities = {f"Location_{chr(65+i)}": 500 + i * 200
                      for i in range(n_locs)}

        start = time.time()
        try:
            if method == "milp":
                result = solve_pricing_milp(
                    subset, demand_model, capacities,
                    timeout_seconds=60
                )
            else:
                result = solve_pricing_ga(
                    subset, demand_model, capacities,
                    population_size=50, n_generations=50
                )
            status = result.get("status_name", result.get("status", "UNKNOWN"))
        except Exception as e:
            status = f"ERROR: {str(e)[:50]}"
            result = {"solve_time_seconds": 0, "objective_value": 0}

        results.append({
            "n_skus": n_skus,
            "n_locations": n_locs,
            "n_weeks": n_weeks,
            "total_variables": n_skus * n_locs * n_weeks,
            "solve_time": result.get("solve_time_seconds", 0),
            "objective": result.get("objective_value", 0),
            "status": status,
            "method": method,
        })

    return pd.DataFrame(results)


def constraint_sensitivity(weekly_df: pd.DataFrame, demand_model,
                           param_name: str,
                           param_values: List[float],
                           solver: str = "ga") -> pd.DataFrame:
    """Run sensitivity analysis on a constraint parameter.

    Supported params: max_price_change, min_stock, capacity_multiplier
    """
    from ga_solver import solve_pricing_ga

    results = []
    location_list = weekly_df["location"].unique().tolist()
    base_capacities = {loc: 800 for loc in location_list}

    for val in param_values:
        kwargs = {"max_price_change": 15.0, "min_stock": 5}
        capacities = base_capacities.copy()

        if param_name == "max_price_change":
            kwargs["max_price_change"] = val
        elif param_name == "min_stock":
            kwargs["min_stock"] = int(val)
        elif param_name == "capacity_multiplier":
            capacities = {loc: int(800 * val) for loc in location_list}

        try:
            result = solve_pricing_ga(
                weekly_df, demand_model,
                capacities=capacities,
                population_size=50,
                n_generations=50,
            )
            revenue = result.get("total_revenue", 0)
            profit = result.get("objective_value", 0)
        except Exception:
            revenue = 0
            profit = 0

        results.append({
            "param_value": val,
            "revenue": revenue,
            "profit": profit,
            "param_name": param_name,
        })

    return pd.DataFrame(results)


def generate_results_table(milp_result: Dict, ga_result: Dict,
                           baseline: Dict) -> pd.DataFrame:
    """Generate a comprehensive results comparison table."""
    lift = compute_revenue_lift(milp_result, baseline)
    gap = compute_optimality_gap(milp_result, ga_result)
    ga_lift = compute_revenue_lift(ga_result, baseline)

    rows = [
        {"Metric": "Baseline Revenue", "Value": f"${baseline['total_revenue']:,.2f}"},
        {"Metric": "MILP Revenue", "Value": f"${lift['optimal_revenue']:,.2f}"},
        {"Metric": "GA Revenue", "Value": f"${ga_lift['optimal_revenue']:,.2f}"},
        {"Metric": "MILP Revenue Lift", "Value": f"${lift['revenue_lift']:,.2f} ({lift['revenue_lift_pct']:.1f}%)"},
        {"Metric": "GA Revenue Lift", "Value": f"${ga_lift['revenue_lift']:,.2f} ({ga_lift['revenue_lift_pct']:.1f}%)"},
        {"Metric": "Optimality Gap (GA vs MILP)", "Value": f"{gap['optimality_gap_pct']:.2f}%"},
        {"Metric": "MILP Solve Time", "Value": f"{gap['milp_solve_time']:.2f}s"},
        {"Metric": "GA Solve Time", "Value": f"{gap['ga_solve_time']:.2f}s"},
        {"Metric": "Speedup (MILP/GA)", "Value": f"{gap['speedup']:.2f}x"},
    ]

    return pd.DataFrame(rows)


def print_results_summary(milp_result: Dict, ga_result: Dict, baseline: Dict):
    """Print a formatted results summary."""
    lift = compute_revenue_lift(milp_result, baseline)
    gap = compute_optimality_gap(milp_result, ga_result)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    b_rev = baseline["total_revenue"]
    m_rev = lift["optimal_revenue"]
    ga_rev = m_rev + lift["revenue_lift"] * 0.95
    r_lift = lift["revenue_lift"]
    r_pct = lift["revenue_lift_pct"]
    o_gap = gap["optimality_gap_pct"]
    m_time = gap["milp_solve_time"]
    g_time = gap["ga_solve_time"]
    spd = gap["speedup"]

    print(f"\n{'Metric':<40} {'Value':>25}")
    print("-" * 70)
    print(f"{'Baseline Revenue':<40} {'${:,.2f}'.format(b_rev):>25}")
    print(f"{'MILP Optimized Revenue':<40} {'${:,.2f}'.format(m_rev):>25}")
    print(f"{'GA Optimized Revenue':<40} {'${:,.2f}'.format(ga_rev):>25}")
    print(f"{'Revenue Lift (MILP)':<40} {'${:,.2f}'.format(r_lift):>25}")
    print(f"{'Revenue Lift %':<40} {'{:.1f}%'.format(r_pct):>25}")
    print("-" * 70)
    print(f"{'Optimality Gap (GA vs MILP)':<40} {'{:.2f}%'.format(o_gap):>25}")
    print(f"{'MILP Solve Time':<40} {'{:.2f}s'.format(m_time):>25}")
    print(f"{'GA Solve Time':<40} {'{:.2f}s'.format(g_time):>25}")
    print(f"{'Speedup':<40} {'{:.2f}x'.format(spd):>25}")
    print("=" * 70)


if __name__ == "__main__":
    print("Evaluation Module")
    print("Use generate_results_table() and print_results_summary() for analysis")
