"""
Genetic Algorithm Solver Module

Implements a GA-based metaheuristic for the Dynamic Pricing & Inventory
Allocation problem. Used as a scalable alternative to the exact MILP,
especially when problem instances grow beyond MILP tractability.

Chromosome: vector of price-level indices per SKU/location/week
Fitness: total expected revenue minus cost
Repair operator: enforces capacity and price-stability constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import random

try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False


class PricingGA:
    """Genetic Algorithm solver for joint pricing and inventory allocation."""

    def __init__(self, sku_list: List[str], location_list: List[str],
                 week_list: List[int],
                 price_grids: Dict[Tuple[str, str], np.ndarray],
                 demand_grids: Dict[Tuple[str, str, int], Dict[float, float]],
                 costs: Dict[str, float],
                 capacities: Dict[str, int],
                 max_price_change: float = 15.0,
                 min_stock: int = 5,
                 population_size: int = 100,
                 n_generations: int = 200,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.15,
                 tournament_size: int = 3,
                 elitism: int = 5,
                 seed: int = 42):
        self.sku_list = sku_list
        self.location_list = location_list
        self.week_list = week_list
        self.price_grids = price_grids
        self.demand_grids = demand_grids
        self.costs = costs
        self.capacities = capacities
        self.max_price_change = max_price_change
        self.min_stock = min_stock
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.seed = seed

        self.price_grid_sizes = {}
        for sku in sku_list:
            for loc in location_list:
                grid = price_grids.get((sku, loc), np.array([10.0]))
                self.price_grid_sizes[(sku, loc)] = len(grid)

        self.solution = None
        self.history = {"best_fitness": [], "avg_fitness": [], "diversity": []}

    def _encode_chromosome(self) -> List[int]:
        """Create a random chromosome: price-level index for each SKU/location/week."""
        chromosome = []
        for sku in self.sku_list:
            for loc in self.location_list:
                n_prices = self.price_grid_sizes.get((sku, loc), 8)
                for week in self.week_list:
                    chromosome.append(random.randint(0, n_prices - 1))
        return chromosome

    def _decode_chromosome(self, chromosome: List[int]) -> Dict:
        """Decode chromosome into pricing and allocation decisions."""
        assignments = {}
        idx = 0
        for sku in self.sku_list:
            for loc in self.location_list:
                grid = self.price_grids.get((sku, loc), np.array([10.0]))
                for week in self.week_list:
                    price_idx = chromosome[idx]
                    price_idx = min(price_idx, len(grid) - 1)
                    price = grid[price_idx]

                    demands = self.demand_grids.get((sku, loc, week), {})
                    demand = demands.get(price, 0)

                    allocation = int(min(demand, self.capacities.get(loc, 1000)))

                    assignments[(sku, loc, week)] = {
                        "price": float(price),
                        "allocation": allocation,
                        "price_idx": price_idx,
                    }
                    idx += 1
        return assignments

    def _fitness(self, chromosome: List[int]) -> float:
        """Evaluate fitness: total revenue minus cost."""
        assignments = self._decode_chromosome(chromosome)

        total_profit = 0
        for (sku, loc, week), decision in assignments.items():
            price = decision["price"]
            allocation = decision["allocation"]
            cost = self.costs.get(sku, 5.0)
            total_profit += (price - cost) * allocation

        total_profit = self._apply_repairs(chromosome, assignments, total_profit)

        return (total_profit,)

    def _apply_repairs(self, chromosome: List[int], assignments: Dict,
                       base_profit: float) -> float:
        """Apply repair operators to enforce constraints and penalize violations."""
        penalty = 0

        loc_totals = {}
        for (sku, loc, week), decision in assignments.items():
            key = (loc, week)
            loc_totals[key] = loc_totals.get(key, 0) + decision["allocation"]

        for (loc, week), total in loc_totals.items():
            capacity = self.capacities.get(loc, 1000)
            if total > capacity:
                overflow_ratio = total / capacity
                penalty += (overflow_ratio - 1) * base_profit * 0.5

        for sku in self.sku_list:
            for loc in self.location_list:
                grid = self.price_grids.get((sku, loc), np.array([10.0]))
                for i in range(1, len(self.week_list)):
                    prev_week = self.week_list[i - 1]
                    curr_week = self.week_list[i]

                    prev_decision = assignments.get((sku, loc, prev_week), {})
                    curr_decision = assignments.get((sku, loc, curr_week), {})

                    if prev_decision and curr_decision:
                        price_diff = abs(curr_decision["price"] - prev_decision["price"])
                        if price_diff > self.max_price_change:
                            penalty += (price_diff - self.max_price_change) * 10

        return base_profit - penalty

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Uniform crossover with crossover probability."""
        if random.random() > self.crossover_prob:
            return parent1[:], parent2[:]

        child1, child2 = parent1[:], parent2[:]
        for i in range(len(child1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def _mutate(self, chromosome: List[int]) -> List[int]:
        """Mutate: randomly change price-level indices."""
        mutated = chromosome[:]
        for i in range(len(mutated)):
            if random.random() < self.mutation_prob:
                sku_idx, loc_idx, week_local = self._index_to_triplet(i)
                sku = self.sku_list[sku_idx]
                loc = self.location_list[loc_idx]
                n_prices = self.price_grid_sizes.get((sku, loc), 8)
                mutated[i] = random.randint(0, n_prices - 1)
        return mutated

    def _index_to_triplet(self, flat_idx: int) -> Tuple[int, int, int]:
        """Convert flat chromosome index to (sku_idx, loc_idx, week_local_idx)."""
        n_weeks = len(self.week_list)
        n_locs = len(self.location_list)

        week_local = flat_idx % n_weeks
        remainder = flat_idx // n_weeks
        loc_idx = remainder % n_locs
        sku_idx = remainder // n_locs

        return sku_idx, loc_idx, week_local

    def _tournament_select(self, population: List, fitnesses: List) -> List[int]:
        """Tournament selection."""
        selected = random.sample(range(len(population)), self.tournament_size)
        best = max(selected, key=lambda i: fitnesses[i][0])
        return population[best][:]

    def solve(self, verbose: bool = True) -> Dict:
        """Run the genetic algorithm."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        start_time = time.time()

        population = [self._encode_chromosome() for _ in range(self.population_size)]
        fitnesses = [self._fitness(ind) for ind in population]

        best_ever = max(range(len(population)), key=lambda i: fitnesses[i][0])
        best_chromosome = population[best_ever][:]
        best_fitness = fitnesses[best_ever][0]

        for gen in range(self.n_generations):
            new_pop = []

            sorted_pop = sorted(range(len(population)),
                                key=lambda i: fitnesses[i][0], reverse=True)
            for i in range(self.elitism):
                new_pop.append(population[sorted_pop[i]][:])

            while len(new_pop) < self.population_size:
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_pop.append(child1)
                if len(new_pop) < self.population_size:
                    new_pop.append(child2)

            population = new_pop[:self.population_size]
            fitnesses = [self._fitness(ind) for ind in population]

            gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i][0])
            gen_best_fitness = fitnesses[gen_best_idx][0]
            gen_avg_fitness = np.mean([f[0] for f in fitnesses])

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_chromosome = population[gen_best_idx][:]

            self.history["best_fitness"].append(best_fitness)
            self.history["avg_fitness"].append(gen_avg_fitness)

            if verbose and (gen + 1) % 50 == 0:
                print(f"  Generation {gen+1}/{self.n_generations}: "
                      f"best={best_fitness:.2f}, avg={gen_avg_fitness:.2f}")

        solve_time = time.time() - start_time

        assignments = self._decode_chromosome(best_chromosome)

        total_revenue = sum(
            d["price"] * d["allocation"]
            for d in assignments.values()
        )
        total_cost = sum(
            self.costs.get(sku, 5.0) * d["allocation"]
            for (sku, loc, week), d in assignments.items()
        )

        self.solution = {
            "status": "OPTIMAL" if best_fitness > 0 else "INFEASIBLE",
            "objective_value": best_fitness,
            "total_revenue": total_revenue,
            "total_cost": total_cost,
            "solve_time_seconds": solve_time,
            "generations_run": self.n_generations,
            "assignments": assignments,
            "history": self.history,
        }

        if verbose:
            print(f"\nGA Solution:")
            print(f"  Best fitness (profit): ${best_fitness:,.2f}")
            print(f"  Total revenue: ${total_revenue:,.2f}")
            print(f"  Solve time: {solve_time:.2f}s")

        return self.solution


def create_ga_from_data(weekly_df: pd.DataFrame, demand_model,
                        capacities: Optional[Dict[str, int]] = None,
                        n_price_levels: int = 8,
                        max_price_change: float = 15.0,
                        min_stock: int = 5,
                        population_size: int = 100,
                        n_generations: int = 200) -> PricingGA:
    """Create a PricingGA instance from processed data and demand model."""
    from milp_solver import create_milp_from_data

    sku_list = sorted(weekly_df["product_id"].unique().tolist())
    location_list = sorted(weekly_df["location"].unique().tolist())
    week_list = sorted(weekly_df["week"].unique().tolist())

    milp = create_milp_from_data(
        weekly_df, demand_model, capacities,
        n_price_levels=n_price_levels,
        max_price_change=max_price_change,
        min_stock=min_stock,
    )

    if capacities is None:
        capacities = {loc: 800 for loc in location_list}

    costs = milp.costs

    ga = PricingGA(
        sku_list=sku_list,
        location_list=location_list,
        week_list=week_list,
        price_grids=milp.price_grids,
        demand_grids=milp.demand_grids,
        costs=costs,
        capacities=capacities,
        max_price_change=max_price_change,
        min_stock=min_stock,
        population_size=population_size,
        n_generations=n_generations,
    )

    return ga


def solve_pricing_ga(weekly_df: pd.DataFrame, demand_model,
                     capacities: Optional[Dict[str, int]] = None,
                     n_price_levels: int = 8,
                     max_price_change: float = 15.0,
                     min_stock: int = 5,
                     population_size: int = 100,
                     n_generations: int = 200) -> Dict:
    """High-level function to solve pricing via GA."""
    print("=" * 60)
    print("GENETIC ALGORITHM SOLVER")
    print("=" * 60)

    ga = create_ga_from_data(
        weekly_df, demand_model, capacities,
        n_price_levels=n_price_levels,
        max_price_change=max_price_change,
        min_stock=min_stock,
        population_size=population_size,
        n_generations=n_generations,
    )

    result = ga.solve(verbose=True)

    print("=" * 60)
    return result


if __name__ == "__main__":
    print("GA Solver Module - use solve_pricing_ga() for the main entry point")
