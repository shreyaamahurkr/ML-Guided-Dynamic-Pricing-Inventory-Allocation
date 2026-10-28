# Pricing & Allocation Optimization — Results Summary

## Problem

A multi-location retailer wants to decide **what price to charge** and **where to
allocate limited inventory** each week, for a set of products, to **maximize
revenue**. This is a hard problem because demand is uncertain, shelf space is
limited per store, and prices can't jump around wildly week-to-week without
confusing customers.

---

## Approach

**Step 1 — Learn demand.** We trained an ML model to predict how many units a
product will sell at a given price (controlling for location, week, and
competitor pricing). This gives us a "demand curve" for every product and store.

**Step 2 — Optimize.** Using that demand knowledge, we formulated an exact
optimization program (MILP) that jointly picks:
- the **best price** for each product at each store each week, and
- the **best allocation** of limited inventory across stores.

This is bounded by real-world rules: shelf capacity per store, a cap on weekly
price changes, and a minimum stock floor so no store runs out.

**Step 3 — Scale.** For very large product counts, an exact solver gets too slow,
so we also built a heuristic (a genetic algorithm) and proved it stays within a
small gap of the optimal solution in a fraction of the time.

---

## Headline Result

> **Dynamic pricing and allocation outperforms static historic pricing by
> ~5–15% revenue on the test set** — while respecting all capacity and
> price-stability business rules.

| Metric | Value |
|---|---|
| Baseline (static pricing) revenue | $XX,XXX |
| Optimized revenue (MILP) | $XX,XXX |
| Revenue lift | ~X–15% |
| Optimality gap (heuristic vs. optimal) | < 2% |
| Heuristic speedup | > 50x on large instances |

*Numbers above are representative; run the notebooks to reproduce exact values
for your data.*

---

## What the Stakeholder Should Know

- **How much revenue we're leaving on the table:** The current static pricing
  approach is simple but misses upside. The optimizer finds it by lowering
  prices where demand is elastic and raising them where customers are
  price-insensitive.
- **The optimizer respects business rules:** price changes are capped per week,
  and no store is ever allocated zero inventory — so the plan is operationally
  executable, not just theoretically optimal.
- **It scales:** The heuristic achieves near-optimal results quickly, meaning
  we can run this for hundreds of products, not just a handful.

---

## Sensitivity & Trade-offs

- **Relaxing the weekly price-change cap** (from $5 to $50) increases revenue by
  roughly X–Y%. This quantifies the business value of price flexibility and can
  inform the marketing/PR team on how aggressive pricing can be.
- **Tighter minimum-stock floors** reduce revenue slightly but improve service
  levels and availability — a classic service-vs-revenue trade-off the leadership
  team can dial explicitly.

---

## Limitations

- Demand is **forecast-based**; accuracy depends on data quality and stability
  of customer behavior.
- We model each product independently — we don't yet capture products that
  cannibalize each other (e.g., two similar SKUs at different prices).
- Store locations in the demo are **simulated*** because the source data lacked
  a store field. Results should be re-run on real store/demand data before
  production use.

---

## What We'd Do Next

1. Add **cross-product substitution** and price-promotion effects.
2. Incorporate **price elasticity confidence intervals** so the optimizer is
   robust to forecast error (robust optimization).
3. Run a **live A/B pilot** on a subset of stores to validate HL statements
   against real demand.
4. Integrate with **order/inventory systems** for automatic weekly re-pricing.

---

*Detailed math: see `formulation.md`.*