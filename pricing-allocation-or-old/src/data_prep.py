"""
Data Preparation Module

Loads the Kaggle Retail Price Optimization dataset, cleans it,
and aggregates to weekly SKU x location granularity with simulated locations.

Assumptions (documented for stakeholder review):
- Since the original dataset lacks location info, we synthetically split
  transactions across 3 locations using category-based assignment.
- Weekly aggregation uses ISO week numbering.
- Price changes beyond 3 standard deviations from the mean are treated as outliers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

NUM_LOCATIONS = 3
LOCATION_CAPACITY = {
    "Location_A": 800,
    "Location_B": 600,
    "Location_C": 500,
}
LOCATION_ASSIGNMENT_SEED = 42
MIN_WEEKLY_TRANSACTIONS = 5
TOP_N_SKUS = 18


def generate_synthetic_dataset(n_rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic retail pricing dataset mimicking the Kaggle structure.

    Uses a negative binomial demand model with realistic price elasticity
    so that the log-log regression recovers meaningful elasticity coefficients.
    """
    rng = np.random.default_rng(seed)

    categories = ["Electronics", "Clothing", "Groceries", "Home", "Sports"]
    base_prices = {
        "Electronics": 150.0,
        "Clothing": 45.0,
        "Groceries": 12.0,
        "Home": 75.0,
        "Sports": 65.0,
    }

    n_skus = 20
    sku_ids = [f"SKU_{i+1:03d}" for i in range(n_skus)]
    sku_categories = rng.choice(categories, n_skus)

    sku_meta = {}
    for sku, cat in zip(sku_ids, sku_categories):
        base_price = base_prices[cat] * (0.7 + 0.6 * rng.random())
        size_param = rng.uniform(40, 200)           # baseline demand magnitude
        elasticity = -rng.uniform(0.8, 2.5)          # demand elasticity [-2.5, -0.8]
        comp_coef = rng.uniform(0.2, 0.5)            # competitor-price sensitivity
        cost = base_price * rng.uniform(0.4, 0.6)
        sku_meta[sku] = {
            "base_price": base_price,
            "size": size_param,
            "elasticity": elasticity,
            "comp_coef": comp_coef,
            "cost": cost,
            "category": cat,
        }

    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    chosen_dates = pd.to_datetime(rng.choice(dates, n_rows))

    # Overshoot so trimming to weekly stays rich
    rows = []
    for i in range(n_rows):
        sku_idx = rng.integers(0, n_skus)
        sku = sku_ids[sku_idx]
        meta = sku_meta[sku]
        date = chosen_dates[i]

        # Weekly price around a central base price with some drift
        price = meta["base_price"] * rng.uniform(0.80, 1.25)
        competitor_price = meta["base_price"] * rng.uniform(0.75, 1.30)

        # Seasonality factor peaks around weeks 30-42
        week = date.isocalendar().week
        seasonal = 1.0 + 0.5 * np.sin(2 * np.pi * (week - 12) / 52)

        # Log-log demand model:
        #   log(qty) = log(size) + beta_p * ln(price/base) + beta_c * ln(comp/base) + seasonal + noise
        log_qty = (
            np.log(meta["size"]) +
            meta["elasticity"] * np.log(price / meta["base_price"]) +
            meta["comp_coef"] * np.log(competitor_price / meta["base_price"]) +
            np.log(seasonal) +
            rng.normal(0, 0.15)
        )
        quantity = max(1, int(np.exp(log_qty)))

        rows.append({
            "product_id": sku,
            "category": meta["category"],
            "date": date,
            "price": round(float(price), 2),
            "competitor_price": round(float(competitor_price), 2),
            "cost": round(float(meta["cost"]), 2),
            "quantity_sold": quantity,
        })

    return pd.DataFrame(rows)


def load_raw_data(force_synthetic: bool = True) -> pd.DataFrame:
    """Load data from raw CSV or generate synthetic if not available."""
    csv_path = RAW_DIR / "retail_price.csv"
    if csv_path.exists() and not force_synthetic:
        df = pd.read_csv(csv_path, parse_dates=["date"] if "date" in pd.read_csv(csv_path, nrows=0).columns else None)
        return df
    else:
        print("Generating synthetic dataset...")
        df = generate_synthetic_dataset()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved synthetic dataset to {csv_path}")
        return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean nulls, outliers, and basic transformations."""
    df = df.copy()

    initial_rows = len(df)

    df = df.dropna(subset=["price", "quantity_sold", "product_id"])

    for col in ["price", "competitor_price", "cost", "quantity_sold"]:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            lower = max(0, mean - 3 * std)
            upper = mean + 3 * std
            before = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            removed = before - len(df)
            if removed > 0:
                print(f"  Removed {removed} rows with outlier {col} values")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    df["price"] = df["price"].astype(float)
    df["quantity_sold"] = df["quantity_sold"].astype(int)

    if "competitor_price" in df.columns:
        df["competitor_price"] = df["competitor_price"].fillna(df["price"])
    if "cost" in df.columns:
        df["cost"] = df["cost"].fillna(df["price"] * 0.5)

    print(f"Cleaned: {initial_rows} -> {len(df)} rows ({initial_rows - len(df)} removed)")
    return df


def assign_locations(df: pd.DataFrame, num_locations: int = NUM_LOCATIONS,
                     seed: int = LOCATION_ASSIGNMENT_SEED) -> pd.DataFrame:
    """Synthetically assign transactions to locations based on category.

    Assumption: Different categories are more prevalent at different locations,
    simulating real store specialization. This is a documented assumption for
    stakeholder review.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    if "category" in df.columns:
        categories = df["category"].unique()
        location_names = [f"Location_{chr(65+i)}" for i in range(num_locations)]

        cat_location_probs = {}
        for cat in categories:
            probs = rng.dirichlet(np.ones(num_locations))
            cat_location_probs[cat] = probs

        assigned_locations = []
        for cat in df["category"]:
            probs = cat_location_probs[cat]
            if len(probs) == len(location_names):
                assigned_locations.append(
                    rng.choice(location_names, p=probs)
                )
            else:
                assigned_locations.append(location_names[0])

        df["location"] = assigned_locations
    else:
        location_names = [f"Location_{chr(65+i)}" for i in range(num_locations)]
        df["location"] = rng.choice(location_names, len(df))

    print(f"Assigned transactions to {df['location'].nunique()} locations")
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to weekly SKU x location granularity."""
    df = df.copy()

    if "date" in df.columns:
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["year"] = df["date"].dt.isocalendar().year.astype(int)
    else:
        raise ValueError("DataFrame must have a 'date' column")

    agg_dict = {
        "price": "mean",
        "quantity_sold": "sum",
    }
    if "competitor_price" in df.columns:
        agg_dict["competitor_price"] = "mean"
    if "cost" in df.columns:
        agg_dict["cost"] = "mean"
    if "category" in df.columns:
        agg_dict["category"] = "first"

    weekly = df.groupby(["product_id", "location", "year", "week"]).agg(agg_dict).reset_index()

    weekly = weekly.sort_values(["product_id", "location", "year", "week"]).reset_index(drop=True)

    print(f"Aggregated to {len(weekly)} weekly SKU x location records")
    return weekly


def select_top_skus(df: pd.DataFrame, n: int = TOP_N_SKUS) -> pd.DataFrame:
    """Select top N SKUs by total transaction volume."""
    sku_volume = df.groupby("product_id")["quantity_sold"].sum().sort_values(ascending=False)
    top_skus = sku_volume.head(n).index.tolist()
    filtered = df[df["product_id"].isin(top_skus)].copy()
    print(f"Selected top {len(top_skus)} SKUs: {top_skus}")
    return filtered


def compute_price_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-log price elasticity per SKU as a sanity-check baseline."""
    from sklearn.linear_model import LinearRegression

    results = []
    for sku in df["product_id"].unique():
        sku_data = df[df["product_id"] == sku].copy()
        if len(sku_data) < 10:
            continue

        sku_data = sku_data[(sku_data["price"] > 0) & (sku_data["quantity_sold"] > 0)]
        if len(sku_data) < 10:
            continue

        log_price = np.log(sku_data["price"].values).reshape(-1, 1)
        log_qty = np.log(sku_data["quantity_sold"].values)

        model = LinearRegression()
        model.fit(log_price, log_qty)

        results.append({
            "product_id": sku,
            "elasticity": model.coef_[0],
            "r_squared": model.score(log_price, log_qty),
            "mean_price": sku_data["price"].mean(),
            "mean_quantity": sku_data["quantity_sold"].mean(),
        })

    elasticity_df = pd.DataFrame(results)
    print(f"Computed elasticity for {len(elasticity_df)} SKUs")
    return elasticity_df


def prepare_dataset(force_synthetic: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full data preparation pipeline.

    Returns:
        weekly_df: Cleaned weekly SKU x location data
        elasticity_df: Price elasticity per SKU
    """
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)

    print("\n1. Loading raw data...")
    raw = load_raw_data(force_synthetic=force_synthetic)

    print("\n2. Cleaning data...")
    cleaned = clean_data(raw)

    print("\n3. Assigning locations...")
    located = assign_locations(cleaned)

    print("\n4. Selecting top SKUs...")
    top_skus = select_top_skus(located)

    print("\n5. Aggregating to weekly granularity...")
    weekly = aggregate_weekly(top_skus)

    print("\n6. Computing price elasticity...")
    elasticity = compute_price_elasticity(weekly)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED_DIR / "weekly_sku_location.csv", index=False)
    elasticity.to_csv(PROCESSED_DIR / "price_elasticity.csv", index=False)
    print(f"\nSaved processed data to {PROCESSED_DIR}")

    print("=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    return weekly, elasticity


def load_processed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously processed data."""
    weekly = pd.read_csv(PROCESSED_DIR / "weekly_sku_location.csv")
    elasticity = pd.read_csv(PROCESSED_DIR / "price_elasticity.csv")
    return weekly, elasticity


if __name__ == "__main__":
    weekly_df, elasticity_df = prepare_dataset()
    print("\nWeekly data shape:", weekly_df.shape)
    print("Elasticity data shape:", elasticity_df.shape)
    print("\nElasticity summary:")
    print(elasticity_df.describe())
