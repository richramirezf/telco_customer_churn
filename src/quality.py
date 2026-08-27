import pandas as pd


def coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert *TotalCharges* to numeric, coercing errors to NaN."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def impute_total_charges_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing *TotalCharges* values with the column median."""
    df = df.copy()
    median = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median)
    return df


def drop_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ``customerID`` column (no predictive value)."""
    df = df.copy()
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    return df


def run_quality_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline in order."""
    df = coerce_total_charges(df)
    df = impute_total_charges_median(df)
    df = drop_customer_id(df)
    nulls = df.isnull().sum().sum()
    print(f"[quality] Pipeline complete – remaining nulls: {nulls}")
    return df
