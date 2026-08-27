import pandas as pd
from pathlib import Path


def load_telco_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the Telco Customer Churn CSV file.

    Parameters
    ----------
    csv_path : str or Path, optional
        Absolute or relative path to the CSV. When *None* the loader looks
        for ``data/Telco-Customer-Churn.csv`` inside the project root.

    Returns
    -------
    pd.DataFrame
        Raw dataframe ready for cleaning.
    """
    if csv_path is None:
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[data_loader] Loaded {len(df):,} rows from {csv_path.name}")
    return df
