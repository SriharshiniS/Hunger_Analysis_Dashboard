import pandas as pd

def load_data(path):
    df = pd.read_csv(path, low_memory=False)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Remove junk columns
    df = df.loc[:, ~df.columns.str.contains('unnamed', case=False)]

    return df


def prepare_features(df, target="hunger_level"):

    if target not in df.columns:
        raise ValueError(f"{target} not found!")

    # ✅ Separate target FIRST
    y = df[target]

    # ✅ Keep only numeric features
    X = df.select_dtypes(include=['number'])

    # ✅ Drop rows where target is missing
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    return X, y