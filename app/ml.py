import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURES = [
    "return_1d", "return_5d", "return_20d", "rsi_14", "macd",
    "macd_signal", "atr_14", "historical_vol_20", "volume_change",
]


def make_training_frame(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    out = df.copy()
    future_return = out["Close"].shift(-horizon) / out["Close"] - 1
    # Explicit research target: UP if future return > 0, otherwise DOWN.
    out["target"] = (future_return > 0).astype(int)
    out.loc[future_return.isna(), "target"] = pd.NA
    return out.dropna(subset=FEATURES + ["target"])


def train_model(train: pd.DataFrame):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")),
    ])
    model.fit(train[FEATURES], train["target"].astype(int))
    return model


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.8):
    cut = int(len(df) * train_fraction)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
