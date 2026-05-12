"""Preprocessing NASA C-MAPSS FD001 for RUL prediction.

Expected layout:
    datasets/NASA/
        preprocessing.py
        raw/CMAPSSData/
            train_FD001.txt
            test_FD001.txt
            RUL_FD001.txt
        processed/

It produces outputs in datasets/NASA/processed:
    X_train.npy, Y_train.npy, X_test.npy, Y_test.npy
    X_train.csv, X_test.csv
    Y_train.csv, Y_test.csv
    train_windows.csv, test_windows.csv
"""

import argparse
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency: {exc.name}. "
        "Install numpy, pandas and scikit-learn before running this script."
    ) from exc


WINDOW_SIZE = 30
RUL_CAP = 125
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw" / "CMAPSSData"
PROCESSED_DIR = SCRIPT_DIR / "processed"

BASE_COLUMNS = [
    "Engine_ID",
    "Cycle",
    "Op_Setting_1",
    "Op_Setting_2",
    "Op_Setting_3",
]
SENSOR_COLUMNS = [f"Sensor_{idx}" for idx in range(1, 22)]
COLUMNS = BASE_COLUMNS + SENSOR_COLUMNS

DROP_COLUMNS = [
    "Op_Setting_1",
    "Op_Setting_2",
    "Op_Setting_3",
    "Sensor_1",
    "Sensor_5",
    "Sensor_16",
    "Sensor_18",
    "Sensor_19",
]


def load_cmapss_file(path: Path) -> pd.DataFrame:
    """Load a C-MAPSS data file and assign the standard FD001 column names."""
    data = pd.read_csv(path, sep=r"\s+", header=None)
    if data.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name} has {data.shape[1]} columns, expected {len(COLUMNS)}."
        )

    data.columns = COLUMNS
    return data


def add_train_rul(data: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Compute capped, decreasing RUL labels for the training set."""
    data = data.copy()
    max_cycle = data.groupby("Engine_ID")["Cycle"].transform("max")
    data["RUL"] = (max_cycle - data["Cycle"]).clip(upper=cap)
    return data


def add_test_rul(
    data: pd.DataFrame, rul_path: Path, cap: int = RUL_CAP
) -> pd.DataFrame:
    """Reconstruct capped RUL labels for all test cycles from final RUL values."""
    data = data.copy()
    final_rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0]
    engine_ids = sorted(data["Engine_ID"].unique())

    if len(final_rul) != len(engine_ids):
        raise ValueError(
            f"{rul_path.name} has {len(final_rul)} RUL values, "
            f"but the test set contains {len(engine_ids)} engines."
        )

    final_rul_by_engine = dict(zip(engine_ids, final_rul.astype(float)))
    max_cycle_by_engine = data.groupby("Engine_ID")["Cycle"].transform("max")
    data["Final_RUL"] = data["Engine_ID"].map(final_rul_by_engine)
    data["RUL"] = (data["Final_RUL"] + max_cycle_by_engine - data["Cycle"]).clip(
        upper=cap
    )
    data = data.drop(columns=["Final_RUL"])
    return data


def drop_unused_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Remove operating settings and constant FD001 sensors."""
    return data.drop(columns=DROP_COLUMNS)


def normalize_features(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fit MinMaxScaler on train sensors only and transform train/test features."""
    train_data = train_data.copy()
    test_data = test_data.copy()
    feature_columns = [
        col for col in train_data.columns if col not in ("Engine_ID", "Cycle", "RUL")
    ]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])

    return train_data, test_data, feature_columns


def extract_sliding_windows(
    data: pd.DataFrame, feature_columns: list[str], window_size: int = WINDOW_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Create temporal windows per engine; target is RUL at the last window cycle."""
    x_windows: list[np.ndarray] = []
    y_values: list[float] = []

    for _, engine_data in data.groupby("Engine_ID", sort=True):
        engine_data = engine_data.sort_values("Cycle")
        features = engine_data[feature_columns].to_numpy(dtype=np.float32)
        rul = engine_data["RUL"].to_numpy(dtype=np.float32)

        if len(engine_data) < window_size:
            continue

        for start_idx in range(0, len(engine_data) - window_size + 1):
            end_idx = start_idx + window_size
            x_windows.append(features[start_idx:end_idx])
            y_values.append(rul[end_idx - 1])

    if not x_windows:
        raise ValueError(
            f"No windows were generated. Check that every engine has at least "
            f"{window_size} cycles."
        )

    return np.asarray(x_windows, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def flattened_window_columns(feature_columns: list[str], window_size: int) -> list[str]:
    """Build stable column names for flattened temporal windows."""
    return [
        f"t{time_idx:02d}_{feature}"
        for time_idx in range(window_size)
        for feature in feature_columns
    ]


def save_outputs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_columns: list[str],
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Save binary NumPy matrices and flattened CSV datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", x_train)
    np.save(output_dir / "Y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", x_test)
    np.save(output_dir / "Y_test.npy", y_test)

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    pd.DataFrame(x_train_flat).to_csv(output_dir / "X_train.csv", index=False, header=False)
    pd.DataFrame(x_test_flat).to_csv(output_dir / "X_test.csv", index=False, header=False)
    pd.DataFrame({"RUL": y_train}).to_csv(output_dir / "Y_train.csv", index=False)
    pd.DataFrame({"RUL": y_test}).to_csv(output_dir / "Y_test.csv", index=False)

    window_columns = flattened_window_columns(feature_columns, x_train.shape[1])
    train_windows = pd.DataFrame(x_train_flat, columns=window_columns)
    train_windows["RUL"] = y_train
    test_windows = pd.DataFrame(x_test_flat, columns=window_columns)
    test_windows["RUL"] = y_test

    train_windows.to_csv(output_dir / "train_windows.csv", index=False)
    test_windows.to_csv(output_dir / "test_windows.csv", index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess NASA C-MAPSS FD001.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory containing train_FD001.txt, test_FD001.txt and RUL_FD001.txt. Default: {RAW_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Directory where processed arrays and CSV files are written. Default: {PROCESSED_DIR}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_path = raw_dir / "train_FD001.txt"
    test_path = raw_dir / "test_FD001.txt"
    rul_path = raw_dir / "RUL_FD001.txt"

    for path in (train_path, test_path, rul_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    train_data = load_cmapss_file(train_path)
    test_data = load_cmapss_file(test_path)

    train_data = add_train_rul(train_data)
    test_data = add_test_rul(test_data, rul_path)

    train_data = drop_unused_columns(train_data)
    test_data = drop_unused_columns(test_data)

    train_data, test_data, feature_columns = normalize_features(train_data, test_data)

    x_train, y_train = extract_sliding_windows(train_data, feature_columns)
    x_test, y_test = extract_sliding_windows(test_data, feature_columns)

    save_outputs(x_train, y_train, x_test, y_test, feature_columns, output_dir)

    print("Preprocessing completed.")
    print(f"Raw input dir: {raw_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Features used: {len(feature_columns)} -> {feature_columns}")
    print(f"X_train shape: {x_train.shape}")
    print(f"Y_train shape: {y_train.shape}")
    print(f"X_test shape:  {x_test.shape}")
    print(f"Y_test shape:  {y_test.shape}")
    print(f"X_train.csv shape: ({x_train.shape[0]}, {x_train.shape[1] * x_train.shape[2]})")
    print(f"X_test.csv shape:  ({x_test.shape[0]}, {x_test.shape[1] * x_test.shape[2]})")


if __name__ == "__main__":
    main()
