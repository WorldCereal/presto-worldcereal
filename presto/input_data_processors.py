import logging
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from presto.dataops import NODATAVALUE

logger = logging.getLogger("__main__")

STATIC_FEATURES = ["DEM-alt-20m", "DEM-slo-20m", "lat", "lon"]
REQUIRED_COLUMNS = ["sample_id", "timestamp"] + STATIC_FEATURES

BAND_MAPPINGS = {
    "10m": ["OPTICAL-B02", "OPTICAL-B03", "OPTICAL-B04", "OPTICAL-B08"],
    "20m": [
        "SAR-VH",
        "SAR-VV",
        "OPTICAL-B05",
        "OPTICAL-B06",
        "OPTICAL-B07",
        "OPTICAL-B11",
        "OPTICAL-B12",
        "OPTICAL-B8A",
    ],
    "100m": ["METEO-precipitation_flux", "METEO-temperature_mean"],
}

FEATURE_COLUMNS = BAND_MAPPINGS["10m"] + BAND_MAPPINGS["20m"] + BAND_MAPPINGS["100m"]

COLUMN_RENAMES: Dict[str, str] = {
    "S1-SIGMA0-VV": "SAR-VV",
    "S1-SIGMA0-VH": "SAR-VH",
    "S2-L2A-B02": "OPTICAL-B02",
    "S2-L2A-B03": "OPTICAL-B03",
    "S2-L2A-B04": "OPTICAL-B04",
    "S2-L2A-B05": "OPTICAL-B05",
    "S2-L2A-B06": "OPTICAL-B06",
    "S2-L2A-B07": "OPTICAL-B07",
    "S2-L2A-B08": "OPTICAL-B08",
    "S2-L2A-B11": "OPTICAL-B11",
    "S2-L2A-B12": "OPTICAL-B12",
    "AGERA5-precipitation-flux": "METEO-precipitation_flux",
    "AGERA5-temperature-mean": "METEO-temperature_mean",
    # since the openEO output has the attribute "valid_time",
    # # we need the following line for compatibility with earlier datasets
    "valid_date": "valid_time",
}

EXPECTED_DISTANCES = {"month": 31, "dekad": 10}


class DataFrameValidator:
    @staticmethod
    def validate_required_columns(df_long: pd.DataFrame) -> None:
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df_long.columns]
        if missing_columns:
            raise AttributeError(
                f"DataFrame must contain the following columns: {missing_columns}"
            )

    @staticmethod
    def validate_timestamps(df_long: pd.DataFrame, freq: str = "month") -> None:
        if freq == "month" and not df_long["timestamp"].dt.is_month_start.all():
            raise ValueError("All monthly timestamps must be at month start")

    @staticmethod
    def check_faulty_samples(df_wide: pd.DataFrame, min_edge_buffer: int) -> pd.DataFrame:
        min_center_point = np.maximum(
            df_wide["available_timesteps"] // 2,
            df_wide["valid_position"] + min_edge_buffer - df_wide["available_timesteps"] // 2,
        )
        max_center_point = np.minimum(
            df_wide["available_timesteps"] - df_wide["available_timesteps"] // 2,
            df_wide["valid_position"] - min_edge_buffer + df_wide["available_timesteps"] // 2,
        )

        faulty_samples = min_center_point > max_center_point
        if faulty_samples.sum() > 0:
            logger.warning(f"Dropping {faulty_samples.sum()} faulty sample(s).")

        return df_wide[~faulty_samples]

    @staticmethod
    def check_min_timesteps(df_wide: pd.DataFrame, required_min_timesteps: int) -> pd.DataFrame:
        samples_with_too_few_ts = df_wide["available_timesteps"] < required_min_timesteps
        if samples_with_too_few_ts.sum() > 0:
            logger.warning(
                f"Dropping {samples_with_too_few_ts.sum()} sample(s) with \
number of available timesteps less than {required_min_timesteps}."
            )
        df_wide = df_wide[~samples_with_too_few_ts]
        if len(df_wide) == 0:
            raise ValueError(
                f"Left with an empty DataFrame! \
All samples have fewer timesteps than required ({required_min_timesteps})."
            )
        else:
            return df_wide

    @staticmethod
    def check_median_distance(df_long: pd.DataFrame, ts_freq: str) -> pd.DataFrame:
        # compute median distance between observations
        # and use it as an approximation for frequency

        # Pre-calculate daily differences without repeated .unique()
        ts_subset_df = (
            df_long[["sample_id", "timestamp"]]
            .drop_duplicates()
            .sort_values(by=["sample_id", "timestamp"])
        )
        ts_subset_df["timestamp_diff_days"] = (
            ts_subset_df.groupby("sample_id")["timestamp"].diff().dt.days.abs()
        )
        median_distance = (
            ts_subset_df.groupby("sample_id")["timestamp_diff_days"].median().fillna(0).astype(int)
        )

        if ts_freq == "month":
            samples_with_mismatching_distance = median_distance[
                median_distance != EXPECTED_DISTANCES["month"]
            ]
        elif ts_freq == "dekad":
            samples_with_mismatching_distance = median_distance[
                median_distance != EXPECTED_DISTANCES["dekad"]
            ]
        else:
            raise ValueError(f"ts_freq {ts_freq} not supported.")

        if len(samples_with_mismatching_distance) > 0:
            logger.warning(
                f"Found {len(samples_with_mismatching_distance)} samples with median distance \
between observations not corresponding to {ts_freq}. \
Removing them from the dataset."
            )
            df_long = df_long[~df_long["sample_id"].isin(samples_with_mismatching_distance.index)]
            if len(df_long) == 0:
                raise ValueError(
                    f"Left with an empty DataFrame! All samples have median distance between \
observations not corresponding to {ts_freq}."
                )
        else:
            logger.info(
                f"Expected observations frequency: {ts_freq}; \
Median observed distance between observations: {median_distance.unique()} days"
            )

        return df_long


class TimeSeriesProcessor:
    @staticmethod
    def calculate_valid_position(df_long: pd.DataFrame) -> pd.DataFrame:
        df_long["valid_time_ts_diff_days"] = (
            df_long["valid_time"] - df_long["timestamp"]
        ).dt.days.abs()
        valid_position = (
            df_long.set_index("timestamp_ind")
            .groupby("sample_id")["valid_time_ts_diff_days"]
            .idxmin()
        )
        df_long["valid_position"] = df_long["sample_id"].map(valid_position)
        return df_long

    @staticmethod
    def fill_missing_dates(
        df_long: pd.DataFrame, ts_freq: str, index_columns: List[str]
    ) -> pd.DataFrame:
        def get_expected_dates(start_date, end_date, ts_freq):
            if ts_freq == "month":
                return pd.date_range(start=start_date, end=end_date, freq="MS")
            elif ts_freq == "dekad":
                return NotImplemented
            else:
                raise ValueError(f"ts_freq {ts_freq} not supported")

        def fill_sample(sample_df):
            expected_dates = get_expected_dates(
                sample_df["start_date"].iloc[0], sample_df["end_date"].iloc[0], ts_freq
            )
            missing_dates = expected_dates.difference(sample_df["timestamp"])
            if not missing_dates.empty:
                static_cols = sample_df.iloc[0][index_columns].to_dict()
                for date in missing_dates:
                    new_row = {**static_cols, "timestamp": date}
                    for col in FEATURE_COLUMNS:
                        new_row[col] = NODATAVALUE
                    sample_df.loc[-1] = new_row
                    sample_df.reset_index(drop=True, inplace=True)
            return sample_df

        unique_date_pairs = df_long[["start_date", "end_date"]].drop_duplicates()
        unique_date_pairs["expected_n_observations"] = unique_date_pairs.apply(
            lambda xx: len(get_expected_dates(xx["start_date"], xx["end_date"], ts_freq)),
            axis=1,
        )
        unique_date_pairs.set_index(["start_date", "end_date"], inplace=True)

        expected_observations_s = df_long.groupby(["sample_id", "start_date", "end_date"])[
            ["sample_id", "start_date", "end_date", "timestamp"]
        ].apply(lambda xx: xx["timestamp"].nunique())
        expected_observations_s.name = "actual_n_observations"
        expected_observations_df = expected_observations_s.reset_index()
        expected_observations_df.set_index(["start_date", "end_date"], inplace=True)

        expected_observations_df["expected_n_observations"] = expected_observations_df.index.map(
            unique_date_pairs["expected_n_observations"]
        )
        expected_observations_df.reset_index(drop=False, inplace=True)

        samples_to_fill = expected_observations_df[
            expected_observations_df["actual_n_observations"]
            != expected_observations_df["expected_n_observations"]
        ]["sample_id"].unique()

        if samples_to_fill.size == 0:
            logger.info("All samples have the expected number of observations.")
            return df_long
        else:
            logger.warning(
                f"{len(samples_to_fill)} samples have missing observations. \
Filling them with NODATAVALUE."
            )
            df_subset = df_long[df_long["sample_id"].isin(samples_to_fill)]
            df_long = df_long[~df_long["sample_id"].isin(samples_to_fill)]
            df_subset = (
                df_subset.groupby("sample_id")[[*index_columns, *FEATURE_COLUMNS, "timestamp"]]
                .apply(fill_sample)
                .reset_index(drop=True)
            )
            return pd.concat([df_long, df_subset], ignore_index=True)

    @staticmethod
    def add_dummy_timestamps(
        df_long: pd.DataFrame, min_edge_buffer: int, ts_freq: str
    ) -> pd.DataFrame:
        def create_dummy_rows(samples_to_add, n_ts_to_add, direction):
            dummy_df = df_long[
                df_long["sample_id"].isin(samples_to_add)
                & (
                    df_long["timestamp_ind"]
                    == (0 if direction == "before" else df_long["timestamp_ind"].max())
                )
            ].copy()
            offset = pd.DateOffset(months=n_ts_to_add * (1 if direction == "after" else -1))
            dummy_df["timestamp"] += offset
            dummy_df[FEATURE_COLUMNS] = NODATAVALUE
            return dummy_df

        if ts_freq != "month":
            raise ValueError(f"ts_freq {ts_freq} not supported")

        latest_obs_position = df_long.groupby("sample_id")[
            ["valid_position", "timestamp_ind", "valid_position_diff"]
        ].max()

        samples_after_end_date = latest_obs_position[
            latest_obs_position["valid_position"] > latest_obs_position["timestamp_ind"]
        ].index.tolist()
        samples_before_start_date = latest_obs_position[
            latest_obs_position["valid_position"] < 0
        ].index.tolist()

        # print(samples_after_end_date)
        # print(samples_after_end_date)

        if (len(samples_after_end_date) > 0) or (len(samples_before_start_date) > 0):
            logger.warning(
                f"Removing {len(samples_after_end_date)} samples with valid_time \
after the end_date and {len(samples_before_start_date)} samples with valid_time \
before the start_date"
            )
            df_long = df_long[
                ~df_long["sample_id"].isin(samples_before_start_date + samples_after_end_date)
            ]

        intermediate_dummy_df = pd.concat(
            [
                create_dummy_rows(
                    latest_obs_position[
                        (min_edge_buffer - latest_obs_position["valid_position"]) >= -n_ts_to_add
                    ].index,
                    n_ts_to_add,
                    "before",
                )
                for n_ts_to_add in range(1, min_edge_buffer)
            ]
            + [
                create_dummy_rows(
                    latest_obs_position[
                        (min_edge_buffer - latest_obs_position["valid_position_diff"])
                        >= n_ts_to_add
                    ].index,
                    n_ts_to_add,
                    "after",
                )
                for n_ts_to_add in range(1, min_edge_buffer)
            ]
        )

        if not intermediate_dummy_df.empty:
            logger.warning(
                f"Added {intermediate_dummy_df['timestamp'].nunique()} dummy timestamp(s) \
for {intermediate_dummy_df['sample_id'].nunique()} samples to fill in the found gaps."
            )

        df_long = pd.concat([df_long, intermediate_dummy_df])

        # re-initilize all dates and positions with respect to potentially added new timestamps
        df_long["timestamp_ind"] = df_long.groupby("sample_id")["timestamp"].rank().astype(int) - 1
        df_long["start_date"] = df_long.groupby("sample_id")["timestamp"].transform("min")
        df_long["end_date"] = df_long.groupby("sample_id")["timestamp"].transform("max")
        df_long = TimeSeriesProcessor.calculate_valid_position(df_long)

        return df_long


class ColumnProcessor:
    @staticmethod
    def rename_columns(df_long: pd.DataFrame) -> pd.DataFrame:
        return df_long.rename(columns=COLUMN_RENAMES)

    @staticmethod
    def add_band_suffix(df_wide: pd.DataFrame) -> pd.DataFrame:
        df_wide.columns = [
            f"{xx[0]}-ts{xx[1]}" if isinstance(xx[1], int) else xx[0]
            for xx in df_wide.columns.to_flat_index()
        ]  # type: ignore
        df_wide.columns = [
            f"{xx}-10m" if any(band in xx for band in BAND_MAPPINGS["10m"]) else xx
            for xx in df_wide.columns
        ]  # type: ignore
        df_wide.columns = [
            f"{xx}-20m" if any(band in xx for band in BAND_MAPPINGS["20m"]) else xx
            for xx in df_wide.columns
        ]  # type: ignore
        df_wide.columns = [
            (f"{xx}-100m" if any(band in xx for band in BAND_MAPPINGS["100m"]) else xx)
            for xx in df_wide.columns
        ]  # type: ignore
        return df_wide

    @staticmethod
    def construct_index(df_long: pd.DataFrame) -> List[str]:
        # for index columns we need to include all columns that are not feature columns
        index_columns = [col for col in df_long.columns if col not in FEATURE_COLUMNS]
        index_columns.remove("timestamp")
        return index_columns

    @staticmethod
    def check_feature_columns(df_long: pd.DataFrame) -> pd.DataFrame:
        # check that all feature columns are present in the DataFrame
        # or initialize them with NODATAVALUE
        missing_features = [col for col in FEATURE_COLUMNS if col not in df_long.columns]
        if len(missing_features) > 0:
            df_long[missing_features] = NODATAVALUE
            logger.warning(
                f"The following features are missing and are filled \
with NODATAVALUE: {missing_features}"
            )
        return df_long

    @staticmethod
    def check_sar_columns(df_long: pd.DataFrame) -> pd.DataFrame:
        # SAR cannot equal 0.0 since we take the log of it
        # TO DO: need to check the behavior of presto itself in this case
        sar_cols = ["SAR-VV", "SAR-VH"]
        faulty_sar_observations = (df_long[sar_cols] == 0.0).sum().sum()
        if faulty_sar_observations > 0:
            affected_samples = df_long[(df_long[sar_cols] == 0.0).any(axis=1)][
                "sample_id"
            ].nunique()
            logger.warning(
                f"Found {faulty_sar_observations} SAR observation(s) \
equal to 0 across {affected_samples} sample(s). \
Replacing them with NODATAVALUE."
            )
            df_long[sar_cols] = df_long[sar_cols].replace(0.0, NODATAVALUE)

        return df_long


def process_parquet(
    df: pd.DataFrame,
    ts_freq: Literal["month", "dekad"] = "month",
    use_valid_time: bool = True,
    required_min_timesteps: Optional[int] = None,
    min_edge_buffer: int = 2,
    return_after_fill: bool = False,
) -> pd.DataFrame:

    if df.empty:
        raise ValueError("Input DataFrame is empty!")

    # Validate input
    validator = DataFrameValidator()
    validator.validate_required_columns(df)
    validator.validate_timestamps(df, ts_freq)
    df = validator.check_median_distance(df, ts_freq)

    # Process columns
    df = (
        df.pipe(ColumnProcessor.rename_columns)
        .pipe(ColumnProcessor.check_feature_columns)
        .pipe(ColumnProcessor.check_sar_columns)
    )
    index_columns = ColumnProcessor.construct_index(df)

    # Assign start_date and end_date as the minimum and maximum available timestamp
    df["start_date"] = df["sample_id"].map(df.groupby(["sample_id"])["timestamp"].min())
    df["end_date"] = df["sample_id"].map(df.groupby(["sample_id"])["timestamp"].max())
    index_columns.extend(["start_date", "end_date"])
    index_columns = list(set(index_columns))

    # Process time series
    processor = TimeSeriesProcessor()
    df = processor.fill_missing_dates(df, ts_freq, index_columns)
    if return_after_fill:
        return df

    # Initialize timestep_ind
    df["timestamp_ind"] = df.groupby("sample_id")["timestamp"].rank().astype(int) - 1

    if use_valid_time:
        df = processor.calculate_valid_position(df)
        index_columns.append("valid_position")
        df["valid_position_diff"] = df["timestamp_ind"] - df["valid_position"]
        df = processor.add_dummy_timestamps(df, min_edge_buffer, ts_freq)

    df["available_timesteps"] = df["sample_id"].map(
        df.groupby("sample_id")["timestamp"].nunique().astype(int)
    )
    index_columns.append("available_timesteps")
    index_columns = list(set(index_columns))

    # Transform to wide format
    df_pivot = df.pivot(
        index=index_columns,
        columns="timestamp_ind",
        values=FEATURE_COLUMNS,
    )
    df_pivot = df_pivot.fillna(NODATAVALUE)
    if df_pivot.empty:
        raise ValueError("Left with an empty DataFrame!")

    df_pivot.reset_index(inplace=True)
    df_pivot = ColumnProcessor.add_band_suffix(df_pivot)

    if use_valid_time:
        df_pivot["year"] = df_pivot["valid_time"].dt.year
        df_pivot["valid_time"] = df_pivot["valid_time"].dt.date.astype(str)
        df_pivot = validator.check_faulty_samples(df_pivot, min_edge_buffer)

    if required_min_timesteps:
        df_pivot = validator.check_min_timesteps(df_pivot, required_min_timesteps)

    df_pivot["start_date"] = df_pivot["start_date"].dt.date.astype(str)
    df_pivot["end_date"] = df_pivot["end_date"].dt.date.astype(str)
    df_pivot = df_pivot.set_index("sample_id")

    return df_pivot
