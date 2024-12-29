import logging
import os

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def filter_outliers(df):
    df = df[~((df["name"] == "PULSE_RATE") & (df["doublevalue"] > 300))]
    df = df[~((df["name"] == "RESPIRATORY_RATE") & (df["doublevalue"] > 60))]
    df = df[
        ~(
            (df["name"] == "SPO2")
            & ((df["doublevalue"] < 50) | (df["doublevalue"] > 100))
        )
    ]
    df = df[~((df["name"] == "COVERED_SKIN_TEMPERATURE") & (df["doublevalue"] > 45))]
    return df


def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())


def reverse_min_max_normalize(column, min_val, max_val):
    return column * (max_val - min_val) + min_val


def prepare_gmm_data(pivot_df, num_timesteps, time_size, vital_signs, min_entries=1):
    input_data = []
    output_data = []
    valid_patient_ids = set()
    # patient_rewards = {}

    for patient_id, group in pivot_df.groupby("patient_id"):
        group = group.sort_values("generatedat")
        valid_sequences = []
        # rewards = []

        for i in range(len(group) - num_timesteps):
            valid_sequence = True
            base_time = group.iloc[i]["generatedat"]
            for j in range(1, num_timesteps + 1):
                expected_time = base_time + pd.Timedelta(minutes=time_size * j)
                if group.iloc[i + j]["generatedat"] != expected_time:
                    valid_sequence = False
                    break
            if valid_sequence:
                input_values = group.iloc[i : i + num_timesteps][
                    vital_signs
                ].values.flatten()
                output_values = group.iloc[i + num_timesteps][vital_signs].values
                valid_sequences.append(
                    (input_values.astype(float), output_values.astype(float))
                )
                # rewards.append(group.iloc[i + num_timesteps]["reward"])

        # patient_rewards[patient_id] = np.mean(rewards)
        if len(valid_sequences) > min_entries:
            valid_patient_ids.add(patient_id)
            for input_values, output_values in valid_sequences:
                input_data.append(input_values)
                output_data.append(output_values)

    return np.hstack((np.array(input_data), np.array(output_data)))


@hydra.main(config_path="conf", config_name="train_gmm", version_base=None)
def main(cfg: DictConfig):
    logger.info("Loading and preprocessing the data")
    df = (
        pd.read_csv(cfg.dataset.path)
        .assign(
            generatedat=lambda x: pd.to_datetime(x["generatedat"], errors="coerce"),
            doublevalue=lambda x: pd.to_numeric(x["doublevalue"], errors="coerce"),
        )
        .dropna(subset=["doublevalue", "patient_id"])
        .pipe(filter_outliers)  # Filter out abnormal values (outliers)
        .set_index("generatedat")
        .sort_index()
    )

    logger.info("Resampling the data and computing median values")
    resampled_df = (
        df.groupby(["patient_id", "name"])
        .resample(f"{cfg.dataset.time_size}min")["doublevalue"]
        .median()
        .dropna()
        .reset_index()
    )

    logger.info("Pivoting the dataframe to have vital signs as columns")
    pivot_df = (
        resampled_df.pivot_table(
            index=["patient_id", "generatedat"], columns="name", values="doublevalue"
        )
        .dropna()
        .reset_index()
    )

    logger.info("Normalizing the data")  # normalize signs in separate column
    min_max = {}
    vital_signs = cfg.dataset.vital_signs
    for sign in vital_signs:
        min_max[sign] = [pivot_df[sign].min(), pivot_df[sign].max()]
        pivot_df[sign] = min_max_normalize(pivot_df[sign])

    logger.info("Preparing data for GMM training")
    gmm_training_data = prepare_gmm_data(
        pivot_df,
        cfg.dataset.num_timesteps,
        cfg.dataset.time_size,
        cfg.dataset.vital_signs,
        cfg.dataset.min_entries,
    )

    logger.info("Fitting the Gaussian Mixture Model")
    gmm = GaussianMixture(
        n_components=cfg.dataset.num_comp, covariance_type="full", random_state=42
    )
    gmm.fit(gmm_training_data)

    # save output
    os.makedirs("models/", exist_ok=True)
    model_name = HydraConfig.get().runtime.choices.dataset
    model_save_path = f"models/{model_name}.npz"
    scaler_min = np.array([min_max[k][0] for k in cfg.dataset.vital_signs])
    scaler_max = np.array([min_max[k][1] for k in cfg.dataset.vital_signs])
    np.savez(
        model_save_path,
        means=gmm.means_,
        covariances=gmm.covariances_,
        weights=gmm.weights_,
        scaler_min=scaler_min,
        scaler_max=scaler_max,
        names=np.array(cfg.dataset.vital_signs, dtype=str),
    )
    logger.info("GMM parameters saved to %s", model_save_path)


if __name__ == "__main__":
    main()
