import logging
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.composition import parse_formula

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_feature_set(feature_set_name: str) -> List[str]:
    """
    Returns a list of additional feature column names based on the specified feature set.

    Args:
        feature_set_name (str): Identifier for the desired feature set.

    Returns:
        List[str]: List of additional column names to be used as features.
    """
    feature_sets = {
        "no_additional_features": [],
        "label_features": ["label"],
        "qr_label_features": ["qr_label5"],
        # Add more feature sets as needed
    }

    if feature_set_name not in feature_sets:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

    return feature_sets[feature_set_name]


def load_and_prepare_data(full_data_loc: str) -> pd.DataFrame:
    """
    Loads the dataset, binarizes labels, renames columns, and ensures required columns exist.

    Args:
        full_data_loc (str): Path to the CSV data file.

    Returns:
        pd.DataFrame: Prepared DataFrame.
    """
    # Load data
    df = pd.read_csv(full_data_loc)
    logging.info(f"Loaded data from {full_data_loc} with shape {df.shape}.")

    # Binarize labels
    df["label"] = np.where(df["label"] == 1, 1, 0).astype(int)
    df = df.rename(columns={"label": "target", "composition": "formula"})
    df["label"] = df["target"].values
    logging.info("Binarized 'label' column and renamed 'composition' to 'formula'.")

    return df


def filtered_train_val_split(
    df: pd.DataFrame,
    excluded_elements: Optional[List[str]],
    out_dir: str,
    val_frac: float = 0.1,
    random_state: int = 42,
    write: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training, validation, and test sets based on excluded elements.

    If `excluded_elements` is provided, entries containing all these elements are used as the test set.
    Otherwise, the data is split into train and validation sets without a separate test set.

    Args:
        df (pd.DataFrame): The input DataFrame.
        excluded_elements (Optional[List[str]]): List of chemical symbols to exclude for the test set.
        out_dir (str): Directory to save the split datasets.
        val_frac (float, optional): Fraction of data to use for the validation set. Defaults to 0.1.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        write (bool, optional): Whether to write the splits to disk. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and test DataFrames.
            - If `excluded_elements` is None, the test DataFrame will be empty.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Parse compositions
    df["pmg_composition"] = df["formula"].apply(parse_formula)
    df = df[df["pmg_composition"].notnull()]
    logging.info(f"Initial dataset size: {df.shape[0]} entries.")

    if excluded_elements:
        exclude_str = "-".join(excluded_elements)
        logging.info(f"Excluding elements for test set: {exclude_str}")

        # Create mask for excluded elements
        mask = (
            df["pmg_composition"]
            .apply(lambda comp: all(elem in comp for elem in excluded_elements))
            .values
        )

        # Split into test and remaining data
        test_df = df[mask].drop(columns=["pmg_composition"])
        remaining_df = df[~mask].drop(columns=["pmg_composition"])

        logging.info(f"Test set size (excluded elements): {test_df.shape[0]} entries.")
        logging.info(f"Remaining data size: {remaining_df.shape[0]} entries.")

        # Train-validation split on remaining data
        train_df, val_df = train_test_split(
            remaining_df,
            test_size=val_frac,
            random_state=random_state,
            stratify=remaining_df["target"],
        )
    else:
        logging.info(
            "No excluded elements provided. Performing train-validation split without a separate test set."
        )
        test_df = pd.DataFrame()  # Empty DataFrame
        train_df, val_df = train_test_split(
            df.drop(columns=["pmg_composition"]),
            test_size=val_frac,
            random_state=random_state,
            stratify=df["target"],
        )

    logging.info(f"Training set size: {train_df.shape[0]} entries.")
    logging.info(f"Validation set size: {val_df.shape[0]} entries.")
    if excluded_elements:
        logging.info(f"Test set size: {test_df.shape[0]} entries.")

    # Write to disk if required
    if write:
        train_path = os.path.join(out_dir, "train.csv")
        val_path = os.path.join(out_dir, "val.csv")
        test_path = os.path.join(out_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        if excluded_elements:
            test_df.to_csv(test_path, index=False)

        logging.info(f"Saved training set to {train_path}")
        logging.info(f"Saved validation set to {val_path}")
        if excluded_elements:
            logging.info(f"Saved test set to {test_path}")

    return train_df, val_df, test_df


def generate_and_save_datasets(
    excluded_elements: Optional[List[str]],
    features: List[str],
    full_data_loc: str,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates and saves training, validation, and test datasets based on the specified features and excluded elements.

    Args:
        excluded_elements (Optional[List[str]]): List of chemical symbols to exclude for the test set.
            If None, performs train-validation split without a separate test set.
        features (List[str]): List of additional feature column names to include.
        full_data_loc (str): Path to the full CSV data file.
        output_dir (str): Base directory where the split datasets will be saved.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and test DataFrames.
    """
    # Load and prepare data
    df = load_and_prepare_data(full_data_loc)

    # Implicitly include 'formula' and 'target'
    required_columns = ["formula", "target"]
    all_features = required_columns + features

    # Validate that all specified features exist in the DataFrame
    missing_features = set(all_features) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"The following required features are missing from the dataset: {missing_features}"
        )

    # Select relevant columns
    df = df[all_features]
    logging.info(f"Selected columns for dataset: {all_features}")

    # Define output subdirectory based on feature set and excluded elements
    if excluded_elements:
        excluded_str = "-".join(excluded_elements)
        subdir_suffix = f"with-{excluded_str}"
    else:
        subdir_suffix = "no-holdout"

    features_str = "_".join(features)
    if len(features_str) > 30:
        features_str = "_".join([f[0:3] for f in features]) + "_etc"

    subdir = os.path.join(output_dir, f"features-{'_'.join(features)}", subdir_suffix)
    logging.info(f"Output directory for splits: {subdir}")

    # Perform train-validation-test split
    train_df, val_df, test_df = filtered_train_val_split(
        df=df,
        excluded_elements=excluded_elements,
        out_dir=subdir,
        val_frac=0.1,
        random_state=42,
        write=True,
    )

    return train_df, val_df, test_df


def initialize_results(all_model_results_path: str) -> pd.DataFrame:
    """
    Initializes or loads the model results DataFrame.

    Args:
        all_model_results_path (str): Path to the results CSV file.

    Returns:
        pd.DataFrame: DataFrame to store model results.
    """
    if os.path.exists(all_model_results_path):
        results_df = pd.read_csv(all_model_results_path)
        logging.info(f"Loaded existing model results from {all_model_results_path}.")
    else:
        results_df = pd.DataFrame(
            columns=[
                "model_name",
                "train_f1",
                "train_auc",
                "train_acc",
                "val_f1",
                "val_auc",
                "val_acc",
                "test_f1",
                "test_auc",
                "test_acc",
            ]
        )
        logging.info("Initialized a new model results DataFrame.")
    return results_df


def main():
    # Configuration
    DATA_DIR = "data/materials_data/oxides"
    MAT_PROP = "vitrification"
    FULL_DATA_LOC = os.path.join(DATA_DIR, "data_v2_added-round2_None.csv")
    RESULTS_PATH = f"results/_all_model_results_{MAT_PROP}.csv"
    OUTPUT_BASE_DIR = f"{DATA_DIR}-processed"

    # Define feature sets using get_feature_set
    feature_set_names = [
        "no_additional_features",
        "label_features",
        "qr_label_features",
    ]

    # Define holdout options
    holdout_options = [
        ["Fe", "P", "O"],  # Example holdout symbols
        # None  # Option without holdout
    ]

    # Initialize results DataFrame
    all_model_results = initialize_results(RESULTS_PATH)

    # Iterate over feature sets and holdout options
    for feature_set_name in feature_set_names:
        # Retrieve the list of additional features
        try:
            additional_features = get_feature_set(feature_set_name)
        except ValueError as e:
            logging.error(e)
            continue

        for excluded_elements in holdout_options:
            if excluded_elements:
                excluded_str = "-".join(excluded_elements)
                logging.info(
                    f"Processing feature set '{feature_set_name}' with excluded elements: {excluded_str}"
                )
            else:
                logging.info(
                    f"Processing feature set '{feature_set_name}' without excluded elements."
                )

            # Generate and save datasets
            try:
                train_df, val_df, test_df = generate_and_save_datasets(
                    excluded_elements=excluded_elements,
                    features=additional_features,
                    full_data_loc=FULL_DATA_LOC,
                    output_dir=OUTPUT_BASE_DIR,
                )
            except Exception as e:
                logging.error(
                    f"Failed to generate datasets for feature set '{feature_set_name}' with excluded elements '{excluded_elements}': {e}"
                )
                continue

            # Placeholder for model training and evaluation
            # TODO: Implement model training and evaluation here
            # Example:
            # from your_model_library import YourModel
            #
            # model = YourModel(parameters)
            # model.fit(train_df[additional_features], train_df['target'])
            #
            # val_metrics = model.evaluate(val_df[additional_features], val_df['target'])
            # if not test_df.empty:
            #     test_metrics = model.evaluate(test_df[additional_features], test_df['target'])
            #
            # result = {
            #     "model_name": f"{feature_set_name}-holdout-{excluded_str}" if excluded_elements else f"{feature_set_name}-no-holdout",
            #     "train_f1": val_metrics['f1'],
            #     "train_auc": val_metrics['auc'],
            #     "train_acc": val_metrics['accuracy'],
            #     "val_f1": val_metrics_val['f1'],
            #     "val_auc": val_metrics_val['auc'],
            #     "val_acc": val_metrics_val['accuracy'],
            #     "test_f1": test_metrics['f1'] if excluded_elements else np.nan,
            #     "test_auc": test_metrics['auc'] if excluded_elements else np.nan,
            #     "test_acc": test_metrics['accuracy'] if excluded_elements else np.nan,
            # }
            # all_model_results = all_model_results.append(result, ignore_index=True)

    # # Save all model results
    # all_model_results.to_csv(RESULTS_PATH, index=False)
    # logging.info(f"All model results saved to {RESULTS_PATH}.")


if __name__ == "__main__":
    main()
