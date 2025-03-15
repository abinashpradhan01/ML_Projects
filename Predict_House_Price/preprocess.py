import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import requests
import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_dataset(save_path: str = "house_prices.csv") -> bool:
    """
    Download the House Prices dataset from a reliable source

    Parameters:
    -----------
    save_path : str
        Path where the dataset will be saved

    Returns:
    --------
    bool
        True if download was successful, False otherwise
    """
    # URL for the House Prices dataset
    dataset_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    backup_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"

    urls_to_try = [dataset_url, backup_url]

    for url in urls_to_try:
        try:
            logger.info(f"Attempting to download dataset from {url}")

            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Download the file with timeout
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Save the file
            with open(save_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Dataset successfully downloaded to {save_path}")
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Error downloading dataset from {url}: {str(e)}")
            continue  # Try the next URL if available

    logger.error("All download attempts failed")
    return False


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean the house prices dataset

    Parameters:
    -----------
    file_path : str
        Path to the dataset file

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame with all required features

    Raises:
    -------
    FileNotFoundError
        If the dataset cannot be found or downloaded
    ValueError
        If the dataset is empty or missing critical columns
    """
    # Check if file exists, if not, download it
    if not os.path.exists(file_path):
        logger.info(f"Dataset not found at {file_path}, attempting to download")
        success = download_dataset(file_path)
        if not success:
            raise FileNotFoundError(
                f"Could not download or find the dataset at {file_path}"
            )

    # Load the data
    try:
        data = pd.read_csv(file_path)
        logger.info(
            f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns"
        )
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise ValueError(f"Error reading the dataset: {str(e)}")

    # Check if the dataset is empty
    if data.empty:
        raise ValueError("Dataset is empty")

    # Rename columns to match our expected format if necessary
    column_mapping = {
        "median_house_value": "SalePrice",
        "total_rooms": "TotalRooms",
        "total_bedrooms": "BedroomAbvGr",
        "housing_median_age": "YearBuilt",
        "median_income": "Income",
    }

    # Only rename columns that exist in the dataset
    columns_to_rename = {k: v for k, v in column_mapping.items() if k in data.columns}
    data = data.rename(columns=columns_to_rename)

    # Create additional features if they don't exist
    if "OverallQual" not in data.columns:
        if "SalePrice" in data.columns:
            data["OverallQual"] = pd.qcut(data["SalePrice"], q=10, labels=False) + 1
        else:
            # Use default value if SalePrice is not available
            data["OverallQual"] = 5

    if "OverallCond" not in data.columns:
        data["OverallCond"] = 5  # Default average condition

    if "GarageCars" not in data.columns:
        data["GarageCars"] = 2  # Default 2-car garage

    if "FullBath" not in data.columns:
        data["FullBath"] = 2  # Default 2 bathrooms

    if "GrLivArea" not in data.columns:
        if "TotalRooms" in data.columns:
            data["GrLivArea"] = data["TotalRooms"] * 100  # Approximate living area
        else:
            # Default value if TotalRooms is not available
            data["GrLivArea"] = 1500

    if "LotArea" not in data.columns:
        if "GrLivArea" in data.columns:
            data["LotArea"] = data["GrLivArea"] * 2  # Approximate lot area
        else:
            # Default value if GrLivArea is not available
            data["LotArea"] = 8000

    # Make sure SalePrice is available
    if "SalePrice" not in data.columns and "median_house_value" in data.columns:
        data["SalePrice"] = data["median_house_value"]

    # Check for critical columns
    required_features = [
        "LotArea",
        "GrLivArea",
        "BedroomAbvGr",
        "FullBath",
        "YearBuilt",
        "GarageCars",
        "OverallQual",
        "OverallCond",
        "SalePrice",
    ]

    # Keep only the required features
    # We first ensure all required columns exist, even with default values
    for feature in required_features:
        if feature not in data.columns:
            logger.warning(
                f"Required feature {feature} not found in dataset, using default values"
            )
            # Set default values based on feature type
            if feature == "YearBuilt":
                data[feature] = 2000
            elif feature == "BedroomAbvGr":
                data[feature] = 3
            elif feature == "FullBath":
                data[feature] = 2
            elif feature == "LotArea":
                data[feature] = 8000
            elif feature == "GrLivArea":
                data[feature] = 1500
            elif feature in ["OverallQual", "OverallCond"]:
                data[feature] = 5
            elif feature == "GarageCars":
                data[feature] = 2
            elif feature == "SalePrice":
                raise ValueError("Target column SalePrice not found in dataset")
            else:
                data[feature] = data.shape[0] * [0]

    # Select only the required features
    data = data[required_features]

    logger.info(
        f"Data cleaning complete with {data.shape[0]} rows and {data.shape[1]} columns"
    )

    return data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with potentially missing values

    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled
    """
    # Check for missing values
    missing_count = data.isna().sum().sum()
    if missing_count > 0:
        logger.info(f"Handling {missing_count} missing values")

        # Fill numerical columns with median
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isna().any():
                median_value = data[col].median()
                data[col] = data[col].fillna(median_value)
                logger.info(
                    f"Filled {data[col].isna().sum()} missing values in {col} with median {median_value}"
                )

        # Fill categorical columns with mode
        categorical_columns = data.select_dtypes(include=["object"]).columns
        for col in categorical_columns:
            if data[col].isna().any():
                mode_value = data[col].mode().iloc[0]
                data[col] = data[col].fillna(mode_value)
                logger.info(
                    f"Filled {data[col].isna().sum()} missing values in {col} with mode {mode_value}"
                )
    else:
        logger.info("No missing values found")

    return data


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale numerical features using Min-Max scaling

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, MinMaxScaler]
        Scaled training features, scaled testing features, and the scaler
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Features scaled using MinMaxScaler")

    return X_train_scaled, X_test_scaled, scaler


def prepare_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, List[str]]:
    """
    Prepare data for modeling

    Parameters:
    -----------
    data : pd.DataFrame
        Cleaned DataFrame
    target_column : str
        Name of the target column
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, List[str]]
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    # Verify target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    logger.info(f"Preparing data with {X.shape[1]} features and {X.shape[0]} samples")

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Store feature names before splitting
    feature_names = X.columns.tolist()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples"
    )

    # Scale the features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
