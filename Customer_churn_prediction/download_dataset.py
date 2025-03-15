import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset():
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    print("Downloading Telco Customer Churn dataset...")
    api.dataset_download_files(
        "blastchar/telco-customer-churn", path="data", unzip=True
    )

    # Rename the file if necessary
    if os.path.exists("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        os.rename(
            "data/WA_Fn-UseC_-Telco-Customer-Churn.csv", "data/customer_churn.csv"
        )

    print("Dataset downloaded successfully to data/customer_churn.csv")


if __name__ == "__main__":
    download_dataset()
