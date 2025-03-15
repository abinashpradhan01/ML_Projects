import numpy as np
import pandas as pd
import pickle
import logging
import os
from typing import Dict, List, Union, Optional, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CustomLinearRegression:
    """A custom implementation of linear regression using the normal equation."""

    def __init__(self):
        """Initialize the model with empty parameters."""
        self.theta = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomLinearRegression":
        """
        Fit the linear regression model using the normal equation.

        Parameters:
        -----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)

        Returns:
        --------
        self : CustomLinearRegression
            Returns self.

        Raises:
        -------
        ValueError
            If the input data is invalid or computation fails
        """
        try:
            # Check input
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have the same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}"
                )

            # Add intercept term
            X_with_intercept = np.c_[np.ones(X.shape[0]), X]

            # Calculate theta using the normal equation with regularization for numerical stability
            # Add a small value to diagonal elements to ensure matrix is invertible
            XTX = X_with_intercept.T.dot(X_with_intercept)
            reg_term = 1e-8 * np.eye(XTX.shape[0])
            self.theta = np.linalg.inv(XTX + reg_term).dot(X_with_intercept.T).dot(y)

            self.fitted = True
            logger.info(
                f"CustomLinearRegression fitted successfully with {X.shape[1]} features"
            )

            return self

        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error during fitting: {str(e)}")
            raise ValueError(f"Failed to compute normal equation: {str(e)}")
        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)

        Returns:
        --------
        np.ndarray
            Predicted values of shape (n_samples,)

        Raises:
        -------
        ValueError
            If the model has not been fitted or input data is invalid
        """
        if not self.fitted or self.theta is None:
            raise ValueError("Model has not been fitted yet! Call fit before predict.")

        try:
            # Add intercept term
            X_with_intercept = np.c_[np.ones(X.shape[0]), X]

            # Check dimensions
            if X_with_intercept.shape[1] != len(self.theta):
                raise ValueError(
                    f"Feature dimension mismatch. Model was trained with {len(self.theta)-1} features, "
                    f"but received {X.shape[1]} features."
                )

            return X_with_intercept.dot(self.theta)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise


class HousePriceModel:
    """
    House Price prediction model that supports both custom and scikit-learn implementations.
    This class handles feature scaling and prediction.
    """

    def __init__(self, use_custom: bool = False):
        """
        Initialize the model.

        Parameters:
        -----------
        use_custom : bool, default=False
            If True, use custom implementation, else use scikit-learn's LinearRegression
        """
        self.model = CustomLinearRegression() if use_custom else LinearRegression()
        self.scaler = None
        self.feature_names = None
        self.use_custom = use_custom
        self.is_fitted = False

        logger.info(
            f"Initialized HousePriceModel with {'custom' if use_custom else 'scikit-learn'} implementation"
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "HousePriceModel":
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X_train : np.ndarray
            Training features of shape (n_samples, n_features)
        y_train : np.ndarray
            Target values of shape (n_samples,)

        Returns:
        --------
        self : HousePriceModel
            Returns self.

        Raises:
        -------
        ValueError
            If the input data is invalid
        """
        try:
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(
                    f"X_train and y_train must have the same number of samples"
                )

            self.model.fit(X_train, y_train)
            self.is_fitted = True

            logger.info(
                f"Model fitted successfully with {X_train.shape[1]} features and {X_train.shape[0]} samples"
            )
            return self

        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Parameters:
        -----------
        X : Union[pd.DataFrame, np.ndarray]
            The input samples to predict of shape (n_samples, n_features)

        Returns:
        --------
        np.ndarray
            The predicted values of shape (n_samples,)

        Raises:
        -------
        ValueError
            If the model has not been fitted or scaler is not initialized
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet!")

        if self.scaler is None:
            raise ValueError("Model has not been initialized with a scaler!")

        try:
            # Convert to numpy array if DataFrame
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X

            # Scale the features using the saved scaler
            X_scaled = self.scaler.transform(X_array)

            # Make predictions
            predictions = self.model.predict(X_scaled)

            logger.info(f"Made predictions for {X_array.shape[0]} samples")
            return predictions

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.

        Parameters:
        -----------
        X_test : np.ndarray
            Test features of shape (n_samples, n_features)
        y_test : np.ndarray
            True target values of shape (n_samples,)

        Returns:
        --------
        Dict[str, float]
            Dictionary with evaluation metrics (mse, rmse, r2)

        Raises:
        -------
        ValueError
            If the model has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet!")

        try:
            predictions = self.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mse)

            metrics = {"mse": mse, "rmse": rmse, "r2": r2}
            logger.info(
                f"Model evaluation: MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise

    def save_model_info(self, scaler: Any, feature_names: List[str]) -> None:
        """
        Save additional model information.

        Parameters:
        -----------
        scaler : Any
            Fitted scaler for feature scaling
        feature_names : List[str]
            Names of features used for training

        Raises:
        -------
        ValueError
            If inputs are invalid
        """
        if scaler is None:
            raise ValueError("Scaler cannot be None")

        if feature_names is None or len(feature_names) == 0:
            raise ValueError("Feature names cannot be empty")

        self.scaler = scaler
        self.feature_names = feature_names

        logger.info(f"Model information saved with {len(feature_names)} features")

    def save_model(self, filepath: str = "house_price_model.pkl") -> None:
        """
        Save the model to a pickle file.

        Parameters:
        -----------
        filepath : str, default="house_price_model.pkl"
            Path to save the model

        Raises:
        -------
        ValueError
            If the model has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save a model that has not been fitted")

        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @staticmethod
    def load_model(filepath: str = "house_price_model.pkl") -> "HousePriceModel":
        """
        Load a saved model from a pickle file.

        Parameters:
        -----------
        filepath : str, default="house_price_model.pkl"
            Path to the saved model

        Returns:
        --------
        HousePriceModel
            Loaded model

        Raises:
        -------
        FileNotFoundError
            If the model file is not found
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        try:
            with open(filepath, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
