# 🏠 House Price Prediction

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.1%2B-orange)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

A robust machine learning application that predicts house prices using both custom and scikit-learn linear regression models. This application is deployment-ready and uses best practices for ML application development.

## ✨ Features

- **Interactive Web Interface**: Intuitive Streamlit web application for easy interaction
- **Automatic Dataset Management**: Automatic downloading and preprocessing of housing data
- **Robust Error Handling**: Comprehensive error management and user feedback
- **Feature Importance Visualization**: See which features impact the price prediction most
- **Performance Metrics**: View model performance statistics (R², RMSE, MSE)
- **Sample Data Templates**: Pre-configured examples for different housing types
- **Dual Model Implementation**: Both custom and scikit-learn linear regression

## 🛠️ Technologies Used

- Python 3.8+
- NumPy 1.26.4+
- Pandas 2.2.1+
- Scikit-learn 1.4.1+
- Streamlit 1.31.1+
- Matplotlib 3.8.3+
- Seaborn 0.13.2+

## 📋 Requirements

- Python 3.8 or higher
- pip package manager
- Internet connection (first run only, for dataset download)

## 🚀 Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📊 Usage

1. The application will open in your default web browser at `http://localhost:8501`
2. Enter house features in the input form:
   - Lot Area (sqft)
   - Living Area (sqft)
   - Number of Bedrooms
   - Number of Bathrooms
   - Year Built
   - Garage Capacity
   - Overall Quality
   - Overall Condition
3. Click the "Predict Price" button
4. View the predicted house price and feature importance visualization

## 🗂️ Project Structure

```
house-price-prediction/
├── app.py                # Streamlit web application
├── model.py              # Machine learning model implementation
├── preprocess.py         # Data preprocessing utilities
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── house_prices.csv      # Dataset (downloaded automatically if not present)
```


## 🔍 Data Sources

The application uses the California Housing dataset, which includes information about houses in California. The dataset features include:
- Median income
- House age
- Number of rooms
- Number of bedrooms
- Population
- Households
- Latitude
- Longitude

## 🧪 Model Performance

The linear regression model achieves:
- **R² Score**: Typically between 0.65-0.75
- **RMSE**: Average error of approximately $50,000 - $70,000
- **MSE**: Mean squared error around 2.5e9 - 5e9



## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
