# Customer Churn Prediction

A machine learning application that predicts customer churn in a telecom company using both custom-implemented and scikit-learn logistic regression models.

## 🔴 Live Demo
[Live Application](https://customerchur.streamlit.app/)

## 📝 Description

This project implements a customer churn prediction system for a telecom company. It features both a custom implementation of logistic regression and a scikit-learn model, providing a comparative analysis of predictions. The application includes an interactive web interface built with Streamlit for real-time predictions.

### What is Customer Churn?
Customer churn refers to when a customer stops using a company's services. In the telecom industry, predicting churn is crucial for:
- Retaining valuable customers
- Reducing customer acquisition costs
- Maintaining stable revenue streams
- Improving service quality

## 🎯 Features

- **Dual Model Prediction System**:
  - Custom-implemented logistic regression using NumPy
  - Scikit-learn's logistic regression implementation
  - Comparative prediction results

- **Interactive Web Interface**:
  - User-friendly form for data input
  - Real-time predictions
  - Visual representation of results
  - Automated retention recommendations

- **Comprehensive Data Processing**:
  - Automated feature scaling
  - Categorical variable encoding
  - Missing value handling
  - Feature order preservation

- **Smart Retention Strategies**:
  - Personalized recommendations based on customer profiles
  - Risk-level assessment
  - Targeted intervention suggestions

## 🛠️ Technical Details

### Models
1. **Custom Logistic Regression**:
   - Implementation: Pure NumPy
   - Features: Gradient descent optimization
   - Sigmoid activation function
   - Customizable learning rate and iterations

2. **Scikit-learn Model**:
   - Implementation: scikit-learn's LogisticRegression
   - Optimized for performance
   - Standard scaling
   - Label encoding for categorical variables

### Features Used
- **Numerical Features**:
  - Tenure
  - Monthly Charges
  - Total Charges
  - Senior Citizen Status

- **Categorical Features**:
  - Contract Type
  - Payment Method
  - Internet Service
  - Online Security
  - Online Backup
  - Device Protection
  - Tech Support
  - Streaming Services
  - Multiple Lines
  - Gender
  - Partner Status
  - Dependents
  - Phone Service
  - Paperless Billing

## 📊 Project Structure

```
Customer_churn_prediction/
├── data/
│   └── customer_churn.csv       # Dataset file
├── models/                      # Saved model files
│   ├── custom_model.pkl
│   ├── sklearn_model.pkl
│   ├── label_encoders.pkl
│   └── scaler.pkl
├── src/
│   ├── __init__.py
│   ├── custom_model.py         # Custom logistic regression implementation
│   ├── preprocess.py          # Data preprocessing functions
│   └── train.py              # Model training scripts
├── app.py                    # Streamlit application
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## 🚀 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Customer_churn_prediction.git
   cd Customer_churn_prediction
   ```

2. **Create and Activate Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Models**
   ```bash
   python src/train.py
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 📦 Dependencies

- Python 3.8+
- NumPy >= 1.26.4
- Pandas >= 2.2.1
- Scikit-learn >= 1.4.1.post1
- Streamlit >= 1.32.2
- Kaggle >= 1.6.6

## 💻 Usage

1. Start the application using `streamlit run app.py`
2. Fill in the customer details in the web form
3. Click "Predict Churn" to get predictions
4. Review the prediction probabilities and risk assessment
5. For high-risk customers, review the recommended retention strategies

## 🎯 Model Performance

### Custom Model Metrics:
- Precision: 0.86 (Class 0), 0.65 (Class 1)
- Recall: 0.89 (Class 0), 0.59 (Class 1)
- F1-score: 0.87 (Class 0), 0.62 (Class 1)
- Overall Accuracy: 0.81

### Scikit-learn Model Metrics:
- Precision: 0.86 (Class 0), 0.68 (Class 1)
- Recall: 0.90 (Class 0), 0.58 (Class 1)
- F1-score: 0.88 (Class 0), 0.62 (Class 1)
- Overall Accuracy: 0.82

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Inspired by real-world telecom industry challenges
- Built with Streamlit's amazing framework


⭐️ If you found this project helpful, please give it a star! 
