# 🌸 Iris Flower Classification Project

This project implements a machine learning solution for classifying Iris flowers into their respective species using both K-Nearest Neighbors (KNN) and Decision Tree algorithms. The project includes model training, evaluation, and an interactive Streamlit web application for real-time predictions.

## 📊 Dataset

The project uses the famous Iris dataset from scikit-learn, which includes:
- 150 samples
- 3 different species of Iris (Setosa, Versicolor, Virginica)
- 4 features: sepal length, sepal width, petal length, and petal width
- All features are measured in centimeters

## 🛠️ Project Structure

```
IRIS_FLOWER_CLASSIFICATION/
│
├── model.py           # Model training and evaluation script
├── streamlit_app.py            # Streamlit web application
├── requirements.txt  # Project dependencies
├── README.md        # Project documentation
│
└── models/          # Directory for saved models (created during training)
    ├── knn_model.pkl
    ├── dt_model.pkl
    ├── scaler.pkl
    ├── knn_confusion_matrix.png
    └── dt_confusion_matrix.png
```


### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/IRIS_FLOWER_CLASSIFICATION.git
cd IRIS_FLOWER_CLASSIFICATION
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. Train the models:
```bash
python model.py
```
This will:
- Load and preprocess the Iris dataset
- Train both KNN and Decision Tree models
- Generate evaluation metrics and confusion matrices
- Save the trained models and scaler

2. Run the Streamlit app:
```bash
streamlit run app.py
```
This will launch a web interface where you can:
- Input flower measurements
- Get predictions from both models
- View probability distributions for predictions

## 📈 Model Performance

The project includes two different models:
1. K-Nearest Neighbors (KNN)
   - Uses 5 neighbors
   - Features are scaled using StandardScaler
   
2. Decision Tree
   - Uses default parameters
   - Provides feature importance information

Both models are evaluated using:
- Accuracy score
- Confusion matrix
- Classification report (precision, recall, f1-score)

## 🌟 Features

- Interactive web interface
- Real-time predictions
- Probability distributions for predictions
- Comparison between two different models
- Data preprocessing and feature scaling
- Comprehensive model evaluation

## 📱 Web Application

The Streamlit app provides:
- Input fields for all four flower measurements
- Side-by-side model predictions
- Probability distributions for each species
- User-friendly interface
- Instant predictions


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 
