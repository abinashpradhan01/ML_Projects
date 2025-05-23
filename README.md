# 🚀 Resume-Worthy Machine Learning Projects (End-to-End)

**Abinash – May 2025**

---

## 📌 Project 1: **EduDrop – Predicting College Dropout Risk**

### 🎯 Objective
Predict whether a student is at risk of dropping out and suggest actionable interventions like counseling, financial aid, or academic support.

### 📊 Dataset
- [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)
- [Students Performance in Exams – Kaggle](https://www.kaggle.com/spscientist/students-performance-in-exams)
- [Higher Education Dropout Dataset](https://www.kaggle.com/datasets/alexandradeis/academic-performance-indicators)

### 🔨 ML Pipeline
- **Data Collection & Merging**
- **EDA & Visualization** (attendance, parental education, study hours, scores)
- **Preprocessing** (imputation, encoding, scaling)
- **Modeling** (Logistic Regression, Random Forest, XGBoost)
- **Interpretability** (SHAP, student-wise reports)
- **Deployment** (Streamlit app + PDF intervention reports)

---

## 🏥 Project 2: **SmartHealth – Lifestyle-Based Disease Risk Detection**

### 🎯 Objective
Predict health risks (diabetes, heart disease, mental health) using survey-style input and generate personalized lifestyle recommendations.

### 📊 Dataset
- [BRFSS 2020 Dataset](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)
- [Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- [Mental Health in Tech Survey](https://www.kaggle.com/osmi/mental-health-in-tech-survey)

### 🔨 ML Pipeline
- **Preprocessing** (class imbalance, encoding)
- **EDA** (risk by BMI, age, activity)
- **Modeling** (Logistic Regression, Random Forest, KNN)
- **Interpretability** (SHAP for recommendations)
- **Deployment** (Streamlit form + recommendation engine)

---

## 🌾 Project 3: **AgriCast – Crop Yield & Rainfall Prediction**

### 🎯 Objective
Forecast rainfall & crop yield based on seasonal data to support farming decisions and minimize agricultural risks.

### 📊 Dataset
- [Indian Crop Yield Dataset](https://www.kaggle.com/datasets/rajanand/crop-production-statistics)
- [Indian Rainfall Dataset](https://www.kaggle.com/datasets/rajanand/rainfall-in-india)

### 🔨 ML Pipeline
- **Data Merging** (district-wise rainfall + yield)
- **EDA** (seasonal trends, drought years)
- **Preprocessing** (grouping, normalization)
- **Modeling**
  - Rainfall: SVR, Linear Regression, XGBoost
  - Yield: Random Forest, Ridge, Polynomial Regression
- **Deployment**
  - Streamlit: District + crop inputs → Forecasts
  - Bonus: Alerts for low-yield/drought predictions

---

## 📰 Project 4: **PolicyWatch – Fake News & Propaganda Detection (Traditional NLP ML)**

### 🎯 Objective
Detect whether a political news piece or tweet is real, fake, or propaganda using traditional NLP-based ML models.

### 📊 Dataset
- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- [Propaganda Techniques Corpus (SemEval 2020)](https://propaganda.qcri.org/semeval2020-task11/)
- [Fake News Kaggle Dataset](https://www.kaggle.com/c/fake-news/data)

### 📎 Features
- **TF-IDF n-grams**
- **Lexical & POS features**
- **Named Entity counts**
- **Sentiment & subjectivity (TextBlob/VADER)**
- **Bias/propaganda lexicon scores**

### 🔨 ML Pipeline
- **Text Cleaning & Preprocessing**
- **EDA** (wordclouds, sentiment spread, propaganda types)
- **Modeling**: SVM, Logistic Regression, Naive Bayes, XGBoost
- **Explainability**: `eli5`, LIME
- **Deployment**: Streamlit app – paste article → get label + explanation

---

## 🔧 Common Tech Stack

| Component             | Tools/Frameworks                              |
|----------------------|-----------------------------------------------|
| Language             | Python 3.10+                                   |
| EDA & Viz            | Pandas, Seaborn, Plotly                        |
| Modeling             | Scikit-learn, XGBoost                          |
| NLP Preprocessing    | NLTK, SpaCy, TextBlob, VADER                   |
| Interpretability     | SHAP, LIME, eli5                               |
| Deployment           | Streamlit (Frontend), FastAPI/Flask (Backend) |
| Hosting              | Streamlit Cloud, Render, HuggingFace Spaces   |
| Model Tracking       | MLflow (optional)                              |
| Version Control      | Git + GitHub                                   |
| PDF Reports          | pdfkit, Jinja2                                 |

---

## 💡 Bonus Tips
- Use Streamlit sidebar for dynamic model selection
- Add explanations with real-world intervention or policy suggestions
- Host & link each app with a professional custom subdomain
- Write SEO-friendly Medium blogs per project explaining your workflow and key learnings

---

