from setuptools import setup, find_packages

setup(
    name="customer_churn_prediction",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.1",
        "scikit-learn>=1.4.1.post1",
        "streamlit>=1.32.2",
        "joblib>=1.3.2",
    ],
    author="YourName",
    author_email="your.email@example.com",
    description="A machine learning application that predicts customer churn in a telecom company",
    keywords="machine learning, churn prediction, streamlit",
    url="https://github.com/yourusername/Customer_churn_prediction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
