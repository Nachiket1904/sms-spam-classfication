# **Spam SMS Classification: End-to-End Machine Learning Project**

This repository contains an end-to-end machine learning project focused on classifying SMS messages as either **Spam** or **Not Spam**. The project uses the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository and applies various machine learning techniques to build an accurate spam detection model. The project culminates in a web-based application deployed using **Streamlit**.

## **Table of Contents**
- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Deployment](#deployment)
- [Acknowledgements](#acknowledgements)

## **Project Overview**

The aim of this project is to classify SMS messages as spam or not spam. The workflow includes:
- Data loading and cleaning.
- Exploratory Data Analysis (EDA).
- Text preprocessing and transformation.
- Building machine learning models.
- Model evaluation.
- Deployment of a Streamlit app to allow users to classify new SMS messages.

### **Key Components:**
- **Text Preprocessing**: Includes tokenization, stopword removal, stemming, and vectorization using TF-IDF.
- **Machine Learning Models**: Several models including Naive Bayes, Logistic Regression, Support Vector Machines, Random Forest, and Gradient Boosting are trained and compared.
- **Deployment**: A user-friendly web interface is built using Streamlit for real-time predictions.

## **Data Source**
- **Dataset**: [Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection](https://doi.org/10.24432/C5CC84), UCI Machine Learning Repository.

## **Installation**

1. **Clone the repository**:
```bash
git clone https://github.com/Nachiket1904/Spam-SMS-Classification.git
```
2.**Navigate to the project directory**:
```bash
cd Spam-SMS-Classification
```
3.**Install the required dependencies**:
```bash
pip install -r requirements.txt
```
4.Download the dataset:
Download the dataset from the [UCI Machine Learning Repository](https://doi.org/10.24432/C5CC84) and save it as spam.csv in the dataset/ directory.

## **Usage**

### **Run the Streamlit App:**
To launch the Streamlit web application, follow these steps:

1.**Run the App**:
   ```bash
   streamlit run app.py
   ```
2.**Use the Application**:
- Enter an SMS message into the text input box on the Streamlit interface.
- Click the Predict button to classify the message as either Spam or Not Spam.

## **Modeling and Evaluation**

The following machine learning models were trained and evaluated:

- **Naive Bayes** (Multinomial, Bernoulli, Gaussian)
- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **Random Forest**
- **Gradient Boosting**

### **Performance Metrics**:
- **Accuracy**
- **Precision**
- **Confusion Matrix**

After testing various models, **Multinomial Naive Bayes** performed the best for classifying spam messages. The results of each model were evaluated based on accuracy and precision to determine the best-performing classifier.

## **Deployment**

The final model is deployed as a web application using **Streamlit Cloud**, allowing users to interactively classify SMS messages as spam or not spam.

- **Model**: The trained model is saved as `model1.pkl`.
- **Vectorizer**: The TF-IDF vectorizer used to preprocess the input text is saved as `vectorizer1.pkl`.

To access the deployed app, [click here](https://streamlit.cloud/your-app-url).

## **Acknowledgements**

The dataset is sourced from the [UCI Machine Learning Repository](https://doi.org/10.24432/C5CC84), created by **Tiago Almeida** and **José Hidalgo**.

