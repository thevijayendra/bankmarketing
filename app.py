# ============================================================
# Bank Marketing - Term Deposit Prediction Dashboard
# Author: Vijayendra Chaudhary
# Description:
#   Streamlit-based ML dashboard to train and evaluate
#   multiple classification models on Bank Marketing dataset.
# ============================================================

import streamlit as st
st.set_page_config(page_title="Bank Deposit Prediction", layout="wide")

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Import modular ML models
from models.logistic_model import LogisticModel
from models.decision_tree_model import DecisionTreeModel
from models.knn_model import KNNModel
from models.naive_bayes_model import NaiveBayesModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel


# ============================================================
# Utility: Model Factory
# ============================================================
def get_model(model_name: str):
    """
    Returns the corresponding model class
    based on user selection.
    """
    model_registry = {
        "Logistic Regression": LogisticModel,
        "Decision Tree": DecisionTreeModel,
        "kNN": KNNModel,
        "Naive Bayes": NaiveBayesModel,
        "Random Forest (Ensemble)": RandomForestModel,
        "XGBoost (Ensemble)": XGBoostModel
    }

    return model_registry[model_name]()


# ============================================================
# Sidebar Configuration
# ============================================================
st.sidebar.markdown("## ⚙️ Model Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV)",
    type=["csv"]
)

# Load dataset
if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    if os.path.exists("bank.csv"):
        data = pd.read_csv("bank.csv")
    else:
        st.error("❌ Dataset not found. Please upload a CSV file.")
        st.stop()

# Training mode selection
training_mode = st.sidebar.radio(
    "Select Training Mode",
    ["Train Selected Model", "Train & Compare All Models"]
)

# Model selection
selected_model_name = st.sidebar.selectbox(
    "Choose Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)"
    ]
)

run_button = st.sidebar.button("🚀 Run Model")


# ============================================================
# Main Dashboard View
# ============================================================
st.title("🏦 Bank Marketing - Term Deposit Prediction")
st.markdown("### 📄 Dataset Preview")
st.dataframe(data.head(), use_container_width=True)


# ============================================================
# Data Preprocessing
# ============================================================

# Convert target column to binary
data['deposit'] = data['deposit'].map({'yes': 1, 'no': 0})

# Encode categorical features
for column in data.select_dtypes(include='object').columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])

# Split features and target
X = data.drop("deposit", axis=1)
y = data["deposit"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# Model Execution Logic
# ============================================================
if run_button:

    # --------------------------------------------------------
    # Train Selected Model
    # --------------------------------------------------------
    if training_mode == "Train Selected Model":

        model_instance = get_model(selected_model_name)

        # Load existing model if available
        if os.path.exists(model_instance.model_path):
            model_instance.load()
            st.info("📂 Loaded saved model from disk.")
        else:
            model_instance.train(X_train, y_train)
            model_instance.save()
            st.success("✅ Model trained and saved successfully.")

        # Make predictions
        y_pred, y_prob = model_instance.predict(X_test)

        # Display Evaluation Metrics
        st.subheader(f"📊 Evaluation Results - {selected_model_name}")

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC Score": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }

        st.dataframe(pd.DataFrame([metrics]).round(4))

        # Visualization Section
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔍 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            st.pyplot(fig)

        with col2:
            st.markdown("### 📑 Classification Report")
            st.text(classification_report(y_test, y_pred))

    # --------------------------------------------------------
    # Train & Compare All Models
    # --------------------------------------------------------
    else:

        st.subheader("📊 Model Comparison Overview")

        model_list = [
            "Logistic Regression",
            "Decision Tree",
            "kNN",
            "Naive Bayes",
            "Random Forest (Ensemble)",
            "XGBoost (Ensemble)"
        ]

        comparison_results = []

        for model_name in model_list:

            model_instance = get_model(model_name)

            if os.path.exists(model_instance.model_path):
                model_instance.load()
            else:
                model_instance.train(X_train, y_train)
                model_instance.save()

            y_pred, y_prob = model_instance.predict(X_test)

            comparison_results.append({
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_prob),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "MCC": matthews_corrcoef(y_test, y_pred)
            })

        comparison_df = pd.DataFrame(comparison_results).round(4)
        st.dataframe(comparison_df)