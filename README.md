# Bank Marketing - Term Deposit Prediction

## A. Problem Statement

Financial institutions conduct direct marketing campaigns to promote long-term financial products such as term deposits. These campaigns involve contacting customers through phone calls, emails, or other communication channels. However, contacting every customer is costly and inefficient.

The objective of this project is to build a machine learning classification system that can predict whether a customer will subscribe to a term deposit based on their demographic details, financial status, and previous campaign interactions.

By accurately predicting customer responses, the bank can:

- Improve marketing campaign efficiency
- Reduce operational costs
- Target high-potential customers
- Increase subscription rates
- Optimize resource allocation

This project evaluates and compares multiple classification algorithms to determine the best-performing model.

### 🎯 Project Goal

Develop and compare six machine learning models to predict the binary outcome:

- **Yes (1)** – Customer subscribes to term deposit  
- **No (0)** – Customer does not subscribe  

The models are evaluated using the following performance metrics:

- Accuracy  
- AUC (Area Under ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

The final outcome includes a comparison table summarizing the performance of all six models.



## B. Dataset Description

The dataset used in this project is the **Bank Marketing Dataset**, which contains customer information collected during direct marketing campaigns conducted by a banking institution. The purpose of the dataset is to predict whether a client will subscribe to a term deposit.

### 📌 Dataset Type
- Supervised Learning Dataset
- Binary Classification Problem
- Structured Tabular Data

---

### 🎯 Target Variable

| Column Name | Description |
|-------------|------------|
| `deposit`   | Indicates whether the customer subscribed to a term deposit (`yes` / `no`) |

For modeling purposes:
- `yes` → 1  
- `no` → 0  

---

### 📊 Feature Description

The dataset consists of multiple features categorized as follows:

---

#### 1️⃣ Client Information

- `age` – Age of the customer  
- `job` – Type of occupation  
- `marital` – Marital status  
- `education` – Education level  
- `default` – Has credit in default?  
- `balance` – Average yearly account balance  
- `housing` – Has housing loan?  
- `loan` – Has personal loan?  

---

#### 2️⃣ Contact Information

- `contact` – Communication type (e.g., cellular, telephone)  
- `day` – Last contact day of the month  
- `month` – Last contact month  
- `duration` – Duration of last contact (in seconds)  

---

#### 3️⃣ Campaign Information

- `campaign` – Number of contacts during current campaign  
- `pdays` – Days since the client was last contacted  
- `previous` – Number of contacts performed before this campaign  
- `poutcome` – Outcome of previous marketing campaign  

---

### 🔄 Data Preprocessing Performed

The following preprocessing steps were applied:

- Converted target variable (`deposit`) into binary format (1/0)
- Applied **Label Encoding** to categorical variables
- Split dataset into:
  - 80% Training Data
  - 20% Testing Data
- Applied feature scaling for models that require it (e.g., Logistic Regression, kNN)

---

### 📈 Objective of Using This Dataset

The dataset enables the development of predictive models to:

- Identify customers likely to subscribe to term deposits
- Improve marketing campaign efficiency
- Support data-driven decision making in banking operations


## C. Models Used

The following six classification models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

Each model was trained using an 80-20 train-test split and evaluated using the following performance metrics:

- Accuracy  
- AUC (Area Under ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## 📊 Model Comparison Table

| ML Model Name               | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression         | 0.7936   | 0.8668 | 0.7842    | 0.7835 | 0.7839 | 0.5863 |
| Decision Tree               | 0.7721   | 0.7711 | 0.7672    | 0.7507 | 0.7589 | 0.5429 |
| kNN                         | 0.7734   | 0.8451 | 0.7877    | 0.7198 | 0.7522 | 0.5461 |
| Naive Bayes                 | 0.7501   | 0.8128 | 0.7162    | 0.7901 | 0.7513 | 0.5039 |
| Random Forest (Ensemble)    | 0.8370   | 0.9123 | 0.8086    | 0.8632 | 0.8350 | 0.6757 |
| XGBoost (Ensemble)          | 0.8424   | 0.9143 | 0.8178    | 0.8622 | 0.8394 | 0.6858 |

---

### 🏆 Best Performing Model

Based on overall performance (Accuracy, AUC, F1 Score, and MCC),  
**XGBoost (Ensemble)** achieved the highest predictive performance among all models.

- Highest Accuracy: **0.8424**
- Highest AUC: **0.9143**
- Highest F1 Score: **0.8394**
- Highest MCC: **0.6858**

This indicates that XGBoost provides the most balanced and reliable performance for predicting term deposit subscription.


## D. Observations

### 📊 Observations About Model Performance

| ML Model Name               | Observation about model performance |
|-----------------------------|-------------------------------------|
| Logistic Regression         | Performs well on linearly separable data and provides balanced precision-recall performance. |
| Decision Tree               | Shows tendency to overfit but captures non-linear relationships. |
| kNN                         | Sensitive to scaling but performs moderately well. |
| Naive Bayes                 | Works well when feature independence assumption holds. |
| Random Forest (Ensemble)    | Improves generalization and reduces overfitting compared to Decision Tree. |
| XGBoost (Ensemble)          | Achieves highest AUC and overall best performance on the dataset. |