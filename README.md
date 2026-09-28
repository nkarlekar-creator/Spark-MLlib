# Spark MLlib Classification and Feature Engineering

This project demonstrates two practical Spark MLlib tasks:

1. Load a dataset into Apache Spark and implement a classification algorithm using Spark MLlib.
2. Perform feature engineering using feature extraction, transformation, scaling, and feature selection techniques.

## Technology

- Apache Spark
- PySpark
- Spark MLlib
- Python
- Logistic Regression
- CSV dataset

## Project Structure

```text
Spark_MLlib_Classification_Feature_Engineering/
│
├── data/
│   └── customer_churn.csv
│
├── src/
│   └── spark_mllib_classification.py
│
├── requirements.txt
└── README.md
```

## Dataset

The sample dataset is a synthetic customer-churn dataset containing:

- customer_id
- age
- tenure_months
- monthly_charges
- support_calls
- usage_hours
- city
- plan
- churn

The target variable is `churn`:

- `0` = Customer retained
- `1` = Customer churned

## MLlib Feature Engineering

The project demonstrates:

### 1. Feature Extraction
Categorical features such as `city` and `plan` are converted into numerical representations using:

- StringIndexer
- OneHotEncoder

### 2. Feature Transformation
`VectorAssembler` combines numerical and categorical features into a single feature vector.

### 3. Feature Scaling
`StandardScaler` standardizes the assembled feature vector.

### 4. Feature Selection
`ChiSqSelector` selects the top 8 features based on the chi-square test.

### 5. Classification
Logistic Regression is trained using the selected features.

## Pipeline

```text
CSV Dataset
     |
     v
Load Data into Spark
     |
     v
StringIndexer
     |
     v
OneHotEncoder
     |
     v
VectorAssembler
     |
     v
StandardScaler
     |
     v
ChiSqSelector
     |
     v
Logistic Regression
     |
     v
Predictions
     |
     v
Accuracy / Precision / Recall / F1
```

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Make sure Java is installed and available in your PATH.

## Run the Project

From the project root directory:

```bash
spark-submit src/spark_mllib_classification.py
```

Alternatively:

```bash
python src/spark_mllib_classification.py
```

if PySpark is correctly configured.

## Expected Output

The program displays:

- Dataset schema
- Sample records
- Training and testing record counts
- Predictions
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Example:

```text
=== Model Evaluation ===
Accuracy  : 0.xxxx
Precision : 0.xxxx
Recall    : 0.xxxx
F1-Score  : 0.xxxx
```

Actual values may vary slightly depending on the Spark/PySpark environment.

## GitHub Upload

Create a GitHub repository and upload all files while maintaining this structure:

```text
Spark_MLlib_Classification_Feature_Engineering
├── data
│   └── customer_churn.csv
├── src
│   └── spark_mllib_classification.py
├── requirements.txt
└── README.md
```

## Learning Outcomes

After completing this practical, students should be able to:

- Load CSV data using Spark DataFrame API.
- Apply Spark MLlib classification.
- Convert categorical data into numerical features.
- Assemble multiple features into a feature vector.
- Apply feature scaling.
- Perform feature selection.
- Build an end-to-end ML pipeline.
- Evaluate a classification model using standard metrics.
