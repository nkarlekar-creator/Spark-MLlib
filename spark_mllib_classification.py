"""
Spark MLlib Classification + Feature Engineering
Dataset: data/customer_churn.csv

Tasks covered:
1. Load dataset into Spark.
2. Perform classification using Spark MLlib.
3. Feature extraction/transformation using StringIndexer + OneHotEncoder.
4. Feature vector assembly using VectorAssembler.
5. Feature scaling using StandardScaler.
6. Feature selection using ChiSqSelector.
7. Train Logistic Regression model.
8. Evaluate accuracy, precision, recall, and F1-score.
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
    ChiSqSelector
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def main():
    spark = (
        SparkSession.builder
        .appName("CustomerChurnMLlibClassification")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # 1. Load dataset
    data_path = "data/customer_churn.csv"

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(data_path)
    )

    print("\n=== Dataset Schema ===")
    df.printSchema()

    print("\n=== Sample Records ===")
    df.show(10, truncate=False)

    print("\n=== Dataset Count ===")
    print(df.count())

    # 2. Feature columns
    numeric_features = [
        "age",
        "tenure_months",
        "monthly_charges",
        "support_calls",
        "usage_hours"
    ]

    categorical_features = ["city", "plan"]

    # 3. Convert categorical columns to numeric indices
    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_index",
            handleInvalid="keep"
        )
        for col in categorical_features
    ]

    # 4. One-hot encode categorical features
    encoder = OneHotEncoder(
        inputCols=[f"{c}_index" for c in categorical_features],
        outputCols=[f"{c}_ohe" for c in categorical_features]
    )

    # 5. Assemble all raw numeric + encoded categorical features
    assembler = VectorAssembler(
        inputCols=numeric_features + [f"{c}_ohe" for c in categorical_features],
        outputCol="raw_features"
    )

    # 6. Scale the assembled features
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="scaled_features",
        withMean=False,
        withStd=True
    )

    # 7. Feature selection using Chi-square selector
    selector = ChiSqSelector(
        numTopFeatures=8,
        featuresCol="scaled_features",
        labelCol="label",
        outputCol="selected_features"
    )

    # 8. Logistic Regression classifier
    lr = LogisticRegression(
        featuresCol="selected_features",
        labelCol="label",
        maxIter=50
    )

    # Create label from churn
    from pyspark.ml.feature import VectorAssembler as VA

    # We need label in a clean numeric column
    prepared_df = df.withColumnRenamed("churn", "label")

    # 9. Build ML pipeline
    pipeline = Pipeline(
        stages=indexers + [encoder, assembler, scaler, selector, lr]
    )

    # 10. Train/test split
    train_df, test_df = prepared_df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    print("\nTraining records:", train_df.count())
    print("Testing records :", test_df.count())

    # 11. Train model
    model = pipeline.fit(train_df)

    # 12. Make predictions
    predictions = model.transform(test_df)

    print("\n=== Predictions ===")
    predictions.select(
        "customer_id",
        "label",
        "prediction",
        "probability"
    ).show(20, truncate=False)

    # 13. Evaluation
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    accuracy = accuracy_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    # 14. Confusion matrix
    print("\n=== Confusion Matrix ===")
    predictions.groupBy("label", "prediction").count().orderBy(
        "label", "prediction"
    ).show()

    spark.stop()


if __name__ == "__main__":
    main()
