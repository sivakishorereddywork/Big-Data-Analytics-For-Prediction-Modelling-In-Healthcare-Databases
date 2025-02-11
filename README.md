# Big-Data-Analytics-For-Prediction-Modelling-In-Healthcare-Databases using PySpark

## Overview
This project implements a diabetes prediction model using Apache Spark's MLlib. The dataset is preprocessed, visualized, and used to train multiple machine learning models, including Random Forest, Decision Tree, and Gradient-Boosted Tree classifiers.

## Installation
Ensure you have Python installed, then install PySpark using:
```sh
pip install pyspark
```

## Libraries Used
```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```

## Creating Spark Session
```python
spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()
```

## Loading the Dataset
```python
dataset_path = "Healthcare_dataset.csv"
df = spark.read.csv(dataset_path, header=True, inferSchema=True)
df.printSchema()
```

## Data Preprocessing
### Checking for Null Values
```python
from pyspark.sql.functions import col
null_columns = [col(column) for column in df.columns if df.filter(col(column).isNull()).count() > 0]
if null_columns:
    print("Warning: There are null values in the dataset.")
else:
    print("No null values found.")
```
### Encoding Categorical Variables
```python
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
smoking_indexer = StringIndexer(inputCol="smoking_history", outputCol="smoking_index")
gender_encoder = OneHotEncoder(inputCol="gender_index", outputCol="gender_encoded")
smoking_encoder = OneHotEncoder(inputCol="smoking_index", outputCol="smoking_encoded")
```

### Feature Engineering
```python
feature_columns = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level",
                   "gender_encoded", "smoking_encoded"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
```

### Creating Preprocessing Pipeline
```python
preprocessing_pipeline = Pipeline(stages=[gender_indexer, smoking_indexer, gender_encoder, smoking_encoder, assembler, scaler])
preprocessed_data = preprocessing_pipeline.fit(df).transform(df)
```

## Data Visualization
### Pairplot
```python
pandas_df = preprocessed_data.select(feature_columns + ["diabetes"]).toPandas()
sns.pairplot(pandas_df, hue="diabetes")
plt.show()
```

### Correlation Matrix
```python
corr_matrix = pandas_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```

## Model Training
### Train-Test Split
```python
train_data, test_data = preprocessed_data.randomSplit([0.8, 0.2], seed=42)
```

### Defining Models
```python
rf = RandomForestClassifier(labelCol='diabetes', featuresCol='features', numTrees=60)
dt = DecisionTreeClassifier(labelCol='diabetes', featuresCol='features')
gbt = GBTClassifier(labelCol='diabetes', featuresCol='features', maxIter=10)
```

### Model Training
```python
rf_model = rf.fit(train_data)
dt_model = dt.fit(train_data)
gbt_model = gbt.fit(train_data)
```

## Model Evaluation
```python
rf_predictions = rf_model.transform(test_data)
dt_predictions = dt_model.transform(test_data)
gbt_predictions = gbt_model.transform(test_data)
```

### Accuracy Calculation
```python
evaluator = MulticlassClassificationEvaluator(labelCol='diabetes', metricName='accuracy')
rf_accuracy = evaluator.evaluate(rf_predictions)
dt_accuracy = evaluator.evaluate(dt_predictions)
gbt_accuracy = evaluator.evaluate(gbt_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Gradient-Boosted Tree Accuracy: {gbt_accuracy}")
```

## Results
- **Random Forest Accuracy:** `~97.3%`
- **Decision Tree Accuracy:** `~97.3%`
- **Gradient-Boosted Tree Accuracy:** `~97.3%`

## Conclusion
This project successfully implements a diabetes prediction system using PySpark. The models achieve high accuracy in classifying diabetes patients, and the exploratory data analysis provides meaningful insights into the dataset.

## Future Enhancements
- Implement additional feature selection techniques.
- Optimize hyperparameters using Grid Search.
- Deploy the model as a REST API.

## Author
**Siva Kishore Reddy**

## License
This project is licensed under the MIT License.


