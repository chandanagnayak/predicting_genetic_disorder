# Inherited Genetic Disorder Risk Prediction

## Overview

This project aims to predict the risk of offspring inheriting a genetic disorder based on a dataset of patient information. It utilizes two powerful machine learning algorithms: XGBoost and Random Forest. The predictions from these models are then combined using a simple averaging ensemble method to potentially improve the overall predictive performance.

## Dataset

The project uses a tabular dataset (`genetic_disorder_data.csv`) containing various features related to patients and their medical history, including:

* Demographic information (e.g., Age, Smoker, Mother's age)
* Genetic factors (e.g., Infected from father, Maternal genes, Paternal genes)
* Maternal health during pregnancy (e.g., Folic acid, Serious maternal illness, Radiation exposure, Substance abuse)
* Pregnancy history (e.g., History of anomalies, Number of previous abortions/stillbirths)
* Birth details (e.g., Birth defects)
* Medical test results (e.g., Blood cell counts, Blood test result)
* Symptoms
* The target variable: `Risk` (indicating the presence or absence of a higher risk of inherited genetic disorder in offspring).

**Note:** The `genetic_disorder_data.csv` file used in this project is a simulated dataset for demonstration purposes. Real-world genetic disorder prediction would require a more comprehensive and clinically validated dataset.

## Project Structure

The project consists of a single Python script (`your_script_name.py`, assuming you saved the provided code in a file with this name).

## Libraries Used

* **pandas:** For data manipulation and reading the CSV file.
* **numpy:** For numerical operations.
* **scikit-learn:** For machine learning tasks such as data splitting (`train_test_split`), classification models (`RandomForestClassifier`), and evaluation metrics (`accuracy_score`, `classification_report`, `confusion_matrix`).
* **xgboost:** For the XGBoost classification model (`XGBClassifier`).
* **matplotlib:** For creating static, interactive, and animated visualizations in Python (used for plotting the confusion matrices).
* **seaborn:** For making statistical graphics in Python (used for creating aesthetically pleasing confusion matrix heatmaps).

## Workflow

1.  **Data Loading:** The script loads the dataset from the `genetic_disorder_data.csv` file into a pandas DataFrame.
2.  **Feature Engineering (Minimal):** Categorical features in the dataset are converted into a numerical format using one-hot encoding (`pd.get_dummies`).
3.  **Data Splitting:** The dataset is split into training (80%) and testing (20%) sets using `train_test_split`. Stratification is used to ensure a balanced representation of the `Risk` classes in both sets.
4.  **Model Training:**
    * An XGBoost classifier is initialized with a random state for reproducibility and trained on the training data.
    * A Random Forest classifier is initialized with a random state and trained on the training data.
5.  **Prediction:** Both trained models are used to make predictions on the unseen test data. Probability predictions are obtained for the positive class (`Risk=1`).
6.  **Ensemble Prediction:** The probability predictions from the XGBoost and Random Forest models are combined by simple averaging. A threshold of 0.5 is applied to these combined probabilities to obtain the final binary predictions (Risk or No Risk).
7.  **Model Evaluation:** The performance of both individual models and the combined model is evaluated using:
    * Accuracy score
    * Classification report (precision, recall, F1-score, support)
    * Confusion matrix (visualized as a heatmap)

## How to Run

1.  **Save the Dataset:** Ensure that the `genetic_disorder_data.csv` file is in the same directory as your Python script.
2.  **Install Libraries:** If you haven't already, install the necessary libraries using pip:
    ```bash
    pip install pandas scikit-learn xgboost matplotlib seaborn
    ```
3.  **Run the Script:** Execute the Python script from your terminal:
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file).

## Results

The script will output the accuracy and classification reports for both the XGBoost and Random Forest models individually, as well as for the combined ensemble model. It will also display confusion matrices as visual plots, providing insights into the models' performance in classifying the risk of inherited genetic disorders.

## Further Improvements

* **Feature Engineering:** Explore creating new features from the existing data that might be more informative for the models.
* **Hyperparameter Tuning:** Optimize the hyperparameters of the XGBoost and Random Forest models using techniques like GridSearchCV or RandomizedSearchCV to potentially improve performance.
* **Feature Selection:** Investigate and select the most relevant features to reduce dimensionality and potentially enhance model generalization.
* **More Sophisticated Ensembling:** Experiment with more advanced ensembling techniques like stacking or weighted averaging.
* **Cross-Validation:** Implement cross-validation during training for a more robust evaluation of the model's performance.
* **External Data Integration:** If available, incorporate external datasets that might provide additional relevant information.
* **Explainable AI (XAI):** Explore techniques to make the model's predictions more interpretable, especially in a sensitive domain like genetic disorder prediction.
