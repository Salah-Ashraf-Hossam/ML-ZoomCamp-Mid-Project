# ML-ZoomCamp-Mid-Project
# Titanic Survival Prediction

## Problem Description

On April 15, 1912, the RMS Titanic sank after hitting an iceberg, and more than 1,500 of the roughly 2,200 people on board died. Survival was not random. Factors such as age, sex, passenger class, and family size appear to have influenced who made it onto a lifeboat.

This project uses the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic) to predict whether a passenger survived. This is a **supervised binary classification** problem:

- **Target:** `Survived` (0 = did not survive, 1 = survived)
- **Features:** passenger class (`Pclass`), sex, age, number of siblings/spouses aboard (`SibSp`), number of parents/children aboard (`Parch`), ticket fare, and port of embarkation
- **Evaluation metric:** [ AUC ]

The training set has 891 labeled passengers and contains missing values, notably in `Age` and `Cabin`, so data cleaning is part of the task.

## How the Model Will Be Used

The final model takes a passenger's recorded details and outputs the probability that they survived. It is intended for:

- **Coursework:** demonstrating an end-to-end machine learning workflow, from data preparation to deployment.
- **Kaggle submission:** generating predictions on the unlabeled test set (`test.csv`).
- **Interpretation:** identifying which features most influenced survival.
- **Serving predictions:** the trained model is deployed as a prediction service that other applications can call.

**Out of scope:** This is a historical dataset and an educational project. The model should not be used to make real-world decisions about people. It reflects the biases of 1912 (for example, priority given by sex and class) and is not meant to generalize to other populations or events.

## Project Steps

1. **Data preparation:** handled missing values, encoded categorical features, and cleaned the data.
2. **Exploratory data analysis (EDA):** explored distributions and relationships between features and survival.
3. **Feature importance analysis:** used **mutual information** and **correlation** to measure how strongly each feature relates to survival.
4. **Model training:** trained four models: Logistic Regression, Decision Tree, Random Forest, and XGBoost.
5. **Hyperparameter tuning:** tuned the parameters of each model to improve validation performance.
6. **Model selection:** compared the tuned models and chose **XGBoost** as the final model.
7. **Deployment:** deployed the final model as a prediction service, containerized with Docker.

### Model Comparison

| Model | [AUC] |
|---|---|
| Logistic Regression | 0.838 |
| Decision Tree | 0.865 |
| Random Forest | 0.873 |
| **XGBoost (selected)** | 0.882 |

XGBoost was chosen because [ it achieved the highest validation AUC score of the four models].

## Repository Contents

| File | Description |
|---|---|
| `eda_and_model_selection.ipynb` | Data preparation, EDA, feature importance analysis, model training, tuning, and selection |
| `final_train.py` | Training script exported from the notebook |
| `predict.py` | Prediction service exported from the notebook |
| `pyproject.toml` / `uv.lock` | Dependency and environment files |
| `Dockerfile` | Builds the container for the prediction service |
|  `demo.mp4` | video of the deployed model |

## Running the Project

### Prerequisites
- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker](https://docs.docker.com/get-docker/) (installed and running)

### 1. Clone the repo
```bash
git clone https://github.com/Salah-Ashraf-Hossam/ML-ZoomCamp-Mid-Project/tree/main.git
cd <ML-ZoomCamp-Mid-Project>
```

### 2. Install dependencies
```bash
uv sync
```
This reads `pyproject.toml` and `uv.lock` and creates a `.venv` with the exact versions used.

### 3. Train the model
```bash
uv run python final_train.py
```
This saves the trained model to `model.bin`.

### 4. Build and run the prediction service
```bash
docker build -t titanic-service .
docker run -p 9696:9696 titanic-service
```

### 5. Test it
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 3, "Sex": "male", "Age": 22, "Fare": 7.25}'
```
