# ML-ZoomCamp-Mid-Project
## Problem Description

On April 15, 1912, the RMS Titanic sank after hitting an iceberg, and more than 1,500 of the roughly 2,200 people on board died. Survival was not random. Factors such as age, sex, passenger class, and family size appear to have influenced who made it onto a lifeboat.

This project uses the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic) to predict whether a passenger survived. This is a **supervised binary classification** problem:

- **Target:** `Survived` (0 = did not survive, 1 = survived)
- **Features:** passenger class (`Pclass`), sex, age, number of siblings/spouses aboard (`SibSp`), number of parents/children aboard (`Parch`), ticket fare, port of embarkation, and engineered features such as family size and title extracted from the name
- **Evaluation metric:** [accuracy / AUC / F1: pick the one you used]

The dataset is small (891 labeled passengers in the training set) and has missing values, notably in `Age` and `Cabin`. Part of the work is cleaning the data and engineering features.

## How the Model Will Be Used

The model takes a passenger's recorded details and outputs the probability that they survived. It is intended for:

- **Learning and benchmarking:** demonstrating an end-to-end machine learning workflow (data cleaning, feature engineering, training, validation, and tuning) on a well-known dataset.
- **Kaggle submission:** generating predictions on the unlabeled test set (`test.csv`) and submitting them to the Titanic competition leaderboard.
- **Exploratory insight:** identifying which features most strongly influenced survival, for example through feature importance.

**Out of scope:** This is a historical dataset and an educational project. The model should not be used to make real-world decisions about people. It reflects the biases of 1912 (for example, priority given by sex and class) and is not meant to generalize to other emergencies or populations.
