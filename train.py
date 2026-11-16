import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

from xgboost import XGBClassifier


def load_data():
    df= pd.read_csv('train.csv')
    # filling null values
    df['Age']= df['Age'].fillna(df['Age'].median())
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df


def train_model(df):

    categorical=['Name', 'Sex','Ticket','Cabin', 'Embarked']
    numerical=['PassengerId','Pclass','Age', 'SibSp', 'Parch', 'Fare']


    # spliting the data

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train.Survived.values
    y_test = df_test.Survived.values

    del df_full_train['Survived']
    del df_test['Survived']

    # making a data pipeline 
    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        XGBClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=9,
            min_child_weight=4,
            subsample=0.8,
            colsample_bytree=0.96,
            reg_lambda=0.16,
            gamma=2.29,
            objective='binary:logistic',
            n_jobs=8,
            random_state=1,
        )
    )

    # Train: the pipeline takes the list of dicts directly
    dicts_full_train = df_full_train.to_dict(orient='records')
    pipeline.fit(dicts_full_train, y_full_train)

    return pipeline


def save_model(filename, model):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)
    print(f'model saved to {filename}')


df=load_data()
pipeline=train_model(df)
save_model('model.bin',pipeline)
