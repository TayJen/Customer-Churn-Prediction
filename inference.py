import csv
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import xgboost as xgb
from final_train import load_model
from preprocessing import test_preprocessing

np.random.seed(59)


def read_data():
    path = 'data\\test.csv'
    df = pd.read_csv(path)
    return df

def make_save_prediction(model, data):
    ids = data['id'].astype(np.int32)
    X = data.drop('id', axis=1)

    preds = model.predict(X)
    submission_df = pd.DataFrame({
        'id': ids,
        'churn': preds
    })

    submission_df['churn'] = submission_df['churn'].map({1: 'yes', 0: 'no'})
    submission_df.to_csv('submissions\\first.csv', index=False)

if __name__ == '__main__':
    df = read_data()
    df = test_preprocessing(df)
    model = load_model()
    make_save_prediction(model, df)
