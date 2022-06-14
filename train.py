import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from preprocessing import convert_data, engineer_features, select_features
from preprocessing import destroy_imbalance

np.random.seed(59)


def read_data():
    path = 'data\\train.csv'
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = convert_data(df)
    df = engineer_features(df)
    df = select_features(df)
    return df

def train_model(X, y):
    '''
        After tuning with hyperopt^
        Best model - XGBClassifier (among Catboost and LGBM)
        Best tuned and non-tuned parameters are below
    '''
    params = {
        'alpha': 0.003,
        'colsample_bytree': 0.8,
        'gamma': 5,
        'lambda': 0.003,
        'learning_rate': 0.007,
        'max_depth': 10,
        'min_child_weight': 0.96,
        'subsample': 0.53,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 59
    }

    train = xgb.DMatrix(X, y)
    model = xgb.train(params, train)

    return model

def save_model(model):
    model.save_model('models\\model_xgb.json')

def load_model():
    model = xgb.XGBClassifier()
    model.load_model('models\\model_xgb.json')
    return model


if __name__ == '__main__':
    # Read and preprocess the data
    df = read_data()
    df = preprocess_data(df)

    # Split features and target
    X, y = df.drop('churn', axis=1), df.churn

    # Deal with imbalance
    X, y = destroy_imbalance(X, y)

    # Fit the model
    model = train_model(X, y)

    # Save the model, so we can get it for the inference
    save_model(model)

    # Load the model (checking)
    model = load_model()
    preds = model.predict(X)
    print(classification_report(y, preds))
