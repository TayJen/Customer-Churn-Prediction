from sklearn.preprocessing import StandardScaler, Normalizer
import pandas as pd
import numpy as np


class DataPreprocessingPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.norm = Normalizer()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.norm.fit(X)

        return self

    def transform(self, X, y=None):
        X = self.scaler.transform(X)
        X = self.norm.transform(X)

        return X

def convert_data(df):
    bin_columns = ['international_plan', 'voice_mail_plan', 'churn']
    for col in bin_columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    feature_types = ['minutes', 'calls', 'charge']

    for feature in feature_types:
        df['total_' + feature] = df['total_day_' + feature] + df['total_eve_' + feature] + df['total_night_' + feature]
        df.drop(['total_day_' + feature, 'total_eve_' + feature, 'total_night_' + feature], axis=1, inplace=True)

    top5_states = df[df['churn'] == 1]['state'] \
        .value_counts() \
        .sort_values(ascending=False)[:5] \
        .index.values

    df['state'] = df['state'].apply(lambda x: x if x in top5_states else 'other')
    df = pd.get_dummies(df, columns=['area_code', 'state'])

    return df
