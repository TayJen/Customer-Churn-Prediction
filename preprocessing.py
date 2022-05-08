import pandas as pd
import numpy as np


def data_preprocessing(df):
    '''
        This function transforms df:
            - Yes/no features such as `international_plan`, `voice_mail_plan` and `churn` are transformed into 1/0
            - day/eve/night features are summed up into one new `total` feature
            - in `state` feature only top 5 churn states are saved, other are labeled as `other`
            - one-hot-encoding for `state` and `area_code`
        Input: clean DataFrame
        Output: preprocessed DataFrame
    '''
    
    bin_columns = ['international_plan', 'voice_mail_plan', 'churn']
    for col in bin_columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})
        
    feature_types = ['minutes', 'calls', 'charge']

    for feature in feature_types:
        df['total_' + feature] = df['total_day_' + feature] + df['total_eve_' + feature] + df['total_night_' + feature]
        df.drop(['total_day_' + feature, 'total_eve_' + feature, 'total_night_' + feature], axis=1, inplace=True)
        
    top5_states = df[df['churn'] == 1]['state']\
                    .value_counts()\
                    .sort_values(ascending=False)[:5]\
                    .index.values
                    
    df['state'] = df['state'].apply(lambda x: x if x in top5_states else 'other')
    df = pd.get_dummies(df, columns=['area_code', 'state'])
    
    return df