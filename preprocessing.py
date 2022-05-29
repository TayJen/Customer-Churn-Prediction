from sklearn.preprocessing import StandardScaler, Normalizer
import pandas as pd
import numpy as np


def convert_data(df):
    # Remove outliers (less than 10 vmail messages and more than 50)
    mask_vm_true = (df.voice_mail_plan == 'yes')
    mask_vm_outlier = ((df.number_vmail_messages > 50) | (df.number_vmail_messages < 10))
    df = df.drop(df[mask_vm_true & mask_vm_outlier].index)

    # More or equal to 5 service calls for non-churn customers
    mask_ch_true = (df.churn == 'no')
    mask_ch_outlier = (df.number_customer_service_calls >= 5)
    df = df.drop(df[mask_ch_true & mask_ch_outlier].index)

    # Map categorical columns to numbers
    bin_columns = ['international_plan', 'voice_mail_plan', 'churn']
    for col in bin_columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Sum up total columns, instead of different times of a day
    feature_types = ['minutes', 'calls', 'charge']

    for feature in feature_types:
        df['total_' + feature] = df['total_day_' + feature] + df['total_eve_' + feature] + df['total_night_' + feature]
        df.drop(['total_day_' + feature, 'total_eve_' + feature, 'total_night_' + feature], axis=1, inplace=True)

    # Find Top 5 states by churn
    top5_states = df[df['churn'] == 1]['state'] \
        .value_counts() \
        .sort_values(ascending=False)[:5] \
        .index.values

    # Keep Top 5 and label others as Other
    df['state'] = df['state'].apply(lambda x: x if x in top5_states else 'other')
    df = pd.get_dummies(df, columns=['area_code', 'state'])

    return df


def engineer_features(df):
    # Generate features of average month charge/minutes/calls.
    features = ['minutes', 'calls', 'charge']
    for feature in features:
        df['avg_mt_' + feature] = df['total_' + feature] / df['account_length']

    # Same for average call, how much does it cost, how long is it (in minutes)
    df['avg_call_charge'] = df['total_charge'] / df['total_calls']
    df['avg_intl_call_charge'] = df['total_intl_charge'] / df['total_intl_calls']

    df['avg_call_minutes'] = df['total_minutes'] / df['total_calls']
    df['avg_intl_call_minutes'] = df['total_intl_minutes'] / df['total_intl_calls']

    # New feature as intersection of international plan and voice_mail plan
    df['both_plans'] = df['international_plan'] & df['voice_mail_plan']

    # Fill NaNs
    df.fillna(0, inplace=True)

    return df


def select_features(df):
    # Drop strongly correlated features (threshold = 0.8)
    df.drop(['total_intl_charge', 'total_charge', 'avg_mt_minutes', 'avg_mt_calls',
             'avg_call_minutes', 'avg_intl_call_minutes', 'voice_mail_plan'], axis=1, inplace=True)

    # Drop `state` as this feature has low impact
    df.drop(['state_WV', 'state_NJ', 'state_MN',
             'state_MD', 'state_TX'], axis=1, inplace=True)

    return df
