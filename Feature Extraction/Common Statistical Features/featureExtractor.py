#Ref: https://www.kaggle.com/code/tyrionlannisterlzy/xgboost-dnn-ensemble-lb-0-980

import numpy as np
import pandas as pd

train = pd.read_csv('dataset/train.csv')
labels = pd.read_csv('dataset/train_labels.csv')
test = pd.read_csv('dataset/test.csv')

def aggregated_features(df, aggregation_cols=['sequence'], prefix=''):
    agg_strategy = {'sensor_00': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_01': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_02': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_03': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_04': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_05': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_06': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_07': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_08': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_09': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_10': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_11': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_12': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    }
    group = df.groupby(aggregation_cols).aggregate(agg_strategy)
    group.columns = ['_'.join(col).strip() for col in group.columns]
    group.columns = [str(prefix) + str(col) for col in group.columns]
    group.reset_index(inplace=True)

    temp = (df.groupby(aggregation_cols).size().reset_index(name=str(prefix) + 'size'))
    group = pd.merge(temp, group, how='left', on=aggregation_cols, )
    return group


train_merge_data = aggregated_features(train, aggregation_cols = ['sequence', 'subject'])
test_merge_data = aggregated_features(test, aggregation_cols = ['sequence', 'subject'])

train_subjects_merge_data = aggregated_features(train, aggregation_cols = ['subject'], prefix = 'subject_')
test_subjects_merge_data = aggregated_features(test, aggregation_cols = ['subject'], prefix = 'subject_')
