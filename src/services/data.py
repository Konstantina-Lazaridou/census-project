import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder


def clean_data(data: pd.DataFrame, label: str = 'income') -> pd.DataFrame:
  """
  Define data types, sanitize class values, replace missing values with average value
  """
  data.astype({'age': 'float32', 'workclass': 'string', 'fnlwgt': 'float32', 'education': 'string', 'education_num': 'float32',
  'marital_status': 'string', 'occupation': 'string', 'relationship': 'string', 'race': 'string',
  'sex': 'string', 'capital_gain': 'float32', 'capital_loss': 'float32', 'hours_per_week': 'float32',
  'native_country': 'string', 'income': 'string'})
  data.replace(' ?', '?', inplace=True)
  data[label] = data[label].str.replace(' ', '')
  data[label] = data[label].str.replace('.', '')
  for column in data:
    if data[column].dtype != np.int64 and data[column].dtype != np.float64:
      missing_values = len(data[data[column] == '?'])
      if missing_values > 0:
        values = [value for
        items, value in (data.groupby(column)[column].value_counts().iteritems())]
        average_value = int(sum(values)/len(values))
        value_counts_dict = data.groupby(column)[column].value_counts()
        missing_value = min(value_counts_dict.items(), key=lambda value: abs(value[1] - average_value))[0][0]
        print(f'Replacing missing values in {column} with {missing_value}')
        data.replace(to_replace={column:'?'}, value=missing_value)
  return data


def split_data(data: pd.DataFrame, label: str = 'income') -> Tuple[np.array, np.array, np.array, np.array]:
  """
  Subsample large data class, create training and test sets
  """
  # TODO: gradually introduce rest of the data in the biggest class
  sample_size = 11687   # half of the largest class
  # balanced_data = data
  balanced_data = data.groupby(label).apply(lambda x: x.sample(n=sample_size)).reset_index(drop = True)
  Y = balanced_data[[label]]
  X = balanced_data.drop([label], axis=1)
  print(f'Split data. Attributes shape: {X.shape}\nClasses shape: {Y.shape}')
  # TODO: use cross validation instead
  x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=.8, random_state=1)
  return x_train, x_test, y_train, y_test


def represent_data(x_train: pd.DataFrame, x_test: pd.DataFrame, continuous_attributes: list, categorical_attributes: list) ->Tuple[list, list, dict, dict]:
  """
  Create data as lists, create embedding configuration
  """
  x_train_as_list = []
  x_test_as_list = []
  for attribute in categorical_attributes:
    attribute_encoder = LabelEncoder()
    all_values= set(x_train[attribute].values) | set(x_test[attribute].values)
    attribute_encoder.fit(np.array(list(all_values)))
    x_train_as_list.append(attribute_encoder.transform(x_train[attribute].values))
    x_test_as_list.append(attribute_encoder.transform(x_test[attribute].values))
  x_train_as_list.append(x_train[continuous_attributes].values)
  x_test_as_list.append(x_test[continuous_attributes].values)
  attribute_sizes = {}
  attribute_embsizes = {}
  for attribute in categorical_attributes:
    attribute_sizes[attribute] = x_train[attribute].nunique()
    attribute_embsizes[attribute] = min(50, attribute_sizes[attribute]//2+1)
  return x_train_as_list, x_test_as_list, attribute_embsizes, attribute_sizes