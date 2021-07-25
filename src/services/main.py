import pandas as pd
import argparse
from data import split_data, clean_data, represent_data
from model import create_model
from train_test import train, evaluate
from sklearn.preprocessing import LabelEncoder


parser = argparse.ArgumentParser(description='Give input file path')
parser.add_argument('--filepath', required=True, type=str)
args = parser.parse_args()
data_file = args.filepath
with open(data_file, "r") as fp:
  data = pd.read_csv(fp)
  # data preparation
  print(f'Attribute distribution\n: {data.nunique()}')
  data = clean_data(data=data, label='income')
  x_train, x_test, y_train, y_test = split_data(data=data, label='income')
  continuous_attributes = ['age', 'fnlwgt', 'education_num', 'capital_loss', 'capital_gain', 'hours_per_week']
  categorical_attributes = list(x_train.drop(continuous_attributes, axis=1).columns)
  x_train_as_list, x_test_as_list, attribute_embsizes, attribute_sizes = represent_data(x_train, x_test, continuous_attributes, categorical_attributes)
  # model input preparation
  y_train_as_list = []
  y_train_as_list.append(y_train['income'].values)
  y_test_as_list = []
  y_test_as_list.append(y_test['income'].values)
  label_encoder = LabelEncoder()
  label_encoder.fit(['<=50K', '>50K'])
  y_train_as_list = label_encoder.transform(y_train_as_list[0])
  y_test_as_list = label_encoder.transform(y_test_as_list[0])
  print(f'Negative and positive labels: {label_encoder.classes_}')
  # learn from data
  model = create_model(attribute_embsizes, attribute_sizes, continuous_attributes)
  model = train(model=model, training_set=x_train_as_list, training_labels=y_train_as_list)
  evaluate(model=model, test_set=x_test_as_list, test_labels=y_test_as_list)
  



