from keras.layers import Dense, Dropout, Embedding, Input, Reshape, Concatenate
from keras.models import Model
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score

import tensorflow as tf

def train(model: Model, training_set: list, training_labels: list) -> Model:
  model.fit(x=training_set, y=training_labels, epochs=20, batch_size=32)
  return model

def evaluate(model: Model, test_set: list, test_labels: list):
  predictions = model.predict(x=test_set)
  print(predictions)
  model.evaluate(x=test_set, y=test_labels)
  print(f'Precision: {precision_score(test_labels, predictions.round())}')
  print(f'Recall: {recall_score(test_labels, predictions.round())}')
  
