from keras.layers import Dense, Dropout, Embedding, Input, Reshape, Concatenate
from keras.models import Model
import tensorflow as tf

def create_model(attribute_embsizes: dict, attribute_sizes: dict, continuous_attributes: list) -> Model:
  """
  Create embeddings for categorical attributes, compile model layers
  """
  input_layers = []
  concatanate_layes = []
  for attribute in attribute_embsizes.keys():
    categorical_input = Input((1,), name=attribute)
    input_layers.append(categorical_input)
    categorical_input = Embedding(attribute_sizes[attribute]+1, attribute_embsizes[attribute], input_length=1)(categorical_input)
    categorical_input = Reshape((attribute_embsizes[attribute],))(categorical_input)
    concatanate_layes.append(categorical_input)
  
  continuous_input = Input((len(continuous_attributes),), name='continuous_attributes')
  input_layers.append(continuous_input)
  concatanate_layes.append(continuous_input)

  concatanate = Concatenate()(concatanate_layes)
  dense = Dense(100, activation= 'relu')(concatanate)
  output = Dense(1, activation='sigmoid')(dense)
  model = Model(input_layers, output) # number_parameters = (emb_dim * (input_dim+1))
  model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])
  model.summary()
  
  return model