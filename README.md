# Binary income classification with Keras

## Dataset
The dataset is the Cencus Income Data Set. It contains continous and categorical attributes. 
Each row corresponds to a person and it contains their income range and personal characteristics.
You can download the data [here][https://archive.ics.uci.edu/ml/datasets/census+income].

## Model
The model is a multi-input fully connected neural network built with Functional API of Keras.
It combines the continous variables and the embedded categorical variables.
It uses the logistic sigmoid function as activation function, binary cross entropy as loss function, and
Adam as optimizer.

## How to run
`conda env create -f environment.yml`

`conda activate census`

`python src/services/main.py --filepath 'yourdatapath'`

You will see the attribute value counts, details about the missing data, the model summary and then
the loss and accuracy per epoch. The training will run for 20 epochs. At the end, the accuracy, precision and recall will be printed.
