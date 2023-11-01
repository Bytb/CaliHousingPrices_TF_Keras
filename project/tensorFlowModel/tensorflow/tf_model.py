#make import statements
import tensorflow as tf
import pandas as pd
import numpy as np

#creating the dataframes for Y_test and Y_train that do not have the output
df_train = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/tf_train_df.csv')
df_test = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/tf_test_df.csv')
Y_train = df_train.pop('Median_House_Value')
Y_test = df_test.pop('Median_House_Value')

#create input function
def make_input_fn(data_df, label_df, num_epochs = 10, shuffle = True, batch_size = 32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

#using the input functions to create model (this creates nodes and epochs)
train_input_fn = make_input_fn(df_train, Y_train) #NOTE: default num_epochs value is 10
test_input_fn = make_input_fn(df_test, Y_test, num_epochs = 1)

#creating the estimator for linear reg
linear_est = tf.estimator.LinearClassifier(feature_columns = df_train.columns.values.any()) #NOTE: in the training video, the syntax is feature_columns = feature_columns

#TRAINING THE MODEL
linear_est.train(train_input_fn) #NOTE: THIS IS THE WHERE THE ERROR IS BEING THROWN
#the result data
result = linear_est.evaluate(test_input_fn)
#prints the value for how accurate the model is
print(result['accuracy'])