#Step 1: import modules and functions from libraries
from keras.models import Sequential
from keras.layers import *
import pandas as pd

#import the training data
training_data_df = pd.read_csv("/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/training_data_scaled.csv")

#STEP 2: create the input and output features of the TRAINING data (we will do the testing data later)
input_train = training_data_df.drop(['Median_House_Value'], axis = 1).values
output_train = training_data_df[['Median_House_Value']].values

#STEP 3: create the actual keras model
model = Sequential()
#add in the layers
#NOTE: since this is first layer, you have to state how many input nodes there will be 
#NOTE: there NEEDS to be 9 inputs
model.add(Dense(100, input_dim = 9, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
#NOTE: you have to make the last one "linear"
model.add(Dense(1, activation = 'linear'))
#NOTE: you have to compile the model at the end and put in the optimizer and the loss function
model.compile(optimizer = 'adam', loss = 'MSE')

#STEP 4: train the data
model.fit(
    input_train,
    output_train, 
    epochs = 50,
    shuffle = True, 
    verbose = 2
)

#STEP 5: add in the test data
test_data_df = pd.read_csv("/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/test_data_scaled.csv")
input_test = test_data_df.drop(['Median_House_Value'], axis = 1).values
output_test = test_data_df[['Median_House_Value']].values

test_error_rate = model.evaluate(input_test, output_test, verbose=0)
print(f"The test error rate is {test_error_rate}")

#STEP 6: save trained model to disk
#NOTE: the .h5 extension is crucial
model.save("trained_model.h5")