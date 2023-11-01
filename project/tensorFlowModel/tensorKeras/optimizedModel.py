#all the imports needed
# pip install -U keras-tuner
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python import keras 
from keras import layers
from keras.layers import *
import keras_tuner as kt
from tensorflow.python.keras.layers import Dense
import keras.api._v2.keras as keras
import numpy as np

#scale values:
SUB_VALUE = -0.000413
DIVIDE_VALUE = 0.0000009825

dftrain = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/tf_train_df.csv')
dftest = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/tf_test_df.csv')

#sets the y_train and y-val to the labels of the output columns
input_train = dftrain.drop(['Median_House_Value'], axis = 1).values
y_train = dftrain.pop('Median_House_Value')
input_test = dftest.drop(['Median_House_Value'], axis = 1).values
y_test = dftest.pop('Median_House_Value')

#NOTE: OPTIMIZING THE MODEL
#CREATING THE MODEL
class model_builder(kt.HyperModel):
    def build(self, hp):    
        model = keras.models.Sequential()
        model.add(layers.Flatten())
        hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
        hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=1000, step=100)
        hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=1000, step=100)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.add(Dense(hp_layer_1, input_dim=9, activation=hp_activation))
        model.add(Dense(hp_layer_2, activation=hp_activation))
        model.add(Dense(1, activation=hp_activation))

        #loss and optimizer functions
        loss = keras.losses.MeanAbsoluteError() #NOTE: you can also used MeanSquaredError
        optim = keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate)

        # #compiling the model
        model.compile(optimizer=optim,loss=loss, metrics=['accuracy'])
        return model

my_model = model_builder()
#using hyper parameter tuning
tuner = kt.Hyperband(my_model, objective='val_accuracy', max_epochs = 50, factor=3, directory='hyperTuning', project_name='Test_Run')
tuner.search_space_summary()
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) # was 3
tuner.search(input_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

#returning the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps)

model = my_model.build(best_hps)

history = model.fit(input_train, y_train, epochs = 50, validation_split = 0.2, callbacks=[stop_early])

#printing out the results
test_error_rate = model.evaluate(input_test, y_test, verbose=1)
print(test_error_rate)

#getting in predict data
pred1 = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/predict/pred1.csv')
pred1 = pd.DataFrame(pred1)
pred2 = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/predict/pred2.csv')
pred2 = pd.DataFrame(pred2)
#saving the actual values
value1 = (pred1.iloc[0][0] + SUB_VALUE) / DIVIDE_VALUE
value2 = (pred2.iloc[0][0] + SUB_VALUE) / DIVIDE_VALUE
#droping the output column
pred1 = pred1.drop(["Median_House_Value"], axis = 1)
pred2 = pred2.drop(["Median_House_Value"], axis = 1)
#predicting
prediction1 = model.predict(pred1)[0][0]
prediction2 = model.predict(pred2)[0][0]
#comparing the actual and predicted result
print(f"\n\n\nTest Run with Nodes | Epochs")
print(f'Prediction 1:\n\tActualValue: {value1:,.2f}\n\tPrediction: {prediction1:,.2f}')
print(f'Prediction 2:\n\tActualValue: {value2:,.2f}\n\tPrediction: {prediction2:,.2f}')

#plotting the history
# plot_loss(history)
#plotting the predictions vs actual values
house = [1, 2, 3, 4, 5]
actual_values = [value1, value2]
predictions = [prediction1, prediction2]

X_axis = np.arange(len(house)) 

print(type(value1))
plt.bar(X_axis - 0.2, actual_values, 0.4, label = 'Actual Values') 
plt.bar(X_axis + 0.2, predictions, 0.4, label = 'Predictions') 

plt.xticks(X_axis, house) 
plt.xlabel("Houses") 
plt.ylabel("Value (in Millions)") 
plt.title(f"Test") 
plt.legend() 

plt.show() 