#all the imports needed
import matplotlib.pyplot as plt
import pandas as pd
import keras 
import numpy as np
from keras import layers

#Creating node input
user_node = int(input("Enter node number: "))
#adding epoch input
user_epoch = int(input("Enter epoch number: "))

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

#CREATING THE MODEL
model = keras.models.Sequential([
    layers.Dense(user_node, activation='relu'),
    layers.Dense(user_node, activation='relu'),
    layers.Dense(1)
])

#loss and optimizer functions
loss = keras.losses.MeanAbsoluteError() #NOTE: you can also used MeanSquaredError
optim = keras.optimizers.legacy.Adam(learning_rate=0.05)

# #compiling the model
model.compile(optimizer=optim,loss=loss)

#training the model
history = model.fit(
    input_train,
    y_train, 
    epochs = user_epoch, 
    verbose = 1,
    # Calculate validation results on 20% of the training data
    validation_split=0.2
)

#plotting the model to see how it performs using matplotlib (NOTE: this can also be done in TensorBoard)
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.ylim([0,25])
    plt.xlabel(['Epoch'])
    plt.ylabel(['Error [MPG]'])
    plt.legend()
    plt.grid(True)
    plt.show()

#printing out the results
test_error_rate = model.evaluate(input_test, y_test, verbose=1)
print(f'\n{test_error_rate}\n')

#getting in predict data
pred1 = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/predict/pred1.csv')
pred1 = pd.DataFrame(pred1)
pred2 = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/predict/pred2.csv')
pred2 = pd.DataFrame(pred2)
pred3 = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/predict/pred3.csv')
pred3 = pd.DataFrame(pred3)
pred4 = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/predict/pred4.csv')
pred4 = pd.DataFrame(pred4)
pred5 = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/predict/pred5.csv')
pred5 = pd.DataFrame(pred5)

#saving the actual values
value1 = (pred1.iloc[0][0] + SUB_VALUE) / DIVIDE_VALUE
value2 = (pred2.iloc[0][0] + SUB_VALUE) / DIVIDE_VALUE
value3 = (pred3.iloc[0][0] + SUB_VALUE) / DIVIDE_VALUE
value4 = (pred4.iloc[0][0] + SUB_VALUE) / DIVIDE_VALUE
value5 = (pred5.iloc[0][0] + SUB_VALUE) / DIVIDE_VALUE

#droping the output column
pred1 = pred1.drop(["Median_House_Value"], axis = 1)
pred2 = pred2.drop(["Median_House_Value"], axis = 1)
pred3 = pred3.drop(["Median_House_Value"], axis = 1)
pred4 = pred4.drop(["Median_House_Value"], axis = 1)
pred5 = pred5.drop(["Median_House_Value"], axis = 1)

#predicting
prediction1 = (model.predict(pred1)[0][0] + SUB_VALUE) / DIVIDE_VALUE
prediction2 = (model.predict(pred2)[0][0] + SUB_VALUE) / DIVIDE_VALUE
prediction3 = (model.predict(pred3)[0][0] + SUB_VALUE) / DIVIDE_VALUE
prediction4 = (model.predict(pred4)[0][0] + SUB_VALUE) / DIVIDE_VALUE
prediction5 = (model.predict(pred5)[0][0] + SUB_VALUE) / DIVIDE_VALUE

#comparing the actual and predicted result
print(f"\n\n\nTest Run with {user_node} Nodes | {user_epoch} Epochs")
print(f'Prediction 1:\n\tActualValue: {value1:,.2f}\n\tPrediction: {prediction1:,.2f}')
print(f'Prediction 2:\n\tActualValue: {value2:,.2f}\n\tPrediction: {prediction2:,.2f}')
print(f'Prediction 3:\n\tActualValue: {value3:,.2f}\n\tPrediction: {prediction3:,.2f}')
print(f'Prediction 4:\n\tActualValue: {value4:,.2f}\n\tPrediction: {prediction4:,.2f}')
print(f'Prediction 5:\n\tActualValue: {value5:,.2f}\n\tPrediction: {prediction5:,.2f}')
print(f"The test error rate is {test_error_rate}")

#plotting the history
# plot_loss(history)
#plotting the predictions vs actual values
house = [1, 2, 3, 4, 5]
actual_values = [value1, value2, value3, value4, value5]
predictions = [prediction1, prediction2, prediction3, prediction4, prediction5]

X_axis = np.arange(len(house)) 

plt.bar(X_axis - 0.2, actual_values, 0.4, label = 'Actual Values') 
plt.bar(X_axis + 0.2, predictions, 0.4, label = 'Predictions') 

plt.xticks(X_axis, house) 
plt.xlabel("Houses") 
plt.ylabel("Value (in Millions)") 
plt.title(f"Loss: {test_error_rate:.4f}: {user_node} Nodes | {user_epoch} Epochs") 
plt.legend() 

plt.show() 