#STEP !: import modules
import pandas as pd
from keras.models import load_model

#STEP 2: Load in training algorithm
model = load_model("/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/kerasModel/trained_model.h5")

#STEP 3: Load in predict data
predict_data = pd.read_csv("/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/predict_data_1.csv")
# removing the House Price column from data
predict_data = predict_data.drop(['Median_House_Value'], axis = 1).values

#STEP 4: setting up prediction
prediction = model.predict(predict_data)
#NOTE: this will always return a 2D array, so to get one input, you have to do this -->
prediction = prediction[0][0]
#NOTE: important to recale value
prediction += 0.000405
prediction /= 0.0000008358

#this is the actual value
actual_value = 0.18824871
actual_value += 0.000405
actual_value /= 0.0000008358

#STEP 5: print out values
print(f"Predicted House Value: {prediction}")
print(f"Actual Value: {actual_value}")