# First step is to install pandas, keras, scikit, and tensorflor 
# This part only deals with Pandas though
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#STEP 1: import data
updated_data = pd.read_csv("/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/California_Housing_CitiesAdded.csv")
data_df = pd.DataFrame(updated_data)

#STEP 2: Reduce dimensionality
data_df = data_df.drop(['Latitude', 'Longitude', 'Tot_Bedrooms', 'Distance_to_SanJose', 'ocean_proximity', 'City'], axis = 1)

#STEP 3: Create the training and test files
#NOTE: the "shuffle = true" will shuffle the data so you get a wide variety
#NOTE: "test_size = 0.3" means that the test data will recieve 30% of the data
train, test = train_test_split(data_df, shuffle = True, test_size= 0.3)

#STEP 4: Scale data
#create scaler function
scaler = MinMaxScaler(feature_range=(0,1)) #take in the range in which should be scaled
scaled_training = scaler.fit_transform(train)
scaled_test = scaler.transform(test)
#NOTE: important to save the scale values
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

#STEP 5: convert into DF
#NOTE: We need the column names intact, thats why we add in the "columns = " parameter
training_scaled_df = pd.DataFrame(scaled_training, columns=data_df.columns.values)
test_scaled_df = pd.DataFrame(scaled_test, columns=data_df.columns.values)

#STEP 6: Creating new CSV files
training_scaled_df.to_csv("training_data_scaled.csv", index = False)
test_scaled_df.to_csv("test_data_scaled.csv", index = False)