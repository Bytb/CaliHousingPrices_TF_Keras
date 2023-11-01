#importing modules
import pandas as pd
import tensorflow as tf
from keras import *
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

#import dataset
data = pd.read_csv('/Users/calebfernandes/Downloads/CaliforniaHousing_Keras_Pandas/project/data/California_Housing_CitiesAdded.csv')
df = pd.DataFrame(data)

#removing unneeded columns (REDUCE DIMENSIONALITY)
new_df = df.drop(['Latitude', 'Longitude'], axis = 1)

#encoding using ONE HOT ENCODER
ohe = OneHotEncoder()
feature_array = ohe.fit_transform(new_df[["ocean_proximity", "City"]]).toarray()
feature = ohe.categories_
feature_labels = np.concatenate((feature[0], feature[1]))
features = pd.DataFrame(feature_array, columns = feature_labels)

# droping the columns because they are not needed anymore
new_df = new_df.drop(["ocean_proximity", "City"], axis = 1)
# concatenating the two dataFrames
final_df = pd.concat([new_df, features.set_index(new_df.index)], axis=1)

# creating test and training data set
train, test = train_test_split(final_df, shuffle = True, test_size = 0.3)
train = pd.DataFrame(train)
test = pd.DataFrame(test)

#Scale the values
scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(train)
test = scaler.transform(test)
#NOTE: important to save the scale values
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

#convert to dataframes
train = pd.DataFrame(train, columns = final_df.columns.values)
test = pd.DataFrame(test, columns = final_df.columns.values)

#creating predict files from the test file
pred1 = pd.DataFrame(train.iloc[[0]])
pred2 = pd.DataFrame(train.iloc[[1]])
pred3 = pd.DataFrame(train.iloc[[2]])
pred4 = pd.DataFrame(train.iloc[[3]])
pred5 = pd.DataFrame(train.iloc[[4]])

# #convert to csv files
train.to_csv("tf_train_df.csv", index=False)
test.to_csv("tf_test_df.csv", index=False)
# # #prediction files
pred1.to_csv("pred1.csv", index=False)
pred2.to_csv("pred2.csv", index=False)
pred3.to_csv("pred3.csv", index=False)
pred4.to_csv("pred4.csv", index=False)
pred5.to_csv("pred5.csv", index=False)