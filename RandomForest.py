# Import pandas and numpy to read csv file
import pandas as pd
import numpy as np

# Read the csv file
data=pd.read_csv("cleaned_data.csv")
                                                             
encode=['company','fuel','seller_type','transmission','owner']
for col in encode:
   dummy = pd.get_dummies(data[col], prefix = col)
   data = pd.concat([data,dummy],axis = 1)
   del data[col]

# Separate input & target data
inputs_df = data.drop("selling_price",axis=1)
targets = data["selling_price"]


# Import train_test_split from sklearn library to make split of data into train sets and validation sets
from sklearn.model_selection import train_test_split
train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs_df,targets,test_size=0.25,random_state=42)

# Training a Random Forest 
                                                                         
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_jobs=-1, random_state=42)
RF.fit(train_inputs, train_targets)     

### Create a Pickle file using serialization 
import pickle
pickle_out = open("RF.pkl","wb")
pickle.dump(RF, pickle_out)
pickle_out.close()        
    

