#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Build machine learning app


# In[70]:


# Import packages
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
import pickle


# In[50]:


# Import dataset
data = pd.read_csv("C:\\Users\\65961\\OneDrive\\Desktop\\Data_Products\\webscraping_hdb_requests_20210211224417.csv")


# In[51]:


# EDA
print(f"Columns in dataset: {'|'.join(list(data.columns))}")
print(f"First 5 records of data:\n{data.head()}")


# In[52]:


# Convert string categories to float

def label_encoder(df_column, df_column_name):
    
    item_list = []
    
    values_list = list(df_column.unique())
    
    num_name = df_column_name + "_num"
    
    num = 0
    for item in values_list:
        num += 1
        item_dict = {df_column_name: item,
                    num_name: num }
        item_list.append(item_dict)
    
    item_df = pd.DataFrame(item_list)
    
    # Output
    return item_df


# In[53]:


# Prepare data

# Convert string categories to float
data_town = label_encoder(data["town"], "town")
data_flat_type = label_encoder(data["flat_type"], "flat_type")
data = pd.merge(data, data_town, on = "town", how = "inner").reset_index(drop = True)
data = pd.merge(data, data_flat_type, on = "flat_type", how = "inner")

# Drop irrelevant columns
data.drop(columns = ["Unnamed: 0", "flat_model", "street_name", "remaining_lease", "lease_commence_date", "storey_range", "_id", "block"], axis = 1, inplace = True)

# Modify existing columns
data["month"] = data["month"].apply(lambda x: float(str(x).split(sep = "-")[0].strip()))
data["floor_area_sqm"] = data["floor_area_sqm"].apply(lambda x: float(str(x).strip()))
data["resale_price"] = data["resale_price"].apply(lambda x: float(str(x).strip()))
data["town_num"] = data["town_num"].apply(lambda x: str(x).lower().strip())
data["flat_type_num"] = data["flat_type_num"].apply(lambda x: str(x).lower())

# Drop records with at least 1 NA
data.dropna(how = "any", axis = 0, inplace = True)
data = data.reset_index(drop = True)

# Show dataframe
print(f"Number of rows: {data.shape[0]}\nNumber of columns: {data.shape[1]}")
print(f"First 10 records:\n{data.head(10)}")


# In[39]:


data.loc[:, data.columns != "resale_price"].values


# In[71]:


# Create a model
model = LinearRegression()

# Determine predictor and response variables
X = data.loc[:, ~data.columns.isin(["resale_price", "town", "flat_type"])].values
Y = data.loc[:, data.columns == "resale_price"].values

# Fit the model
model.fit(X, Y)

# Save model
with open("C:\\Users\\65961\\OneDrive\\Desktop\\Data_Products\\hdb_machine_learning_app_model.pkl", "wb") as f:
    pickle.dump(model, f)
