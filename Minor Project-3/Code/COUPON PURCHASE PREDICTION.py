#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

def list_all_files_in(dirpath):
    for dirname, _, filenames in os.walk(dirpath):
        for filename in filenames:
            print(os.path.join(dirname, filename))

list_all_files_in('../input')


# In[3]:


# Dataframes
import pandas as pd

# Linear algebra
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# List shifting
from collections import deque

# Sparse matrices
from scipy import sparse

# Displaying stuff
from IPython.display import display

# ZIP I/O
import zipfile

# Paths
from pathlib import Path

# Timing
import time

# Disable warnings
import warnings; warnings.simplefilter('ignore')


# In[10]:


df_coupon_list_train = pd.read_csv('coupon_list_train.csv')
# df_coupon_area_train = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_area_train_translated.csv')
df_coupon_detail_train = pd.read_csv('coupon_detail_train.csv')
# df_coupon_visit_train = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_visit_train_translated.csv')

df_coupon_list_test = pd.read_csv('coupon_list_test.csv')
# df_coupon_area_test = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_area_test_translated.csv')

df_user_list = pd.read_csv('user_list.csv')
# df_prefecture_locations = pd.read_csv('../input/coupon-purchase-prediction-translated/prefecture_locations_translated.csv')
# df_submission = pd.read_csv('../input/coupon-purchase-prediction-translated/sample_submission.csv')


# In[11]:


df_purchased_coupons_train = df_coupon_detail_train.merge(df_coupon_list_train, on='COUPON_ID_hash', how='inner')


# In[12]:


features = ['COUPON_ID_hash', 'USER_ID_hash', 'GENRE_NAME', 'DISCOUNT_PRICE', 'large_area_name', 'ken_name', 'small_area_name']
df_purchased_coupons_train = df_purchased_coupons_train[features]
df_purchased_coupons_train


# In[14]:


df_coupon_list_test['USER_ID_hash'] = 'dummyuser'
df_coupon_list_test = df_coupon_list_test[features]
df_coupon_list_test


# In[15]:


df_combined = pd.concat([df_purchased_coupons_train, df_coupon_list_test], axis=0)
df_combined['DISCOUNT_PRICE'] = 1 / np.log10(df_combined['DISCOUNT_PRICE'])
features.extend(['DISCOUNT_PRICE'])
df_combined


# In[16]:


categoricals = ['GENRE_NAME', 'large_area_name', 'ken_name', 'small_area_name']
df_combined_categoricals = df_combined[categoricals]
df_combined_categoricals = pd.get_dummies(df_combined_categoricals, dummy_na=False)
df_combined_categoricals


# In[17]:


continuous = list(set(features) - set(categoricals))
df_combined = pd.concat([df_combined[continuous], df_combined_categoricals], axis=1)
print(df_combined.isna().sum())
NAN_SUBSTITUTION_VALUE = 1
df_combined = df_combined.fillna(NAN_SUBSTITUTION_VALUE)
df_combined


# In[18]:


df_train = df_combined[df_combined['USER_ID_hash'] != 'dummyuser']
df_test = df_combined[df_combined['USER_ID_hash'] == 'dummyuser']
df_test = df_test.drop('USER_ID_hash', axis=1)
display(df_train)
display(df_test)


# In[19]:


df_train_dropped_coupons = df_train.drop('COUPON_ID_hash', axis=1)
df_user_profiles = df_train_dropped_coupons.groupby('USER_ID_hash').mean()
df_user_profiles


# In[20]:


FEATURE_WEIGHTS = {
    'GENRE_NAME': 2,
    'DISCOUNT_PRICE': 2,
    'large_area_name': 0.5,
    'ken_name': 1.5,
    'small_area_name': 5
}


# In[21]:


def find_appropriate_weight(weights_dict, colname):
    for col, weight in weights_dict.items():
        if col in colname:
            return weight
    raise ValueError


# In[22]:


W_values = [find_appropriate_weight(FEATURE_WEIGHTS, colname)
            for colname in df_user_profiles.columns]
W = np.diag(W_values)
W


# In[23]:


df_test_only_features = df_test.drop('COUPON_ID_hash', axis=1)
similarity_scores = np.dot(np.dot(df_user_profiles, W), df_test_only_features.T)
similarity_scores


# In[24]:


s_coupons_ids = df_test['COUPON_ID_hash']
index = df_user_profiles.index
columns = pd.Series([s_coupons_ids[i] for i in range(0, similarity_scores.shape[1])], name='COUPON_ID_hash')
df_results = pd.DataFrame(index=index, columns=columns, data=similarity_scores)
df_results


# In[25]:


def get_top10_coupon_hashes_string(row):
    sorted_row = row.sort_values()
    return ' '.join(sorted_row.index[-10:][::-1].tolist())


# In[26]:


output = df_results.apply(get_top10_coupon_hashes_string, axis=1)
output


# In[27]:


df_output = pd.DataFrame(data={'USER_ID_hash': output.index, 'PURCHASED_COUPONS': output.values})
df_output


# In[29]:


df_output_all = pd.merge(df_user_list, df_output, how='left', on='USER_ID_hash')
df_output_all.to_csv('cosine_sim_python.csv', header=True, index=False, columns=['USER_ID_hash', 'PURCHASED_COUPONS'])
df_output_all


# In[ ]:




