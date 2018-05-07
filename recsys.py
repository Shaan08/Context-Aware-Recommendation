
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cross_validation import train_test_split
import time
from sklearn.externals import joblib


# In[97]:


get_ipython().magic('matplotlib inline')


# In[98]:


order = pd.read_excel('C:/Users/ezarpkm/Desktop/CaseData/OrderRows.xlsx')
productinfo = pd.read_excel('C:/Users/ezarpkm/Desktop/CaseData/ProductInfo.xlsx')
productiddf = pd.read_excel('C:/Users/ezarpkm/Desktop/CaseData/Sheet2.xlsx')
#order1 = pd.read_excel(order, 'Sheet1')
#order2 = pd.read_excel(order, 'Sheet2')


# In[99]:


order.head()


# In[130]:


filterprod = pd.merge(productinfo, productiddf, on=['productno'])
filterorder = pd.merge(filterprod,order, on = ['productno'])
filterorder = filterorder.drop('createdon',axis =1)
filterorder.productid = filterorder.productid.astype(str) 
filterorder.dtypes
filterorder['category'] = filterorder['category'].map(str) + " -> " + filterorder['productid']
filterorder = filterorder.drop(['productid','orderno','productno'], axis = 1)
filterorder


# In[131]:


#aggregation
ford = filterorder.groupby(['category']).agg({'quantity': 'count'}).reset_index()
grouped_sum = ford['quantity'].sum()
ford['percentage']  = ford['quantity'].div(grouped_sum)*100

ford = ford.sort_values(['quantity', 'category'], ascending = [0,1])
data = ford
data


# In[132]:


#unique product productmanufacturerid
prod_manu_id = filterorder['category'].unique()
len(prod_manu_id)


# In[146]:


#create a product recommender
train_data, test_data = train_test_split(filterorder, test_size = 0.20, random_state = 0)
print(train_data.head(5))


# In[150]:


import rec as Recommenders


# In[151]:


pm = Recommenders.popularity_recommender_py()
pm.create(train_data,'productmanufacturerid','category')


# In[13]:


df = pd.DataFrame(filterorder.category.str.split('-',1).tolist(),
                                   columns = ['cat','sub-cat'])

temp1 = df['cat']
pd.DataFrame(temp1)
filterorder.append(temp1)


# In[ ]:


(order1.quantity).hist()
#np.log(order1.quantity).hist()
#productinfo.shape


# In[ ]:


filterprod.shape


# In[24]:


order2.head()


# In[33]:


#orderinfo = [order1, order2]
#new_order = pd.concat(orderinfo,  join = 'outer')
new_order = order1.append(order2)
new_order.head()


# In[11]:


order.head()
product.head()

