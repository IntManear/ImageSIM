#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
digits = datasets.load_digits()
def display_img(img_no):
    fig , ax = plt.subplots()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.matshow(digits.images[img_no], cmap = plt.cm.binary)
display_img(0)


# In[5]:


digits.images[0]


# In[6]:


digits.data[0].shape


# In[7]:


digits.target[0]


# In[8]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# In[10]:


X = digits.data
co_sim = cosine_similarity(X[0].reshape(1,-1), X)


# In[11]:


cosf = pd.DataFrame(co_sim).T
cosf.columns = ['similarity']
cosf.sort_values('similarity',ascending = False)


# In[12]:


display_img(1626)


# In[13]:


from sklearn.metrics.pairwise import chi2_kernel
k_sim = chi2_kernel(X[0].reshape(1,-1),X)
kf = pd.DataFrame(k_sim).T
kf.columns = ['similarity']
kf.sort_values('similarity', ascending = False)


# In[14]:


display_img(1167)


# In[15]:


import graphlab
graphlab.canvas.show()


# In[16]:


graphlab.canvas.set_target('ipynb')


# In[18]:


gl_img = graphlab.SFrame('http://s3.amazonaws.com/dato-datasets/coursera/deep_learning/image_train_data')
gl_img


# In[39]:


gl_img['image'][0:5].show()


# In[21]:


graphlab.image_analysis.resize(gl_img['image'][2:3], 96,96).show()


# In[27]:


img = graphlab.Image('/home/dhruv/Pictures/Webcam/2019-04-27-024041.jpg')
ppsf = graphlab.SArray([img])
ppsf = graphlab.image_analysis.resize(ppsf,32,32)
ppsf.show()


# In[29]:


ppsf = graphlab.SFrame(ppsf).rename({'X1' : 'image'})
ppsf


# In[40]:


deep_learing_model = graphlab.load_model('https://static.turi.com/products/graphlab-create/resources/models/python2.7/imagenet_model_iter45')
# ppsf['deep_features'] = deep_learning_model.extract_features(ppsf)
# ppsf


# In[41]:


ppsf['deep_features'] = deep_learing_model.extract_features(ppsf)
ppsf


# In[42]:


ppsf['label'] = 'me'
gl_img['id'].max()


# In[43]:


ppsf['id'] = 50000
ppsf


# In[44]:


labels = ['id','image','label','deep_features']
part_train = gl_img[labels]
new_train = part_train.append(ppsf[labels])
new_train.tail()


# In[45]:


knn_model = graphlab.nearest_neighbors.create(new_train, features = ['deep_features'], label = 'id')


# In[47]:


cat_test = new_train[-2:-1]
graphlab.image_analysis.resize(cat_test['image'],96,96).show()


# In[48]:


sin_frame = knn_model.query(cat_test)
sin_frame


# In[50]:


def reveal_my_twin(x):
    return gl_img.filter_by(x['reference_label'],'id')

spirit_animal = reveal_my_twin(knn_model.query(cat_test))
spirit_animal['image'].show()


# In[51]:


me_test = new_train[-1:]
graphlab.image_analysis.resize(me_test['image'], 96,96).show()


# In[52]:


sim_frame = knn_model.query(me_test)
sim_frame


# In[53]:


spirit_animal = reveal_my_twin(knn_model.query(me_test))
spirit_animal['image'].show()


# In[55]:


me_test = new_train[-1:]
graphlab.image_analysis.resize(me_test['image'],96,96).show()


# In[56]:


sim_frame = knn_model.query(me_test)
sim_frame


# In[57]:


spirit_animal = reveal_my_twin(knn_model.query(me_test))
spirit_animal['image'].show()


# In[58]:


graphlab.image_analysis.resize(spirit_animal['image'][0:1],96,96).show()


# In[ ]:




