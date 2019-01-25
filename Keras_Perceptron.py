
# coding: utf-8

# In[1]:


from keras.layers import *
from keras.models import *
from keras import optimizers


# In[2]:


x = [1,2,3,4,5,6]
y = [2,3,6,5,10,13]
x, y


# In[3]:


model = Sequential([
    Dense(1, input_shape=(1,))
])
model.summary()


# In[4]:


sgd=optimizers.SGD(lr=0.0005)
model.compile(loss='mse',optimizer=sgd)


# In[5]:


model.fit(x,y,epochs=1000)


# In[6]:


y_pred=model.predict([7,8,9])
y_pred

