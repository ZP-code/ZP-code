#!/usr/bin/env python
# coding: utf-8

# # Iris Dataset - Sklearn

# In[4]:


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix

# Load Iris Dataset
iris = datasets.load_iris()   
x = iris.data
y = iris.target
class_names = iris.target_names # Names of the labels 
    
# Plot 2 first columns
plt.scatter(x[:,0], x[:,1], c = y, cmap = 'winter', label=iris.target_names)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Iris - Sepal')
plt.show()


# In[28]:


#Split data to train and test samples
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 42)
print(f'Size of train {x_train.shape[0]}, size of test {x_test.shape[0]}')


# In[29]:


# Train the model Logistic Regression
model = LogisticRegression(solver = 'liblinear', random_state = 0)
model.fit(x_train, y_train)
print("", model.classes_[0], class_names[0],'\n', model.classes_[1], class_names[1],'\n', model.classes_[2], class_names[2])


# In[30]:


# Test the model
prediction = model.predict(x_test)
print(prediction)


# In[31]:


expected = y_test
print(expected)


# In[32]:


expected == prediction


# In[33]:


# Classification Report & Accuracy score
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))


# In[39]:


titles_options = [("Confusion Matrix", None), ("Normalized Confusion Matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model, x_test, y_test, display_labels = class_names,cmap = plt.cm.Blues, normalize = normalize)
    disp.ax_.set_title(title)
plt.show()

