# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the necessary python libraries to predict the employee churn rate.
2. Use pd.read_csvfrom pandas library to deploy dataset. 
3. Declare x variable as input features except Departments as it have categorical values and use x.head to display the information of x.
4. Declare y as predicted variable to predict feaure name left y.head to display the information of y.
5. Use decisiontreeClassifier from sklearn.tree to classify the features as variable as dt.
6. Use fit to train the model before use train_test_split for training and test data.
7. Use predict for prediction and store in y_pred.
8. Use plt_plot from sklearn.tree to construct decision tree.
9. Use plt.show() to show the construction of decision tree.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Harish R
RegisterNumber:  212224230085
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\admin\\Downloads\\Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
features=["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]
x=data[features]
x.head()
y=data["left"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:

![Screenshot 2025-04-24 003331](https://github.com/user-attachments/assets/2b316f8a-d8a4-45d3-88a5-9b8f1381252d)

![Screenshot 2025-04-24 002817](https://github.com/user-attachments/assets/2676716c-fa90-4cee-bd8b-99556af20683)

![Screenshot 2025-04-24 002829](https://github.com/user-attachments/assets/390f7019-7dd5-4ea4-9792-73bfe015b285)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
