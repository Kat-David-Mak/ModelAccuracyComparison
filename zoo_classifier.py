import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#reading the zoo_dataset.csv from end assessment folder
zoo_data = pd.read_csv('end assessment/zoo_dataset.csv')
X = zoo_data.iloc[:,1:17].values
print(X)
y = zoo_data.iloc[:,17:18].values
print(y)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.3,random_state=5)

#Bernoulli Naive Bayes
bernNB = BernoulliNB()
bernNB.fit(X_train, y_train.ravel())
y_pred = bernNB.predict(X_test)

print('Bernoulli Naive Bayes Model Accuracy:   ', accuracy_score(y_test, y_pred))

#Decision Tree
decisionTree = DecisionTreeClassifier()
decisionTree.fit(X_train, y_train.ravel())
y_pred = decisionTree.predict(X_test)

print('Decision Tree Model Accuracy:   ', accuracy_score(y_test, y_pred))
