import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# read in the CSV file as a pandas dataframe
df = pd.read_csv('movie_dataset.csv')

# separate the text and sentiment columns
X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.2,random_state=5)

# define the using Neural Network pipeline
neural_network_model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MLPClassifier(hidden_layer_sizes=(10,), max_iter=10000, random_state=5))
])

# define the Bernoulli Naive Bayes pipeline
bernoulli_naive_bayes = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', BernoulliNB())
])

# train the Neural Network Model
neural_network_model.fit(X_train, y_train.ravel())

# train the Bernoulli Naive Bayes Model
bernoulli_naive_bayes.fit(X_train, y_train.ravel())

# accuracy, confusion matrix, precision score and recall score for Neural Network Model
y_pred = neural_network_model.predict(X_test)
nn_c_matrix = confusion_matrix(y_test, y_pred)
nn_precision = precision_score(y_test, y_pred)
nn_recall = recall_score(y_test, y_pred)
print('Neural Network Model Accuracy:   ', accuracy_score(y_test, y_pred))
print('Neural Network Confusion Matrix: ', nn_c_matrix)
print('Neural Network Precision Score: ', nn_precision)
print('Neural Network Recall Score: ', nn_recall)

# accuracy, confusion matrix, precision score and recall score for Bernoulli Naive Bayes Model
y_pred = bernoulli_naive_bayes.predict(X_test)
bn_c_matrix = confusion_matrix(y_test, y_pred)
bn_precision = precision_score(y_test, y_pred)
bn_recall = recall_score(y_test, y_pred)
print('Bernoulli Naive Bayes Model Accuracy:   ', accuracy_score(y_test, y_pred))
print('Bernoulli Naive Bayes Confusion Matrix: ', bn_c_matrix)
print('Bernoulli Naive Bayes Precision Score: ', bn_precision)
print('Bernoulli Naive Bayes Recall Score: ', bn_recall)
