import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

language_1 = pd.read_csv("dataset.csv")
language_2 = pd.read_csv("Language Detection.csv")
language_1.head()
language_2.head()

language_1 = language_1.rename(columns = {'language':'Language'})
language_1.head()
data = pd.concat([language_1,language_2], ignore_index = True)
data.head()

data.isnull().sum()
data["Language"].value_counts()

X = np.array(data.Text)
y = np.array(data.Language)

cv = CountVectorizer()
X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB(alpha = 0.2, fit_prior = True)
model_next = SVC(kernel = "rbf", decision_function_shape = "ovo", random_state=42)
model.fit(X_train,y_train)
model_next.fit(X_train,y_train)
score = model.score(X_test,y_test)
score_next = model_next.score(X_test,y_test)
accuracy = round(score * 100, 2)
accuracy_next = round(score_next*100,2)
print(accuracy,"%")
print(accuracy_next,"%")

text_input = input("Enter text to detect: ")
text_transform = cv.transform([text_input]).toarray()
output = model.predict(text_transform)
print("Text is",output[0])
