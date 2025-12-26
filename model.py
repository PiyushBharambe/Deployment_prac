from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

data = pd.read_csv("dataset.csv")

x = data[["hours_studied", "attendance", "score"]]
y = data[["result"]] 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

Acc = (accuracy_score(y_test, y_pred)*100)

print("Accuracy: ", Acc)

if Acc >= 90:
    pickle.dump(model, open("model.pkl", "wb"))
    print("Model saved!")
else:
    print("Accuracy low")
