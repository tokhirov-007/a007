from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import seaborn as sns
import pandas as pd 

titanic = sns.load_dataset('titanic')

titanic = titanic.dropna(subset=['survived', 'pclass', 'sex', 'age', 'fare'])

titanic= pd.get_dummies(titanic, columns=['pclass', 'sex'], drop_first=True)

X = titanic[['age', 'fare', 'pclass_2', 'pclass_3', 'sex_male']]

y = titanic['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {acc:.4f}")

with open("task_5.sav", "wb") as f:
    pickle.dump(model, f)

print("MOdel saqlandi task_5.sav")

def get_accuracy():
    return acc

print("pred: ", y_pred)
print("test ", y_test)
