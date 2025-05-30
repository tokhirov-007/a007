from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import pickle

data = fetch_openml(data_id=1489, as_frame=True)

X = data.data
y = data.target

if y.dtype == 'O':
    y = y.astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accurasy score: {acc:.4f}")

with open('task_8.sav', 'wb') as f:
    pickle.dump(model, f)

print('MOdel saqlandi task_8.sav')

def get_accuracy():
    return acc

print('pred: ', y_pred)
print('test: ', y_test)