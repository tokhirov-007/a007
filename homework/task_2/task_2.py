from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy score :{acc:.4f}")

with open("task_2.sav", "wb") as f:
    pickle.dump(model, f)

print("MOdel saqlandi task_2.sav")

def get_accuracy():
    return acc


print("pred ", y_pred)
print("test ",  y_test)