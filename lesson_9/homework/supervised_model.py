from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Загружаем встроенный датасет
data = load_breast_cancer()
X = data.data
y = data.target

# Делим и обучаем
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
print("✅ Supervised model accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(12, 8))
plt.savefig("supervised.png")
plt.show()
