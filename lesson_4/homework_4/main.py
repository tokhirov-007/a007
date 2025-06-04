import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep=';')

X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train_class, y_test_class = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.2, random_state=42)

dt_clf = DecisionTreeClassifier(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
et_clf = ExtraTreesClassifier(random_state=42)

dt_clf.fit(X_train, y_train_class)
rf_clf.fit(X_train, y_train_class)
et_clf.fit(X_train, y_train_class)

dt_reg = DecisionTreeRegressor(random_state=42)
rf_reg = RandomForestRegressor(random_state=42)
et_reg = ExtraTreesRegressor(random_state=42)

dt_reg.fit(X_train_reg, y_train_reg)
rf_reg.fit(X_train_reg, y_train_reg)
et_reg.fit(X_train_reg, y_train_reg)

acc_dt = accuracy_score(y_test_class, dt_clf.predict(X_test))
acc_rf = accuracy_score(y_test_class, rf_clf.predict(X_test))
acc_et = accuracy_score(y_test_class, et_clf.predict(X_test))

print("Classification Accuracy:")
print(f"Decision Tree: {acc_dt:.3f}")
print(f"Random Forest: {acc_rf:.3f}")
print(f"Extra Trees: {acc_et:.3f}")

r2_dt = r2_score(y_test_reg, dt_reg.predict(X_test_reg))
r2_rf = r2_score(y_test_reg, rf_reg.predict(X_test_reg))
r2_et = r2_score(y_test_reg, et_reg.predict(X_test_reg))




print("\nRegression Results:")
print(f"Decision Tree - R2: {r2_dt:.3f}")
print(f"Random Forest - R2: {r2_rf:.3f}")
print(f"Extra Trees - R2: {r2_et:.3f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Classification Accuracy")
plt.bar(["Decision Tree", "Random Forest", "Extra Trees"], [acc_dt, acc_rf, acc_et], color=["#66c2a5", "#fc8d62", "#8da0cb"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.title("Regression R² Score")
plt.bar(["Decision Tree", "Random Forest", "Extra Trees"], [r2_dt, r2_rf, r2_et], color=["#66c2a5", "#fc8d62", "#8da0cb"])
plt.ylim(0, 1)
plt.ylabel("R² Score")

plt.tight_layout()
plt.savefig("tree_models_comparison.png")
plt.show()

