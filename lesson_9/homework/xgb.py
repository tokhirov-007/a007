from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(objective='multi:softprob', num_class=10, eval_metric='mlogloss', use_label_encoder=False)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ XGBoost classification report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(8, 6))
plt.savefig("xgb.png")
plt.show()
