import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

# Загружаем данные
data = load_iris()

# Разбиваем данные
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

# Создаем модели
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
et = ExtraTreesClassifier(random_state=42)

# Обучаем модели
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
et.fit(X_train, y_train)

# Делаем предсказания
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_et = et.predict(X_test)

# Точность моделей
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_et = accuracy_score(y_test, y_pred_et)

print("Decision Tree accuracy:", acc_dt)
print("Random Forest accuracy:", acc_rf)
print("Extra Trees accuracy:", acc_et)

# Визуализация дерева решений (Decision Tree)
plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=data.feature_names, class_names=data.target_names, rounded=True)
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree.png")  # Сохраняем фото
plt.show()

# Функция для графика важности признаков
def plot_feature_importance(model, model_name, filename):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,5))
    plt.title(f"Feature Importances in {model_name}")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), np.array(data.feature_names)[indices], rotation=45)
    plt.savefig(filename)  # Сохраняем фото
    plt.show()

# Важность признаков для всех моделей
plot_feature_importance(dt, "Decision Tree", "feature_importance_dt.png")
plot_feature_importance(rf, "Random Forest", "feature_importance_rf.png")
plot_feature_importance(et, "Extra Trees", "feature_importance_et.png")

# Сравнение точности моделей
accuracies = {
    "Decision Tree": acc_dt,
    "Random Forest": acc_rf,
    "Extra Trees": acc_et
}

plt.figure(figsize=(6,4))
plt.bar(accuracies.keys(), accuracies.values())
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.savefig("model_accuracy_comparison.png")  # Сохраняем фото
plt.show()
