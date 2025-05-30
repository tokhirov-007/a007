import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from task_1.task_1 import get_accuracy as acc1
from task_2.task_2 import get_accuracy as acc2
from task_3.task_3 import get_accuracy as acc3
from task_4.task_4 import get_accuracy as acc4
from task_5.task_5 import get_accuracy as acc5
from task_6.task_6 import get_accuracy as acc6
from task_7.task_7 import get_accuracy as acc7
from task_8.task_8 import get_accuracy as acc8
from task_9.task_9 import get_accuracy as acc9
from task_10.task_10 import get_accuracy as acc10

accuracies = {
    "task_1": acc1(),
    "task_2": acc2(),
    "task_3": acc3(),
    "task_4": acc4(),
    "task_5": acc5(),
    "task_6": acc6(),
    "task_7": acc7(),
    "task_8": acc8(),
    "task_9": acc9(),
    "task_10": acc10()
}



plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values())
plt.title("Logistic Regression Accuracy (10 Datasets)")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')

plt.tight_layout()
plt.savefig("Chart.png")
print("Chart saqlandi: Chart.png")
