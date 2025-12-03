from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time


def train_decision_tree(X_train, y_train, X_test, y_test, max_depth=None):
    """训练决策树分类器"""
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    start_time = time.time()
    dt.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = dt.predict(X_test)
    test_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, train_time, test_time, y_pred