from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time


def train_svm(X_train, y_train, X_test, y_test, kernel='rbf'):
    """训练SVM分类器"""
    svm = SVC(kernel=kernel, random_state=42)

    start_time = time.time()
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = svm.predict(X_test)
    test_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, train_time, test_time, y_pred