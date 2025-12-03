import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
import time
from datetime import datetime
from dimension_reduction import load_mnist_images, load_mnist_labels, apply_dimension_reduction
from svm_classifier import train_svm
from decision_tree import train_decision_tree

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ResultManager:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_results(self, results, params):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mnist_results_{timestamp}.pkl"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump({'results': results, 'params': params}, f)

        return filepath


def run_experiment(n_samples=5000, n_test_samples=1000, n_components=50):
    result_manager = ResultManager()

    print("加载MNIST数据...")
    X_train = load_mnist_images('data/train-images.idx3-ubyte')
    y_train = load_mnist_labels('data/train-labels.idx1-ubyte')
    X_test = load_mnist_images('data/t10k-images.idx3-ubyte')
    y_test = load_mnist_labels('data/t10k-labels.idx1-ubyte')

    X_train = X_train[:n_samples]
    y_train = y_train[:n_samples]
    X_test = X_test[:n_test_samples]
    y_test = y_test[:n_test_samples]

    methods = ['pca', 'kpca', 'mds', 'lle', 'isomap']
    results = {}

    for method in methods:
        print(f"应用 {method.upper()}...")
        try:
            # 记录总开始时间（包含降维）
            total_start_time = time.time()

            # 降维处理
            X_train_reduced, X_test_reduced = apply_dimension_reduction(
                X_train, X_test, n_components=n_components, method=method)

            # 记录降维时间
            reduction_time = time.time() - total_start_time
            print(f"  {method.upper()}降维耗时: {reduction_time:.2f}秒")

            # SVM分类（只记录分类器训练时间）
            svm_start = time.time()
            svm_acc, _, svm_test_time, _ = train_svm(
                X_train_reduced, y_train, X_test_reduced, y_test)
            svm_classification_time = time.time() - svm_start

            # 决策树分类
            dt_start = time.time()
            dt_acc, _, dt_test_time, _ = train_decision_tree(
                X_train_reduced, y_train, X_test_reduced, y_test, max_depth=20)
            dt_classification_time = time.time() - dt_start

            # 计算总训练时间 = 降维时间 + 分类器训练时间
            total_svm_time = reduction_time + svm_classification_time
            total_dt_time = reduction_time + dt_classification_time

            results[method] = {
                'SVM': {
                    'accuracy': svm_acc,
                    'train_time': total_svm_time,  # 总训练时间（包含降维）
                    'test_time': svm_test_time,
                    'reduction_time': reduction_time,  # 降维时间
                    'classification_time': svm_classification_time  # 分类器训练时间
                },
                '决策树': {
                    'accuracy': dt_acc,
                    'train_time': total_dt_time,  # 总训练时间（包含降维）
                    'test_time': dt_test_time,
                    'reduction_time': reduction_time,  # 降维时间
                    'classification_time': dt_classification_time  # 分类器训练时间
                }
            }

            print(f"{method.upper()} - SVM: {svm_acc:.4f}, 决策树: {dt_acc:.4f}")

        except Exception as e:
            print(f"{method} 错误: {e}")
            results[method] = None

    params = {
        'n_samples': n_samples,
        'n_test_samples': n_test_samples,
        'n_components': n_components
    }
    result_manager.save_results(results, params)
    return results, params


def visualize_results(results, params):
    methods = [m for m in ['pca', 'kpca', 'mds', 'lle', 'isomap'] if results.get(m) is not None]
    classifiers = ['SVM', '决策树']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 准确率比较
    accuracies = np.array([[results[method][clf]['accuracy'] for method in methods] for clf in classifiers])
    x = np.arange(len(methods))
    width = 0.35

    bars1 = axes[0, 0].bar(x - width / 2, accuracies[0], width, label='SVM', alpha=0.8, color='skyblue')
    bars2 = axes[0, 0].bar(x + width / 2, accuracies[1], width, label='决策树', alpha=0.8, color='lightcoral')
    axes[0, 0].set_xlabel('降维方法')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_title('分类准确率比较')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([m.upper() for m in methods])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for bar in bars1 + bars2:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)

    # 训练时间比较（包含降维）
    train_times = np.array([[results[method][clf]['train_time'] for method in methods] for clf in classifiers])
    for i, clf in enumerate(classifiers):
        axes[0, 1].plot(methods, train_times[i], marker='o', label=clf, linewidth=2)
    axes[0, 1].set_xlabel('降维方法')
    axes[0, 1].set_ylabel('总训练时间 (秒)')
    axes[0, 1].set_title('总训练时间比较（包含降维）')
    axes[0, 1].set_yscale('log')  # 使用对数尺度，因为MDS时间会很长
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 测试时间比较
    test_times = np.array([[results[method][clf]['test_time'] for method in methods] for clf in classifiers])
    for i, clf in enumerate(classifiers):
        axes[1, 0].plot(methods, test_times[i], marker='s', label=clf, linewidth=2)
    axes[1, 0].set_xlabel('降维方法')
    axes[1, 0].set_ylabel('测试时间 (秒)')
    axes[1, 0].set_title('测试时间比较')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 热力图
    sns.heatmap(accuracies, annot=True, fmt='.3f', xticklabels=[m.upper() for m in methods],
                yticklabels=classifiers, ax=axes[1, 1], cmap='YlOrRd')
    axes[1, 1].set_title('准确率热力图')

    plt.tight_layout()
    plt.savefig('results/comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印详细结果
    print("\n详细结果:")
    data = []
    for method in methods:
        for clf in classifiers:
            metrics = results[method][clf]
            data.append({
                '方法': method.upper(),
                '分类器': clf,
                '准确率': f"{metrics['accuracy']:.4f}",
                '总训练时间(秒)': f"{metrics['train_time']:.2f}",
                '降维时间(秒)': f"{metrics['reduction_time']:.2f}",
                '分类时间(秒)': f"{metrics['classification_time']:.2f}",
                '测试时间(秒)': f"{metrics['test_time']:.4f}"
            })

    df = pd.DataFrame(data)
    print(df.to_string(index=False))

    # 打印时间分析
    print("\n时间分析:")
    for method in methods:
        reduction_time = results[method]['SVM']['reduction_time']
        print(f"{method.upper()}降维时间: {reduction_time:.2f}秒")

        # 降维时间占比
        total_svm_time = results[method]['SVM']['train_time']
        reduction_ratio = (reduction_time / total_svm_time) * 100
        print(f"  SVM中降维时间占比: {reduction_ratio:.1f}%")


def main():
    print("MNIST降维与分类方法比较")

    while True:
        # 选择运行新实验还是加载旧结果
        choice = input("请选择:\n1. 运行新实验\n2. 加载保存的结果\n3. 退出程序\n请输入选择 (1/2/3): ").strip()

        if choice == '3':
            return

        if choice == '2':
            # 加载保存的结果
            from load_results import load_saved_results
            results, params = load_saved_results()

            if results is not None:
                visualize_results(results, params)
            else:
                print("加载失败，改为运行新实验")
                choice = '1'

        if choice == '1':
            # 运行新实验
            results, params = run_experiment(
                n_samples=5000,
                n_test_samples=1000,
                n_components=50
            )
            visualize_results(results, params)

if __name__ == "__main__":
    main()