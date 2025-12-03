import pickle
import os


def load_saved_results(results_dir='results'):
    """加载保存的实验结果"""
    if not os.path.exists(results_dir):
        print("结果目录不存在")
        return None, None

    # 查找所有结果文件
    pkl_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
    if not pkl_files:
        print("未找到结果文件")
        return None, None

    pkl_files.sort()  # 按时间排序

    # 显示可用的结果文件
    print("\n可用的结果文件:")
    for i, filename in enumerate(pkl_files, 1):
        print(f"{i}. {filename}")

    # 选择文件
    try:
        choice = int(input("\n请选择要加载的文件编号: "))
        if choice < 1 or choice > len(pkl_files):
            print("无效的选择")
            return None, None

        filename = pkl_files[choice - 1]
        filepath = os.path.join(results_dir, filename)

        # 加载结果
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        print(f"已加载: {filename}")
        return data['results'], data['params']

    except (ValueError, IndexError):
        print("无效的输入")
        return None, None


def main():
    """主函数 - 选择加载保存的结果"""
    results, params = load_saved_results()

    if results is not None:
        print("\n加载成功！")
        print(f"参数: 样本数={params['n_samples']}, 测试数={params['n_test_samples']}, 维度={params['n_components']}")

        # 显示基本结果
        methods = [m for m in results.keys() if results.get(m) is not None]
        for method in methods:
            svm_acc = results[method]['SVM']['accuracy']
            dt_acc = results[method]['决策树']['accuracy']
            print(f"{method.upper()}: SVM={svm_acc:.4f}, 决策树={dt_acc:.4f}")


if __name__ == "__main__":
    main()