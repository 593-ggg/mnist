项目简介
比较五种降维方法（PCA、KPCA、MDS、LLE、Isomap）在MNIST手写数字数据集上的分类效果，使用SVM和决策树进行分类性能评估。

快速开始
下载数据集data.rar,解压后放在工作目录
安装依赖pip install -r requirements.txt

运行项目
python main_visualization.py

项目结构
├── main_visualization.py     # 主程序
├── dimension_reduction.py    # 降维算法
├── svm_classifier.py         # SVM分类器
├── decision_tree.py          # 决策树分类器
├── load_results.py           # 结果加载
├── requirements.txt          # 依赖配置
