# 机器学习算法实现

本项目实现了三种基础机器学习算法：线性回归、感知机和支持向量机（SVM）。每种算法都应用于不同的数据集，展示了它们在解决实际问题中的应用。

## 项目概览

本项目旨在深入理解机器学习基础算法的原理和实现。通过对真实数据集的应用，展示了这些算法在实际场景中的效果和局限性。

## 算法实现

1. **线性回归**
   - 文件：`Linear_Regression.py`
   - 特点：使用梯度下降优化参数
   - 应用：基于居住面积预测房屋价格

2. **感知机**
   - 文件：`python_HW2_Perceptron.ipynb`
   - 特点：包含权重更新、训练和分类函数
   - 功能：提供交互式界面，可视化决策边界

3. **支持向量机（SVM）**
   - 文件：`SupportVectorMachine.py`
   - 特点：使用梯度下降进行优化
   - 应用：乳腺癌诊断

## 使用说明

1. 克隆仓库：
   ```
   git clone https://github.com/ziyan-guo/ml-algorithms-implementation.git
   ```

2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

3. 运行各算法：
   - 线性回归：`python Linear_Regression.py`
   - 感知机：在Jupyter Notebook中运行`python_HW2_Perceptron.ipynb`
   - SVM：`python SupportVectorMachine.py`

## 数据集

- `kc_house_data.csv`：King County房屋销售数据
- `data.csv`：威斯康星乳腺癌诊断数据集

## 依赖库

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 贡献

欢迎提出问题和改进建议。如有兴趣贡献代码，请查看[issues页面](https://github.com/ziyan-guo/ml-algorithms-implementation/issues)。

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE.md](LICENSE.md) 文件。


# Machine Learning Algorithms Implementation

This project implements three fundamental machine learning algorithms: Linear Regression, Perceptron, and Support Vector Machine (SVM). Each algorithm is applied to a different dataset to demonstrate its practical use in solving real-world problems.

## Table of Contents
- [Overview](#overview)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Dataset](#dataset)

## Overview

This project showcases the implementation and application of three basic machine learning algorithms. It serves as a practical demonstration of how these algorithms work and can be applied to different types of data.

## Algorithms

### 1. Linear Regression
- Implemented in `Linear_Regression.py`
- Uses gradient descent to optimize parameters
- Applied to predict house prices based on living area

### 2. Perceptron
- Implemented in `python_HW2_Perceptron.ipynb`
- Includes functions for weight updating, training, and classification
- Features an interactive interface for visualizing decision boundaries

### 3. Support Vector Machine (SVM)
- Implemented in `SupportVectorMachine.py`
- Uses gradient descent for optimization
- Applied to breast cancer diagnosis

## Installation

To run this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/ziyan-guo/ml-algorithms-implementation.git
cd ml-algorithms-implementation
pip install -r requirements.txt
```

## Usage

Each algorithm can be run separately:

1. Linear Regression:
   ```
   python Linear_Regression.py
   ```

2. Perceptron:
   Open and run `python_HW2_Perceptron.ipynb` in Jupyter Notebook

3. SVM:
   ```
   python SupportVectorMachine.py
   ```

## File Structure

- `Linear_Regression.py`: Linear regression implementation
- `python_HW2_Perceptron.ipynb`: Perceptron implementation
- `SupportVectorMachine.py`: SVM implementation
- `helperfunctions.py`: Utility functions for data loading and visualization
- `kc_house_data.csv`: Dataset for house price prediction
- `data.csv`: Dataset for breast cancer diagnosis

## Dependencies

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Dataset

- The house price dataset (`kc_house_data.csv`) contains information about house sales in King County, USA.
- The breast cancer dataset (`data.csv`) is the Breast Cancer Wisconsin (Diagnostic) Data Set, used for cancer diagnosis.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/ziyan-guo/ml-algorithms-implementation/issues) if you want to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
