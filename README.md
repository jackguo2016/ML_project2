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
