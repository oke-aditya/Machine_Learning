# -*- coding: utf-8 -*-
"""eli5_intro.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C4siSj0CMyQS_4mYQyvTvMV3RXczgE68

# Introduction to Eli5

- ELI5 is a Python library which allows to visualize and debug various Machine Learning models using unified API. It has built-in support for several ML frameworks and provides a way to explain black-box models.

- Supported frameworks as of 13-02-2020

- scikit-learn: -
Currently ELI5 allows to explain weights and predictions of scikit-learn linear classifiers and regressors, print decision trees as text or as SVG, show feature importances and explain predictions of decision trees and tree-based ensembles.

- XGBoost: -
Show feature importances and explain predictions of XGBClassifier, XGBRegressor and xgboost.Booster.

- LightGBM: -
Show feature importances and explain predictions of LGBMClassifier and LGBMRegressor.
- CatBoost: -
Show feature importances of CatBoostClassifier and CatBoostRegressor.
- lightning: -
Explain weights and predictions of lightning classifiers and regressors.

- Keras : -
Explain predictions of image classifiers via Grad-CAM visualizations.

For a better package for black box deep learning models use tf_explain. Which works with tensorflow 2.0

- ELI5 also implements several algorithms for inspecting black-box models.

- TextExplainer allows to explain predictions of any text classifier using LIME algorithm (Ribeiro et al., 2016). There are utilities for using LIME with non-text data and arbitrary black-box classifiers as well, but this feature is currently experimental.

- Permutation Importance method can be used to compute feature importances for black box estimators.

# Installation and Dependencies

- pip install eli5

- Python 3.4 +
- sklearn 0.18 +

# Basic Usage

There are two main ways to look at a classification or a regression model:

- inspect model parameters and try to figure out how the model works globally;
- inspect an individual prediction of a model, try to figure out why the model makes the decision it makes.
"""

# Commented out IPython magic to ensure Python compatibility.
! pip install eli5
# %tensorflow_version 2.x
import tensorflow as tf

print(tf.__version__)

import eli5

# Create a clf and pass it to know the weights or predictions.
eli5.show_weights(clf)
eli5.show_prediction(clf)

"""# Why Eli5 and how to use it

- ELI5 aims to handle not only simple cases, but even for simple cases having a unified API for inspection has a value.

- algorithms like LIME (paper) try to explain a black-box classifier through a locally-fit simple, interpretable classifier. It means that with each additional supported “simple” classifier/regressor algorithms like LIME are getting more options automatically.
"""