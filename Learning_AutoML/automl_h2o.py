# -*- coding: utf-8 -*-
"""AutoML_h2o.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pRKWa9n3bAYt4R0if4ii8vrWHYTDYp2W

# AutoML using H2o

## Installing H2o
"""

!apt-get install defualt-jre
!java -version

!pip install h2o

import h2o

h2o.init()

"""- This will start the h2o cluster. We can start using it now.

## Using H2o AutoML

### Loading Data

Features of H2o: -

- It can do Data preprocessing. Converting categorical and continous.
- Take care of missing value imputation.
- Model selection it allwos a good leaderboard for models.
- It provides a deployment ready code. 
- It gives in multiple format. MOJO, POJO and binary. Recommended MOJO
- Supports GPU for XGBoost.

- Works on both GPU and CPU

- Feature enginerring isn't possible by AutoML. It is not efficient
"""

from h2o.automl import H2OAutoML

# !nvidia-smi

# this returns an h2o dataframe not pandas. So be careful
churn_df = h2o.import_file('https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

churn_df.types

churn_df.describe()

"""- Splitting the dataframe using for train test and val"""

churn_train, churn_test, churn_valid = churn_df.split_frame(ratios=[0.7, 0.15])

churn_train

"""- Remove the y column that you want to use for predictions.
- Predict the valuse of x using these x
"""

y = "Churn"
x = churn_df.columns
x.remove(y)
x.remove("customerID")

"""- Here we use the AutoML suite to providde us models automatically that will be trained.

- These models and hyperparamters are tuned automatically.

- Exclude Algos will exclude the algorithms that are not required.

- nfolds will take nfolds cross-validation steps from train data.

- We can use other parameters such as maxmodel which will restrict the time for convergence.

### Getting the Model and Predictions
"""

clf = H2OAutoML(max_models=10, seed=10, exclude_algos=["StackedEnsemble", "DeepLearning"], verbosity="info", nfolds=0)

# still not loaded to GPU
!nvidia-smi

clf.train(x=x, y=y, training_frame=churn_train, validation_frame=churn_valid)

!nvidia-smi
# It occupies significant GPU storage

leaderboard = clf.leaderboard

leaderboard.head()

churn_pred = clf.leader.predict(churn_test)

# Gives predictions and probabilites
churn_pred.head()

# Generates Performance report
clf.leader.model_performance(churn_test)

"""### Selecting Custom Model from leaderboards and using it"""

model_ids = list(clf.leaderboard['model_id'].as_data_frame().iloc[:, 0])

model_ids

xgb_models = []
for model in model_ids:
    if "XGBoost" in model:
        xgb_models.append(model)
print(xgb_models)

# h2o.get_model([mid for mid in model_ids if "XGBoost" in mid][0])
xgb_model = xgb_models[0]

xgb_chosen = h2o.get_model(xgb_model)

# Note these are h2o params which maybe not interpretable
xgb_chosen.params

# These are XGBoost paramters that are interpretable
xgb_chosen.convert_H2OXGBoostParams_2_XGBoostParams()

print(xgb_chosen)

print(xgb_chosen.confusion_matrix())

# Shows us the variable importance
print(xgb_chosen.varimp_plot())

# Donwloading the model to deploy it
xgb_chosen.download_model('/content/')
xgb_chosen.download_mojo('/content/')
xgb_chosen.download_pojo('/content/')