# -*- coding: utf-8 -*-
"""rapids_data_handling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CWHEMeMD3gGkygzK6b71CGR_-7mKVPtw

# Data Handling in RAPIDS

## Installing Rapids

- Note again use NVIDIA T4 or P4 or P100 GPU only
"""

!nvidia-smi

# Install RAPIDS
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!bash rapidsai-csp-utils/colab/rapids-colab.sh

import sys, os

dist_package_index = sys.path.index('/usr/local/lib/python3.6/dist-packages')
sys.path = sys.path[:dist_package_index] + ['/usr/local/lib/python3.6/site-packages'] + sys.path[dist_package_index:]
sys.path
exec(open('rapidsai-csp-utils/colab/update_modules.py').read(), globals())

"""# Data Analysis"""

import cudf
import numpy as np
import dask_cudf

bank_df = cudf.read_csv('https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/bank-full.csv',sep=';')

"""1 - age (numeric)

2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur", "student","blue-collar","self-employed","retired","technician","services")

3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)

4 - education (categorical: "unknown","secondary","primary","tertiary")

5 - default: has credit in default? (binary: "yes","no")

6 - balance: average yearly balance, in euros (numeric)

7 - housing: has housing loan? (binary: "yes","no")

8 - loan: has personal loan? (binary: "yes","no")

related with the last contact of the current campaign:

9 - contact: contact communication type (categorical: "unknown","telephone","cellular")

10 - day: last contact day of the month (numeric)

11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")

12 - duration: last contact duration, in seconds (numeric)

other attributes:

13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)

15 - previous: number of contacts performed before this campaign and for this client (numeric)

16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

output variable (desired target):

17 - y - has the client subscribed a term deposit? (binary: "yes","no")
"""

! nvidia-smi

print("rows: ", bank_df.shape[0])
print("columns: ", bank_df.shape[1])

bank_df.dtypes

bank_df.isnull().sum()

bank_df['y'].value_counts()

"""# Benchmarking against dask cudf vs cudf"""

import time

start_time = time.time()
bank_df.describe()
end_time = time.time()
print("Time taken on GPU : %s" %(end_time - start_time))

dcudf = dask_cudf.from_cudf(bank_df, npartitions=2)

start_time = time.time()
dcudf.describe()
end_time = time.time()
print("Time taken on GPU : %s" %(end_time - start_time))

"""# Exploring Data"""

bank_df.describe()

bank_df.groupby(['marital', 'y']).agg({'balance':'mean'})

bank_df.groupby(['marital', 'y']).agg({'balance':'mean', 'y': 'count'})

loan_outcome = bank_df.groupby(['loan', 'y']).agg({'balance':'mean','y':'count'})

print(loan_outcome)

def convert_hour(duration):
    return duration / 60

bank_df['duration_hour'] = bank_df['duration'].applymap(convert_hour)

bank_df.head()

bank_df.groupby('y').campaign.mean()

bank_campaign_df = bank_df.query("campaign <= 8")

bank_df['education'].value_counts()