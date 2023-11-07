import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast

# %%

data = pd.read_csv('FollowUpStudymatrixDf_finalpaper.csv')
data = data.dropna(subset=['delta progress'])
data = data.reset_index(drop=True)

# keep relevant data
data_relevant = data[['SUB_INDEX_194',
                      'Total credits',
                      'date granted list',
                      'credit list',
                      'delta progress',
                      'cumulative progress']]

# are all progress arrays the same length (an entry for each day of semester)
for i in range(len(data)):
    print(len(
        ast.literal_eval(data_relevant['cumulative progress'][i])
    ))

# who completed lesser than 7 hrs

# when they hit 7 hr mark vs number of credits completed
# hypothesis: those who completed 7 hr later don't have much time to complete
# more credits (so lose out due to procrastination")

# cluster sequences (completion times)
