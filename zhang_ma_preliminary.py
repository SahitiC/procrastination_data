from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import ast
import scipy.stats
mpl.rcParams['font.size'] = 14

# %%

data = pd.read_csv('zhang_ma/FollowUpStudymatrixDf_finalpaper.csv')

data_relevant = data.dropna(subset=['delta progress'])
data_relevant = data_relevant.reset_index(drop=True)

# are all progress arrays the same length (an entry for each day of semester)
for i in range(len(data_relevant)):
    print(len(
        ast.literal_eval(data_relevant['delta progress'][i])
    ))
    plt.plot(ast.literal_eval(data_relevant['cumulative progress'][i]))

semester_length = len(ast.literal_eval(
    data_relevant['delta progress'][0]))

# %%

# who completed lesser than 7 hrs, when did they do their credits
participants_less7 = data_relevant[data_relevant['Total credits'] < 7]
participants_less7 = participants_less7.reset_index(drop=True)
print(participants_less7['Total credits'])
for i in range(len(participants_less7)):
    plt.plot(np.arange(semester_length),
             ast.literal_eval(participants_less7['cumulative progress'][i]),
             linewidth=2,
             label=f" {participants_less7['SUB_INDEX_194'][i]} ")
plt.legend(frameon=False, title='sub index')
plt.ylabel('cumulative credits completed')
plt.xlabel('day in semester')


# why do participants 24, 55, 126 not finish 7 credits
with pd.option_context('display.max_colwidth', 200):
    print(participants_less7.loc[[1, 2, 5], 'way_allocate_time'])
    print(participants_less7.loc[[1, 2, 5],
          'TextReport_cause_procrastination'])
# why do participants 1, 95, 111 give up after 1 credit, remove them from set
with pd.option_context('display.max_colwidth', 200):
    print(participants_less7.loc[[0, 3, 4], 'way_allocate_time'])
    print(participants_less7.loc[[0, 3, 4],
          'TextReport_cause_procrastination'])
# drop the ones that discontinued (subj. 1, 95, 111)
data_relevant = data_relevant.drop([1, 90, 104])
data_relevant = data_relevant.reset_index(drop=True)

# %%

when_hit_7 = []
for i in range(len(data_relevant)):

    if data_relevant['Total credits'][i] >= 7:
        temp = np.array(
            ast.literal_eval(data_relevant['cumulative progress'][i])
        )
        when_hit_7.append(np.where(temp >= 7)[0][0]+1)
    else:
        when_hit_7.append(np.nan)
data_relevant['when hit 7 credits'] = when_hit_7

plt.figure()
plt.scatter(data_relevant['when hit 7 credits'],
            data_relevant['Total credits'])
plt.xlabel('day when 7 credits are hit')
plt.ylabel('total credits completed')
temp = data_relevant[['when hit 7 credits', 'Total credits']].dropna()
print(scipy.stats.pearsonr(temp['when hit 7 credits'],
                           temp['Total credits']))


# %%
# cluster sequences (unit completion times)

# normalise cumulative series
cumulative_normalised = []
for i in range(len(data_relevant)):
    temp = np.array(
        ast.literal_eval(data_relevant['cumulative progress'][i]))
    cumulative_normalised.append(temp/data_relevant['Total credits'][i])
data_relevant['cumulative progress normalised'] = cumulative_normalised

km = TimeSeriesKMeans(n_clusters=8, n_init=5, metric="euclidean", verbose=True)
timseries_to_cluster = np.vstack(
    data_relevant['cumulative progress normalised'])
labels = km.fit_predict(timseries_to_cluster)
data_relevant['labels'] = labels

for label in set(data_relevant['labels']):
    plt.figure()

    for i in range(len(data_relevant)):

        if data_relevant['labels'][i] == label:
            # ast.literal_eval(data_relevant['delta progress'][i])
            # data_relevant['cumulative progress normalised'][i]
            plt.plot(ast.literal_eval(data_relevant['cumulative progress'][i]),
                     alpha=0.5)

# %%

# when they hit 7 hr mark vs number of credits completed
# hypothesis: those who completed 7 hr later don't have much time to complete
# more credits (so lose out due to procrastination)
fig, ax = plt.subplots()
scatter = ax.scatter(data_relevant['when hit 7 credits'],
                     data_relevant['Total credits'],
                     c=data_relevant['labels'])
ax.set_xlabel('day when 7 credits are hit')
ax.set_ylabel('total credits completed')
legend = ax.legend(*scatter.legend_elements(),
                   title="clusters")
ax.add_artist(legend)
plt.show()

# do they also have high regret
# it depends on cluster! low credits dont necessarily have high regret
plt.figure()
plt.scatter(data_relevant['SatifactionTimeAllocationWay'],
            data_relevant['Total credits'],
            c=data_relevant['labels'])
plt.ylabel('total credits completed')
plt.xlabel('satisfaction with time allocation')
temp = data_relevant[['SatifactionTimeAllocationWay',
                      'Total credits']].dropna()
print(scipy.stats.pearsonr(temp['SatifactionTimeAllocationWay'],
                           temp['Total credits']))

plt.figure()
plt.scatter(data_relevant['SatifactionTimeAllocationWay'],
            data_relevant['when hit 7 credits'],
            c=data_relevant['labels'])
plt.ylabel('day when 7 credits are hit')
plt.xlabel('satisfaction with time allocation')
temp = data_relevant[['SatifactionTimeAllocationWay',
                      'when hit 7 credits']].dropna()
print(scipy.stats.pearsonr(temp['SatifactionTimeAllocationWay'],
                           temp['when hit 7 credits']))

# how do procrastination scores vary with cluster
temp = data_relevant[['labels', 'AcadeProcFreq_mean']].groupby(
    ['labels']).mean()
temp_std = data_relevant[['labels', 'AcadeProcFreq_mean']].groupby(
    ['labels']).std()
temp.plot(kind='bar',
          yerr=temp_std,
          ylabel='Proc score mean',
          xlabel='cluster',
          legend=None)

# what about risk-taking and procrastination reasons?
variables = ['ReasonProc_ExcitingLastMoment',
             'ReasonProc_TimeManagement',
             'ReasonProc_TaskAversiveness',
             'ReasonProc_Laziness',
             'RiskTakingScore',
             'DiscountRate_lnk',
             'RiskAttitude_lnAlpha',
             'SatifactionTimeAllocationWay']
for i_v, v in enumerate(variables):
    temp = data_relevant[['labels', v]].groupby(
        ['labels']).mean()
    temp_std = data_relevant[['labels', v]].groupby(
        ['labels']).std()
    temp.plot(kind='bar',
              yerr=temp_std,
              ylabel=v,
              xlabel='cluster',
              legend=None)

# %%
# save text responses cluster-wise to .txt file
label = 7
temp = []
for i in range(len(data_relevant)):

    if data_relevant['labels'][i] == label:

        temp.append(data_relevant['TextReport_cause_procrastination'][i])
        # TextReport_cause_procrastination
        # way_allocate_time

with open('temp.txt', 'w') as f:
    for i in temp:
        f.write(f"{i}\n")
