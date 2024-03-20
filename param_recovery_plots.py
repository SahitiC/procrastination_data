import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3

# %%


def get_std_errs(inv_hessians):
    """
    sqrt of diagonal elements in a stack of inv hessians
    """

    std_err = []
    for i in range(len(inv_hessians)):

        std_err.append(np.sqrt(np.diagonal(inv_hessians[i, :, :])))

    return std_err


# %%
# load final_params
result = np.load('nyx/560300/result.npy', allow_pickle=True)
params = np.stack(result[:, 0])
inv_hessians = np.stack(result[:, 1])
std_err = np.array(get_std_errs(inv_hessians))

# %%

colors = np.array(['tab:blue', 'tab:orange', 'tab:green'])
# remove estimates where some parameter = 0
mask1 = np.any(params == 0, axis=1)
final_params = params[~mask1]
final_std_err = std_err[~mask1]
cmap = plt.get_cmap('viridis')
# remove params with v high std error
<< << << < HEAD
mask = np.any(final_std_err > 25, axis=1)
final_params = final_params[~mask]
final_std_err = final_std_err[~mask]

# color_dis = np.where(final_params[:, 1] < 0.35, 1, 0)  # efficacy < 0.35
plt.figure(figsize=(7, 5), dpi=300)
plt.scatter(final_params[:, 0], final_params[:, 4],
            c=final_std_err[:, 0], cmap=cmap)  # , color=colors[color]
cbar = plt.colorbar()
cbar.set_label('std error')
plt.xlabel('true discount factor')
plt.ylabel('estimated discount factor')
corr = np.round(np.corrcoef(final_params[:, 0], final_params[:, 4])[0, 1], 3)
plt.text(0.8, 0.05,
         f'corr = {corr}',
         color='red')
== == == =
mask2 = np.any(final_std_err > 20, axis=1)
final_params = final_params[~mask2]
final_std_err = final_std_err[~mask2]

# %%
n_param = 4
param_names = ['discount factor', 'efficacy', 'reward_shirk', 'effort_work',
               'beta']
x_lim = [None, None, (0, 2), (-2, 0), (0, 10)]
y_lim = [(0, 1), (0, 1), (0, 2), (-2, 0), (0, 10)]
bad = [0.25, 0.25, 0.3, 0.3, 2]
markers = np.array(['o', 'x'])
>>>>>> > a649feda558328766fa6a694ff128fd362d53aa4

plt.figure(figsize=(7, 5), dpi=300)
plt.scatter(final_params[:, 1], final_params[:, 5],
            c=final_std_err[:, 1], cmap=cmap)
cbar = plt.colorbar()
cbar.set_label('std error')
plt.xlabel('true efficacy')
plt.ylabel('estimated efficacy')
corr = np.round(np.corrcoef(final_params[:, 1], final_params[:, 5])[0, 1], 3)
plt.text(0.8, 0.18,
         f'corr = {corr}',
         color='red')

plt.figure(figsize=(7, 5), dpi=300)
plt.scatter(final_params[:300, 2], final_params[:300, 6],
            c=final_std_err[:300, 2], cmap=cmap)
cbar = plt.colorbar()
cbar.set_label('std error')
plt.xlabel('true reward shirk')
plt.ylabel('estimated reward shirk')
plt.xlim(0, 2)
plt.ylim(0, 2)

corr = np.round(np.corrcoef(final_params[:, 2], final_params[:, 6])[0, 1], 3)
plt.text(1.3, 1.75,
         f'corr = {corr}',
         color='red')

plt.figure(figsize=(7, 5), dpi=300)
plt.scatter(final_params[:300, 3], final_params[:300, 7],
            c=final_std_err[:300, 3], cmap=cmap)
cbar = plt.colorbar()
cbar.set_label('std error')
plt.xlabel('true effort work')
plt.ylabel('estimated effort work')
plt.xlim(-2, 0)
plt.ylim(-2, 0)

corr = np.round(np.corrcoef(final_params[:, 3], final_params[:, 7])[0, 1], 3)
plt.text(-1.9, -0.3,
         f'corr = {corr}',
         color='red')

# %%
# how do the histograms and pairwise correlations look like
#: for input params kept and output params
param_names = ['discount factor', 'efficacy', 'reward_shirk', 'effort_work']

for i in range(4):
    plt.figure(figsize=(4, 4), dpi=300)
    plt.hist(final_params[:, i])
    plt.xlabel(param_names[i])
    plt.ylabel('frequency')


lim = [None, None, (0, 2), (-2, 0), (0, 10)]
for i in range(n_param):
    for j in range(i+1, n_param):
        plt.figure(figsize=(4, 4), dpi=300)
        plt.scatter(final_params[:, i], final_params[:, j])
        plt.xlabel(param_names[i])
        plt.ylabel(param_names[j])


for i in range(4):
    plt.figure(figsize=(4, 4), dpi=300)
    plt.hist(final_params[:, i+4])
    plt.xlabel(param_names[i])
    plt.ylabel('frequency')


for i in range(4):
    for j in range(i+1, 4):
        plt.figure(figsize=(4, 4), dpi=300)
        plt.scatter(final_params[:, i+4], final_params[:, j+4])
        plt.xlabel(param_names[i])
        plt.ylabel(param_names[j])
        plt.xlim(lim[i])
        plt.ylim(lim[j])

# %%
# pairwise covariances between parameters
final_inv_hessians = inv_hessians[~mask1, :, :]
final_inv_hessians = final_inv_hessians[~mask2, :, :]
# remove super huge covariances
mask3 = np.ones(len(final_inv_hessians), dtype=bool)
mask3[np.unique(np.where(final_inv_hessians > 10)[0])] = False
final_inv_hessians = final_inv_hessians[mask3, :, :]

for i in range(n_param-1):
    cov = final_inv_hessians[:, i, i+1:n_param]
    plt.figure()
    plt.violinplot(cov)
    plt.xticks(np.arange(1, n_param-i),
               labels=param_names[i+1:n_param])
    plt.title(param_names[i])
