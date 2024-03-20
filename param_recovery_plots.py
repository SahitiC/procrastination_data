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
mask = np.any(params == 0, axis=1)
final_params = params[~mask]
final_std_err = std_err[~mask]
cmap = plt.get_cmap('viridis')
# remove params with v high std error
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


for i in range(4):
    for j in range(i+1, 4):
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
        
plt.figure(figsize=(4, 4), dpi=300)
plt.scatter(final_params[:, 6], final_params[:, 7])
plt.xlabel(param_names[2])
plt.ylabel(param_names[3])
plt.ylim(-2, 0)
plt.xlim(0, 2)