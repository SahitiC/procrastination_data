import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 2

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
params = np.loadtxt('result_beta.csv',
                    delimiter=',')

# %%
# load final_params
result = np.load('result_beta.npy', allow_pickle=True)
params = np.stack(result[:, 0])
inv_hessians = np.stack(result[:, 1])
std_err = np.array(get_std_errs(inv_hessians))

# %%

# remove estimates where some parameter = 0
mask = np.any(params == 0, axis=1)
final_params = params[~mask]
final_std_err = std_err[~mask]
cmap = plt.get_cmap('viridis')
# remove params with v high std error
mask = np.any(final_std_err > 20, axis=1)
final_params = final_params[~mask]
final_std_err = final_std_err[~mask]

# %%
n_param = 5
param_names = ['discount factor', 'efficacy', 'reward_shirk', 'effort_work',
               'beta']
x_lim = [None, None, (0, 2), (-2, 0), (0, 10)]
y_lim = [(0, 1), (0, 1), (0, 2), (-2, 0), (0, 10)]
bad = [0.25, 0.25, 0.3, 0.3, 2]
markers = np.array(['o', 'x'])

for i in range(n_param):
    # mark fits that are especially bad
    bad_fit = np.where(
        np.abs(final_params[:, i] - final_params[:, i+n_param]) > bad[i], 1, 0)
    plt.figure(figsize=(7, 5), dpi=300)
    print(np.sum(bad_fit))
    # separately plot different groups
    for i_m, m in enumerate(markers):
        index = (bad_fit == i_m)
        plt.scatter(final_params[index, i], final_params[index, i+n_param],
                    c=final_std_err[index, i], cmap=cmap, marker=m)
        plt.plot(
            np.linspace(y_lim[i][0], y_lim[i][1], 10),
            np.linspace(y_lim[i][0], y_lim[i][1], 10),
            linewidth=1, color='black')
        plt.xlim(x_lim[i])
        plt.ylim(y_lim[i])

    cbar = plt.colorbar()
    cbar.set_label('std error')
    plt.xlabel(f'true {param_names[i]}')
    plt.ylabel(f'estimated {param_names[i]}')
    corr = np.round(np.corrcoef(final_params[:, i],
                                final_params[:, i+n_param])[0, 1], 3)
    plt.title(f'corr = {corr}',
              color='red')

# %%
# how do the histograms and pairwise correlations look like
#: for input params kept and output params

for i in range(n_param):
    plt.figure(figsize=(4, 4), dpi=300)
    plt.hist(final_params[:, i+n_param])
    plt.xlabel(param_names[i])
    plt.ylabel('frequency')


for i in range(n_param):
    for j in range(i+1, 4):
        plt.figure(figsize=(4, 4), dpi=300)
        plt.scatter(final_params[:, i+n_param], final_params[:, j+n_param])
        plt.xlabel(param_names[i])
        plt.ylabel(param_names[j])
