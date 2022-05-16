from utils.data import get_train_test_img_dataset
from utils.visualize import plot_dataset_samples_imgs
from matplotlib.pyplot import subplots, subplot, figure, show, subplots_adjust, scatter
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern, WhiteKernel
from utils.ntbks_helpers import get_gp_datasets
from numpy import reshape, max, min
from utils.visualize import plot_dataset_samples_1d
from pandas import DataFrame
"""
datasets = dict()
_, datasets["CelebA32"] = get_train_test_img_dataset("celeba32")

n_datasets = len(datasets)
fig, axes = subplots(1, n_datasets, figsize=(5 * n_datasets, 5))
for i, (k, dataset) in enumerate(datasets.items()):
    plot_dataset_samples_imgs(dataset, title=k, ax=axes[i])
"""


kernels = dict()
print("###########################################")
print("0: RBF    1: Periodic_Kernel    2: Noisy_Matern_Kernel")
inp = input("wähle Option:").strip()
if(int(inp)==0):
    kernels["RBF_Kernel"] = RBF(length_scale=(0.2))
elif(int(inp)==1):
    kernels["Periodic_Kernel"] = ExpSineSquared(length_scale=1, periodicity=0.5)
else:
    kernels["Noisy_Matern_Kernel"] = WhiteKernel(noise_level=0.1) + Matern(length_scale=0.2, nu=1.5)

print(kernels.keys())
N_sample =5000
N_point =128
file_name = "./x_data.csv"
(datasets, _, __,) = get_gp_datasets(
        kernels,
        is_vary_kernel_hyp=False,  # use a single hyperparameter per kernel
        n_samples=N_sample,  # number of different context-target sets
        n_points=N_point,  # size of target U context set for each sample
        is_reuse_across_epochs=False,  # never see the same example twice
    )
key = list(kernels.keys())
context, target = datasets[key[0]].get_samples()
c_data = reshape(context.numpy(), (N_sample,N_point))
t_data = reshape(target.numpy(), (N_sample,N_point))
print("max", max(c_data), "min",min(c_data), "max ->",max(t_data), "min ->", min(t_data) )
c_df = DataFrame(c_data)
c_df.to_csv(file_name, sep='\t', encoding='utf-8')


n_datasets = len(datasets)
fig, axes = subplots(n_datasets, 1, figsize=(11, 5 * n_datasets), squeeze=False)
for i, (k, dataset) in enumerate(datasets.items()):
    plot_dataset_samples_1d(dataset, title=k.replace("_", " "), ax=axes.flatten()[i], n_samples=4)
show()