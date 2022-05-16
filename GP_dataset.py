
import logging
import os
import warnings

import matplotlib.pyplot as plt
import torch

os.chdir("../..")

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.disable(logging.ERROR)

N_THREADS = 8
IS_FORCE_CPU = False  # Nota Bene : notebooks don't deallocate GPU memory

if IS_FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

torch.set_num_threads(N_THREADS)

from utils.ntbks_helpers import get_all_gp_datasets, get_img_datasets
from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    get_all_indcs
)
from utils.data import cntxt_trgt_collate
from functools import partial
from npf import CNP
from npf.architectures import MLP, merge_flat_input
from utils.helpers import count_parameters
from npf import CNPFLoss
from utils.train import train_models
from utils.ntbks_helpers import PRETTY_RENAMER, plot_multi_posterior_samples_1d, plot_multi_posterior_samples_imgs
from utils.visualize import giffify, plot_config
from plot_targets import plot_multi_posterior_trgts_1d
# DATASETS
# merges : get_datasets_single_gp, get_datasets_varying_hyp_gp, get_datasets_varying_kernel_gp
gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()

print("GP dataset ->", gp_datasets)
print("GP test ->", gp_test_datasets)
print("GP valid ->", gp_valid_datasets)

##############################################################
##################### CONTEXT TARGET SPLIT ##################
############################################################
# 1d
get_cntxt_trgt_1d = cntxt_trgt_collate(
    CntxtTrgtGetter(
        contexts_getter=GetRandomIndcs(a=0.0, b=50), targets_getter=get_all_indcs,
    )
)

#########################################################################################
############################### DEFINE ENCODERS (STRUCTUR) ###############################
#########################################################################################
R_DIM = 128
KWARGS = dict( ##
    XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
    Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
        partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
    ),
    r_dim=R_DIM,
)

# 1D case
model_1d = partial(
    CNP, ##
    x_dim=1,
    y_dim=1,
    XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True,
    ),
    **KWARGS,
)
n_params_1d = count_parameters(model_1d())
print(f"Number Parameters (1D): {n_params_1d:,d}")
#######################################################################################
############################ TRAINING #################################################
#######################################################################################
KWARGS = dict(
    is_retrain=False,  # whether to load precomputed model or retrain
    criterion=CNPFLoss,  # Standard loss for conditional NPFs ##
    chckpnt_dirname="/home/cristian/PycharmProjects/Neural-Process-Family/results/pretrained/",
    device=None,  # use GPU if available
    batch_size=32,
    lr=1e-3,
    decay_lr=10,  # decrease learning rate by 10 during training
    seed=123,
)

# scp -rp /home/cristian/PycharmProjects/Neural-Process-Family/ gucr@cuneus.informatik.tu-chemnitz.de:/scratch/gucr2/
# 1D
trainers_1d = train_models(
    gp_datasets,
    {"CNP": model_1d},
    test_datasets=gp_test_datasets,
    train_split=None,  # No need of validation as the training data is generated on the fly
    iterator_train__collate_fn=get_cntxt_trgt_1d,
    iterator_valid__collate_fn=get_cntxt_trgt_1d,
    max_epochs=10,
    **KWARGS
)

print("1D", type(trainers_1d), len(trainers_1d))
def filter_single_gp(d):
    """Select only data form single GP."""
    return {k: v for k, v in d.items() if ("All" not in k) and ("Variable" not in k)}

trainers=filter_single_gp(trainers_1d)
#print("results ->", trainers.values())
"""
def multi_posterior_gp_gif(filename, trainers, datasets, seed=123, **kwargs):
    giffify(
        save_filename=f"/home/cristian/PycharmProjects/Neural-Process-Family/jupyter/gifs/{filename}.gif",
        gen_single_fig=plot_multi_posterior_samples_1d,  # core plotting
        sweep_parameter="n_cntxt",  # param over which to sweep
        sweep_values=[1,10, 15, 20, 30, 50,100,200],
        fps=1.,  # gif speed
        # PLOTTING KWARGS
        trainers=trainers,
        datasets=datasets,
        is_plot_generator=True,  # plot underlying GP
        is_plot_real=False,  # don't plot sampled / underlying function
        is_plot_std=True,  # plot the predictive std
        is_fill_generator_std=False,  # do not fill predictive of GP
        pretty_renamer=PRETTY_RENAMER,  # pretiffy names of modulte + data
        # Fix formatting for coherent GIF
        plot_config_kwargs=dict(
            set_kwargs=dict(ylim=[-3, 3]), rc={"legend.loc": "upper right"}
        ),
        seed=seed,
        **kwargs,
    )
multi_posterior_gp_gif(
    "CNP_single_gp",
    trainers=filter_single_gp(trainers_1d),
    datasets=filter_single_gp(gp_test_datasets),
)"""
def multi_posterior_gp_plot(filename, trainers, datasets, seed=123, **kwargs):
    giffify(
        save_filename=f"/home/cristian/PycharmProjects/Neural-Process-Family/jupyter/gifs/{filename}.gif",
        gen_single_fig=plot_multi_posterior_trgts_1d,  # core plotting
        sweep_parameter="n_trgt",  # param over which to sweep
        sweep_values=[1],
        fps=1.,  # gif speed
        # PLOTTING KWARGS
        trainers=trainers,
        datasets=datasets,
        is_plot_generator=True,  # plot underlying GP
        is_plot_real=False,  # don't plot sampled / underlying function
        is_plot_std=True,  # plot the predictive std
        is_fill_generator_std=False,  # do not fill predictive of GP
        pretty_renamer=PRETTY_RENAMER,  # pretiffy names of modulte + data
        # Fix formatting for coherent GIF
        plot_config_kwargs=dict(
            set_kwargs=dict(ylim=[-3, 3]), rc={"legend.loc": "upper right"}
        ),
        seed=seed,
        **kwargs,
    )

multi_posterior_gp_plot(
    "CNP_single_gp",
    trainers=filter_single_gp(trainers_1d),
    datasets=filter_single_gp(gp_test_datasets),
)