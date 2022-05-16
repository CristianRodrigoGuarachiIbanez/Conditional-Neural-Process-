from utils.data import cntxt_trgt_collate, get_test_upscale_factor
from functools import partial
from npf import CNP
from npf.architectures import MLP, merge_flat_input
from utils.helpers import count_parameters
# train_models Docstring
import skorch
from npf import CNPFLoss
from utils.ntbks_helpers import add_y_dim
from utils.train import train_models
from utils.ntbks_helpers import PRETTY_RENAMER#, plot_multi_posterior_samples_imgs
from utils.visualize import giffify
from utils.ntbks_helpers import get_img_datasets
from npf.utils.datasplit import (
    GridCntxtTrgtGetter,
    RandomMasker,
    no_masker,
)
# image datasets
img_datasets, img_test_datasets = get_img_datasets(["celeba32"])

print("img dataset ->", img_datasets)
print("img test dataset ->", img_test_datasets)


##############################################################
##################### CONTEXT TARGET SPLIT ##################
############################################################
# same as in 1D but with masks (2d) rather than indices
get_cntxt_trgt_2d = cntxt_trgt_collate(
    GridCntxtTrgtGetter(
        context_masker=RandomMasker(a=0.0, b=0.3), target_masker=no_masker,
    )
)
# for ZSMMS you need the pixels to not be in [-1,1] but [-1.75,1.75] (i.e 56 / 32) because you are extrapolating
get_cntxt_trgt_2d_extrap = cntxt_trgt_collate(
    GridCntxtTrgtGetter(
        context_masker=RandomMasker(a=0, b=0.3),
        target_masker=no_masker,
        upscale_factor=get_test_upscale_factor("zsmms"),
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

# image (2D) case
model_2d = partial(
    CNP, ##
    x_dim=2,
    XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 3), is_sum_merge=True,
    ),
    **KWARGS,
)  # don't add y_dim yet because depends on data (colored or gray scale)

n_params_2d = count_parameters(model_2d(y_dim=3))
print(f"Number Parameters (2D): {n_params_2d:,d}")
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

# 2D
trainers_2d = train_models(
    img_datasets,
    add_y_dim({"CNP": model_2d}, img_datasets),  # y_dim (channels) depend on data
    test_datasets=img_test_datasets,
    train_split=skorch.dataset.CVSplit(0.1),  # use 10% of training for valdiation
    iterator_train__collate_fn=get_cntxt_trgt_2d,
    iterator_valid__collate_fn=get_cntxt_trgt_2d,
    datasets_kwargs=dict(
        zsmms=dict(iterator_valid__collate_fn=get_cntxt_trgt_2d_extrap,)
    ),  # for zsmm use extrapolation
    max_epochs=10,
    **KWARGS
)
# don't add y_dim yet because depends on data (colored or gray scale)
from plot_targets import plot_multi_posterior_samples_imgs
def multi_posterior_imgs_gif(filename, trainers, datasets, seed=123, **kwargs):
    giffify(
        save_filename=f"/home/cristian/PycharmProjects/Neural-Process-Family/jupyter/gifs/{filename}_",
        gen_single_fig=plot_multi_posterior_samples_imgs,  # core plotting
        sweep_parameter="n_cntxt",  # param over which to sweep
        sweep_values=[  0],
        fps=1.,  # gif speed
        # PLOTTING KWARGS
        trainers=trainers,
        datasets=datasets,
        n_plots=3,  # images per datasets
        is_plot_std=True,  # plot the predictive std
        pretty_renamer=PRETTY_RENAMER,  # pretiffy names of modulte + data
        plot_config_kwargs={"font_scale":0.7},
        # Fix formatting for coherent GIF
        seed=seed,
        **kwargs,
    )

def filter_interpolation(d):
    """Filter out zsmms which requires extrapolation."""
    return {k: v for k, v in d.items() if "zsmms" not in k}


multi_posterior_imgs_gif(
    "CNP_img_interp",
    trainers=filter_interpolation(trainers_2d),
    datasets=filter_interpolation(img_test_datasets),
)