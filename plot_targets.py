import torch

from utils.ntbks_helpers import *
from npf.utils.datasplit import *
import numpy as np
class GetIndcs:
    """
    Return random subset of indices.

    Parameters
    ----------
    a : float or int, optional
        Minimum number of indices. If smaller than 1, represents a percentage of
        points.

    b : float or int, optional
        Maximum number of indices. If smaller than 1, represents a percentage of
        points.

    is_batch_share : bool, optional
        Whether to use use the same indices for all elements in the batch.

    """

    def __init__(
            self,
            start=0.5,
            end=0.6,
            data=None
    ):
        self.a = start
        self.b =end
        self.data = data

    def __call__(self):
        possible_points =None
        if self.data is not None:
            possible_points = self.data.numpy()
        indcs = np.where((self.a<possible_points) & (possible_points<self.b))
        return torch.from_numpy(indcs[1])
def input_coordinates(x, flag=True):

    if(x==" "):
        print("No Value was entered!. Defect Value None assigned")
        return None
    else:
        X = [x]
        X=np.asarray(X,dtype=np.float32)
        if(flag is True):
            X = reducer(X)
        return torch.from_numpy(X)
def reducer(arr):
    for i in range(arr.shape[1]):
        if(arr[0,i,0]<=0.0):
            pass
        elif(arr[0,i,0]==1.):
            arr[0,i,0] =0.75
        else:
            arr[0,i,0]-=0.25
    return arr
def get_n_trgt(n_trgt, is_1d=True, upscale_factor=1):
    """Return a context target splitter with a fixed number of context points."""
    if is_1d:
        return CntxtTrgtGetter(
            contexts_getter=GetRandomIndcs(a=n_trgt, b=n_trgt),
            targets_getter=get_all_indcs,
            is_add_cntxts_to_trgts=False,
        )

def plot_multi_posterior_trgts_1d(
    trainers,
    datasets,
    n_trgt,
    trainers_compare=None,
    plot_config_kwargs={},
    title="Model : {model_name} | Data : {data_name} | Num. Context : {n_trgt}",
    left_extrap=0,
    right_extrap=0,
    pretty_renamer=PRETTY_RENAMER,
    is_plot_generator=True,
    imgsize=(8, 3),
    **kwargs,
):
    """Plot posterior samples conditioned on `n_trgt` context points for a set of trained trainers."""

    with plot_config(**plot_config_kwargs):
        n_trainers = len(trainers)

        n_col = 1 if trainers_compare is None else 2
        fig, axes = plt.subplots(
            n_trainers,
            n_col,
            figsize=(imgsize[0] * n_col, imgsize[1] * n_trainers),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        xx = []
        yy = []
        try:
            while True:
                _x = []
                _y = []
                x = input("enter X Data:")
                y = input("enter Y Data:")

                _x.append(float(x))
                _y.append(y)
                xx.append(_x)
                yy.append(_y)
        except KeyboardInterrupt as k:
            print(k)
            pass
        print(xx, yy)
        for j, curr_trainers in enumerate([trainers, trainers_compare]):
            if curr_trainers is None:
                continue

            for i, (k, trainer) in enumerate(curr_trainers.items()):
                data_name = k.split("/")[0]
                model_name = k.split("/")[1]
                dataset = datasets[data_name]
                if title is not None:
                    curr_title = title.format(
                        model_name=pretty_renamer[model_name],
                        n_trgt=n_trgt,
                        data_name=pretty_renamer[data_name],
                    )
                else:
                    curr_title = None

                test_min_max = dataset.min_max
                if (left_extrap != 0) or (right_extrap != 0):
                    test_min_max = (
                        dataset.min_max[0] - left_extrap,
                        dataset.min_max[1] + right_extrap,
                    )
                    trainer.module_.set_extrapolation(test_min_max)

                X, Y = dataset.get_samples(
                    n_samples=1,
                    n_points=3 * dataset.n_points,
                    test_min_max=test_min_max,
                )  # use higher density for plotting
                index = GetIndcs(start=0.2, end=0.21, data=X)
                tensor = index()

                plot_posterior_samples_1d(
                    X,
                    Y,
                    get_n_trgt(n_trgt),
                    trainer.module_,
                    generator=dataset.generator if is_plot_generator else None,
                    train_min_max=dataset.min_max,
                    title=curr_title,
                    ax=axes[i, j],
                    data=tensor,
                    xx=input_coordinates(xx, flag=False),
                    yy=input_coordinates(yy, flag=False),
                    scatter_label="Context Set",
                    **kwargs,
                )

        plt.tight_layout()
        return fig

# TO DO : docstrings
def plot_multi_posterior_samples_imgs(
    trainers,
    datasets,
    n_cntxt,
    plot_config_kwargs={},
    title="{model_name} | {data_name} | C={n_cntxt}",
    pretty_renamer=PRETTY_RENAMER,
    n_plots=4,
    figsize=(3, 3),
    is_superresolution=False,
    **kwargs,
):
    with plot_config(**plot_config_kwargs):
        n_trainers = len(trainers)
        fig, axes = plt.subplots(
            1,
            n_trainers,
            figsize=(figsize[0] * n_plots, figsize[1] * n_trainers),
            squeeze=False,
        )
        xx = []
        yy = []
        try:
            while True:
                _x = []
                _y = []
                x = input("enter X Data:")
                y = input("enter Y Data:")

                _x.append(float(x))
                _y.append(y)
                xx.append(_x)
                yy.append(_y)
        except KeyboardInterrupt as k:
            print(k)
            pass
        for i, (k, trainer) in enumerate(trainers.items()):
            data_name = k.split("/")[0]
            model_name = k.split("/")[1]
            dataset = datasets[data_name]

            if isinstance(n_cntxt, float) and n_cntxt < 1:
                if is_superresolution:
                    n_cntxt_title = f"{int(dataset.shape[1]*n_cntxt)}x{int(dataset.shape[2]*n_cntxt)}"
                else:
                    n_cntxt_title = f"{100*n_cntxt:.1f}%"
            elif isinstance(n_cntxt, str):
                n_cntxt_title = pretty_renamer[n_cntxt]
            else:
                n_cntxt_title = n_cntxt

            curr_title = title.format(
                model_name=pretty_renamer[model_name],
                n_cntxt=n_cntxt_title,
                data_name=pretty_renamer[data_name],
            )
            print("N context ->",n_cntxt)
            upscale_factor = get_test_upscale_factor(data_name)
            if n_cntxt in ["vhalf", "hhalf"]:
                cntxt_trgt_getter = GridCntxtTrgtGetter(
                    context_masker=partial(
                        half_masker, dim=0 if n_cntxt == "hhalf" else 1
                    ),
                    upscale_factor=upscale_factor,
                )
            elif is_superresolution:
                cntxt_trgt_getter = SuperresolutionCntxtTrgtGetter(
                    resolution_factor=n_cntxt, upscale_factor=upscale_factor
                )
            else:
                cntxt_trgt_getter = get_n_cntxt(
                    n_cntxt, is_1d=False, upscale_factor=upscale_factor
                )
            print(" Data ->", xx, yy)
            plot_posterior_samples(
                dataset,
                cntxt_trgt_getter,
                trainer.module_.cpu(),
                is_uniform_grid=isinstance(trainer.module_, GridConvCNP),
                ax=axes.flatten()[i],
                n_plots=n_plots if not isinstance(dataset, SingleImage) else 1,
                is_mask_cntxt=not is_superresolution,
                **kwargs,
            )
            axes.flatten()[i].set_title(curr_title)

    return fig

