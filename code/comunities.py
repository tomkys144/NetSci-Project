import graph_tool.all as gt
import numpy as np

from brainNet import BrainNet


def sbm(brainNet: BrainNet, nmcmc=100):
    if not hasattr(brainNet, "gtGraph"):
        brainNet.get_gt()

    state = gt.minimize_nested_blockmodel_dl(
        brainNet.gtGraph,
        state_args=dict(
            recs=[brainNet.gtGraph.ep.avgRadiusAvg],
            rec_types=["discrete-binomial"]
        )
    )

    print("Calculating MCMC...")
    for i in range(nmcmc):  # this should be sufficiently large
        # if (i / nmcmc * 100) % 1 == 0:
            # print("Progress: ", (i / nmcmc * 100), "%")
        state.multilevel_mcmc_sweep(beta=np.inf, niter=10)

    return state


def draw_sbm(state, outputFile: str = ""):
    state.draw(
        output=outputFile
    )