import graph_tool.all as gt
import numpy as np

from brainNet import BrainNet


def sbm(brainNet: BrainNet, nmcmc=100):
    if not hasattr(brainNet, "gtGraph"):
        brainNet.get_gt()

    state = gt.minimize_nested_blockmodel_dl(
        brainNet.gtGraph,
    )

    print("Calculating MCMC...")
    for i in range(nmcmc):
        if (i / nmcmc * 100) % 1 == 0:
            print("Progress: ", (i / nmcmc * 100), "%")
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
        
    state.print_summary()

    return state


def draw_sbm(state, outputFile: str = ""):
    state.draw(
        output=outputFile
    )


def draw_sbm_shape(state, brainNet: BrainNet, outputFile: str = "", coords=(0,1)):
    lstate = state.get_levels()[0]

    pos3d = brainNet.gtGraph.vp.pos.get_2d_array(pos=[0, 1, 2])
    pos2d = pos3d[list(coords), :]

    pos = brainNet.gtGraph.new_vertex_property("vector<double>")
    pos.set_2d_array(pos2d)

    lstate.draw(
        pos=pos,
        output=outputFile
    )
