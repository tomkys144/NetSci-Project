import graph_tool.all as gt
from graph_tool.draw import BlockState
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


def draw_sbm(state, outputFile: str):
    state.draw(
        layout="radial",  # Forces radial/circular layout
        beta=0.9,  # High bundling makes radial structure clearer
        output=outputFile,
        output_size=(1500, 1500),  # Larger size prevents overlapping text/nodes
    )

def draw_sbm_shape(state: gt.NestedBlockState, brainNet: BrainNet, outputFile: str, coords=(0,1), layer=0):
    pstate = state.project_level(layer)

    pos3d = brainNet.gtGraph.vp.pos.get_2d_array(pos=[0, 1, 2])
    pos2d = pos3d[list(coords), :]

    pos = brainNet.gtGraph.new_vertex_property("vector<double>")
    pos.set_2d_array(pos2d)

    pstate.draw(
        pos=pos,
        output=outputFile,
        output_size=(1500, 1500),
    )
