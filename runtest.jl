using MuJoCo
using MuJoCo.time
using Cxx
using Base.Test

include("src/simenvi.jl")
include("src/ilqg.jl")

# this example is a simplified version of samples/basic.cpp from mujoco



function intermediateCost(x, u)
    sum = 0
    for i=1:length(u)
        sum += u[i] ^ 2
    end
    return sum
end

function finalCost(x)
    handID = mj_name2id(sim.model, icxx"mjOBJ_BODY;", "right_hand")
    targetID = mj_name2id(sim.model, icxx"mjOBJ_BODY;", "target1-1")
    return (1.0e4 * distance(sim.data.xpos[handID, :], sim.data.xpos[targetID, :]))
end

mj_activate()
xmlfile = joinpath(@__DIR__, "reaching.xml")
model = MJModel(xmlfile)
sim = MJSimEnv(model, s -> (0), s -> (false))
x0 = GetSimState(sim)
u0 = GetSimControls(sim)

f(x, u, i) = mjDynamicsF(x, u, i, sim, intermediateCost)
fT(x) = mjDynamicsFT(x, sim, finalCost)
fX(x, u, I) = mjDynamicsFx(x, u, I, sim)

torso_id = mj_name2id(model, icxx"mjOBJ_BODY;", "torso")

# run main loop, target real-time simulation and 60 fps rendering
for i=1:10
    # advance interactive simulation for 1/60 sec
    #  Assuming MuJoCo can simulate faster than real-time, which it usually can,
    #  this loop will finish on time for the next frame to be rendered at 60 fps.
    #  Otherwise add a cpu timer and exit this loop when it is time to render.
    simstart = time(sim.data)
    while  time(sim.data) - simstart < 1.0/60.0

        # run a step of the simulation
        mj_step(model, sim.data)
    end

    # print the torso world coordinates
    torso_xpos = sim.data.xpos[torso_id,:]
    println("$i, $(time(sim.data)), torso_xpos: $torso_xpos")
end

# free MuJoCo model and sim.data, deactivateFloat64
mj_deleteData(sim.data)
mj_deleteModel(model)
mj_deactivate()
