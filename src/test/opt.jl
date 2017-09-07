###This file intends to demonstrate the rendering and basically follows basic.jl
import Cxx
import DifferentialDynamicProgramming
import MuJoCo


using Cxx
using MuJoCo
using MuJoCo.time

include("../mjderivs.jl")

mj_activate()

reaching = true
N = 180

xmlfile = nothing
if reaching
    xmlfile = joinpath(@__DIR__, "../../reaching.xml")
else
    xmlfile = joinpath(@__DIR__, "../../cartpoleBalance.xml")
end

model = MJModel(xmlfile)
target = "2-1"

w_dist = 10
w_vel = 1
w_act = 1

function intermediateCost(x, u, sim)
    s = 0
    for i=1:length(u)
        s += w_act * u[i] ^ 2
    end

    for i=1:sim.data.nv
        s += w_vel * x[i + sim.data.nq] ^ 2
    end

    if reaching
        handID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "right_hand") - 1)
        targetID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "target$target") - 1)
        s += w_dist * sum((x[handID:(handID + 2)] - x[targetID:(targetID + 2)]) .^ 2)
    else
        angle = mj_name2id(sim.model, icxx"mjOBJ_JOINT;", "hinge")
        s += (sim.data.qpos[angle, 1] - pi) ^ 2
    end
    return s
end

function costx(x, sim)
    cx = zeros(size(x))
    for i=1:sim.data.nv
        cx[i + sim.data.nq] = 2 * w_vel * x[i + sim.data.nq]
    end

    if reaching
        handID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "right_hand") - 1)
        targetID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "target$target") - 1)
        cx[handID:(handID + 2)] = w_dist * 2 * (x[handID:(handID + 2)] - x[targetID:(targetID + 2)])
        #cx[targetID:(targetID + 2)] = 2 * (x[handID:(handID + 2)] - x[targetID:(targetID + 2)])
    else
        angle = mj_name2id(sim.model, icxx"mjOBJ_JOINT;", "hinge")
        cx[angle] = 2 * (sim.data.qpos[angle, 1] - pi)
    end
    return cx
end

function costu(u, sim)
    return w_act * 2 * u
    #return ones(size(u))
end

function costxx(x, sim)
    cxx = zeros(size(x, 1), size(x, 1))
    for i=(sim.data.nq + 1):(sim.data.nq + sim.data.nv)
        cxx[i, i] = 2 * w_vel
    end

    if reaching
        handID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "right_hand") - 1)
        targetID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "target$target") - 1)
        cxx[handID:(handID + 2), handID:(handID + 2)] = w_dist * 2 * ones(3, 3)
        #cxx[targetID:(targetID + 2), targetID:(targetID + 2)] = 2 * ones(3, 3)
    else
        angle = mj_name2id(sim.model, icxx"mjOBJ_JOINT;", "hinge")
        cxx[angle, angle] = 2
    end

    return cxx
end

function costuu(u, sim)
    return w_act * 2 * eye(sim.controls.nu)
    #return ones(sim.controls.nu, sim.controls.nu)
end

function finalCost(x, sim)
    if reaching
        handID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "right_hand") - 1)
        targetID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "target$target") - 1)
        return sum((x[handID:(handID + 2)] - x[targetID:(targetID + 2)]) .^ 2)
    else
        angle = mj_name2id(sim.model, icxx"mjOBJ_JOINT;", "hinge")
        return 100 * abs(sim.data.qpos[angle, 1] - pi)
    end
end


sim = MJSimEnv{SimState, AltControls}(model, s -> (0), s -> (false), SimState(), AltControls())
ResetEnv(sim)
if (!reaching)
    for i in 1:50
        mj_step(sim.model, sim.data)
    end
end

x0 = GetSimState(sim)
u0 = randn(sim.controls.nu, N) * 10 - 5

f(x, u, i) = mjDynamicsF(x, u, i, sim, intermediateCost)
fT(x) = mjDynamicsFT(x, sim, finalCost)
fX(x, u, I) = mjDynamicsFx(x, u, I, sim, costx, costu, costxx, costuu, true)

x = u = nothing

ls=zeros(sim.controls.nu, 2)
for i in 1:sim.controls.nu
    ls[i,:] = [-1, 1]
end

print("Performing the optimization...")

x, u, _, __, ___, ____, _____ =
DifferentialDynamicProgramming.iLQG(f, fT, fX, x0, u0, tolFun=1, tolGrad=1e-4, lambdaMax=1e5, verbosity=0)

#save to file
writecsv("u.csv", u)
writecsv("x.csv", x)

println("Done!")
