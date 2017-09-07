###This file intends to demonstrate the rendering and basically follows basic.jl
import Cxx
import DifferentialDynamicProgramming
import Gen
import MuJoCo

using Cxx;
@everywhere using Gen;
using MuJoCo;
using MuJoCo.time;

include("../mjderivs.jl")

mj_activate()

reaching = true
N = 180

function linear_planner_constant(startPos, destPos)
    dir = destPos - startPos
    speed = 5 * .01
    pnt = dir / norm(dir) * speed + startPos

    if speed > norm(dir)
        pnt = destPos
    end

    return pnt
end

@program move_hand_to_destination(planner::Function, sim::MJSimEnv, path_length = N) begin
    handID = mj_name2id(sim.model, icxx"mjOBJ_SITE;", "right_hand")

    mjStart = sim.data.site_xpos[handID, :]

    mv_covar = [1. 0 0 ; 0 1. 0; 0 0 1.] * .01

    startPos = @tag(mvnormal(mjStart, mv_covar), "startPos")

    destinationObject = @tag([uniform_discrete(1, 4), uniform_discrete(1, 4)], "destinationObject")

    targetID = mj_name2id(sim.model, icxx"mjOBJ_SITE;", "target$(destinationObject[1])-$(destinationObject[2])")
    mjObjectPosition = sim.data.site_xpos[targetID, :]

    destinationPos = @tag(mvnormal(mjObjectPosition, mv_covar), "destinationPos")

    #move along path (plus some noise)
    lastPos = startPos
    for i=1:path_length
        nextPos = @tag(mvnormal(planner(lastPos, destinationPos), mv_covar), "pathPos$i")
    end
end

xmlfile = joinpath(@__DIR__, "../../reaching.xml")
model = MJModel(xmlfile)

w_dist = 1
w_vel = .1
w_act = .01

function intermediateCost(x, u, sim, destPos)
    s = 0
    for i=1:length(u)
        s += w_act * u[i] ^ 2
    end

    for i=1:sim.data.nv
        s += w_vel * x[i + sim.data.nq] ^ 2
    end

    handID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "right_hand") - 1)
    s += w_dist * sum((x[handID:(handID+2)] - destPos) .^ 2)

    return s
end

function costx(x, sim, destPos)
    cx = zeros(size(x))
    for i=1:sim.data.nv
        cx[i + sim.data.nq] = 2 * w_vel * x[i + sim.data.nq]
    end

    handID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "right_hand") - 1)
    cx[handID:(handID + 2)] = w_dist * 2 * (x[handID:(handID + 2)] - destPos)

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

    handID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "right_hand") - 1)
    cxx[handID:(handID + 2), handID:(handID + 2)] = w_dist * 2 * ones(3, 3)

    return cxx
end

function costuu(u, sim)
    return w_act * 2 * eye(sim.controls.nu)
    #return ones(sim.controls.nu, sim.controls.nu)
end

function finalCost(x, sim, destPos)
    s = 0

    for i=1:sim.data.nv
        s += w_vel * x[i + sim.data.nq] ^ 2
    end

    handID = sim.data.nq + sim.data.nv * 2 + 3 * (mj_name2id(sim.model, icxx"mjOBJ_SITE;", "right_hand") - 1)
    s += w_dist * sum((x[handID:(handID+2)] - destPos) .^ 2)

    return s
end


sim = MJSimEnv{SimState, AltControls}(model, s -> (0), s -> (false), SimState(), AltControls())
ResetEnv(sim)
if (!reaching)
    for i in 1:50
        mj_step(sim.model, sim.data)
    end
end

xs = zeros(sim.state.nx, N)
us = zeros(sim.controls.nu, N)

traj = Trace()
intervene!(traj, "destinationObject", [2, 2])
@generate(traj, move_hand_to_destination(linear_planner_constant, sim, N))

x0 = GetSimState(sim)
u0 = rand(sim.controls.nu, 1) * 10 - 5

print("Performing the optimization")
for it in 1:N
    print('.')

    destPos = value(traj, "pathPos$it")

    ic(x, u, s) = intermediateCost(x, u, s, destPos)
    fc(x, s) = finalCost(x, s, destPos)
    cx(x, s) = costx(x, s, destPos)

    f(x, u, i) = mjDynamicsF(x, u, i, sim, ic)
    fT(x) = mjDynamicsFT(x, sim, fc)
    fX(x, u, I) = mjDynamicsFx(x, u, I, sim, cx, costu, costxx, costuu, true)

    ls=zeros(sim.controls.nu, 2)
    for i in 1:sim.controls.nu
        ls[i,:] = [-1, 1] * 5
    end

    xn = zeros(size(x0,1),2)
    un = nothing

    #xn, un, _, __, ___, ____, _____ =
    ##DifferentialDynamicProgramming.iLQG(f, fT, fX, x0, u0, tolFun=1e-7, tolGrad=1e-4, lambdaMax=1e5,verbosity=0)#,lims=ls)

    xnext = nothing
    alpha = .1
    err = 1
    while err > .1
        fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu = fX(xn[:, 1], u0, [1])

        xnext = f(x0, u0, 1)[1]
        inverse = ones(size(cxx))
        try
            println(cxx)
            inverse = inv(cxx / 2)
        catch
            break
        end
        xopt = 1/2 * (cxx * xnext - cx[:, 1]) * inverse

        derr = (xnext - xopt)
        err = derr'derr

        println(err)

        du = alpha * 2 * fu' * derr

        u0 += du
        xn = [x0, xnext]
    end
    un = u0

    xs[:, it] = xn[:, 1]
    us[:, it] = un
    x0 = xn[:, 1]
    u0 = un
end

#save to file
writecsv("u2.csv", us)
writecsv("x2.csv", xs)

println("Done!")
