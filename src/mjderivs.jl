import Cxx
import MuJoCo

import Base.getindex
import Base.setindex!

using Cxx
using MuJoCo

include("simenvi.jl")

##STATE

struct SimState
    mjStates::Vector{MJMatrix}
    nx::Int64

    function SimState()
        return new([], 0)
    end

    function SimState(data::MJData)
        l = [data.qpos, data.qvel, data.qacc, data.site_xpos]
        return new(l, data.nq + 2 * data.nv + data.nsite * 3)
    end

end

function getindex(state::SimState, i)
    targList = 1
    while length(state.mjStates[targList]) < i
        i -= length(state.mjStates[targList])
        targList += 1
    end

    return state.mjStates[targList].mat[i]
end

function getindex(state::SimState, l, i, j)
    return state.mjStates[l][i, j]
end

function setindex!(state::SimState, val, i)
    targList = 1
    while length(state.mjStates[targList]) < i
        i -= length(state.mjStates[targList])
        targList += 1
    end

    state.mjStates[targList].mat[i] = val
end

function setindex!(state::SimState, val, l, i, j)
    state.mjStates[l].mat[j, i] = val
end

##CONTROL

#struct AltControls
#    nu::Int64
#    ctrl::MJMatrix
#    normalized::Vector{mjtNum}
#    ctrl_magnitude::MJMatrix
#
#    function AltControls(data::MJData)
#        mat = data.qfrc_applied
#        norm = []
#        rng = MJMatrix(Matrix{mjtNum}(0, 0))
#
#        new(data.nv, mat, norm, rng)
#    end
#
#    function AltControls()
#        return new(0, MJMatrix(Matrix{mjtNum}(0, 0)), [], MJMatrix(Matrix{mjtNum}(0, 0)))
#    end
#end
#
#function SetSimControls(ctrl::AltControls, norm::Vector{mjtNum})
#    for i=1:ctrl.nu
#        ctrl.ctrl[i] = norm[i]
#    end
#
#    return ctrl
#end
#
#function getitem(ctrl::AltControls, i)
#    return ctrl.ctrl[i, 1]
#end
#
#function setitem!(ctrl::AltControls, val, i)
#    ctrl.ctrl.mat[1, i] = val
#end
#

struct AltControls
    nu::Int64
    ctrl::MJMatrix
    normalized::Vector{mjtNum}
    ctrl_magnitude::MJMatrix

    function AltControls(data::MJData)
        mat = data.qpos
        norm = []
        rng = MJMatrix(Matrix{mjtNum}(0, 0))

        new(data.nq, mat, norm, rng)
    end

    function AltControls()
        return new(0, MJMatrix(Matrix{mjtNum}(0, 0)), [], MJMatrix(Matrix{mjtNum}(0, 0)))
    end
end

function SetSimControls(ctrl::AltControls, norm)
    for i=1:ctrl.nu
        ctrl.ctrl[i] = norm[i]
    end

    return ctrl
end

function getitem(ctrl::AltControls, i)
    return ctrl.ctrl[i, 1]
end

function setitem!(ctrl::AltControls, val, i)
    ctrl.ctrl.mat[1, i] = val
end


##Accessories to the SimEnv

function SetSimState(sim::MJSimEnv, xi)
    for i=1:length(xi)
        sim.state[i] = xi[i]
    end
end

function GetSimState(sim::MJSimEnv)
    xi = Matrix{mjtNum}(sim.state.nx, 1)
    for i=1:sim.state.nx
        xi[i] = sim.state[i]
    end
    return xi
end

##CONTROLS

function SetSimControls(sim::MJSimEnv, ui)
    SetSimControls(sim.controls, ui)
    #SetControls(sim)
end

function GetSimControls(sim::MJSimEnv)
    u = zeros(sim.controls.nu, 1)
    for i=1:sim.controls.nu
        u[i] = sim.controls.ctrl[i, 1]
    end
    return u
end



###ACTUAL DYNAMICS FUNCTIONS

#transition
function mjDynamicsF(xi, ui, i, sim, intermediateCost)
    SetSimState(sim, xi)
    SetSimControls(sim, ui)
    mj_forward(sim.model, sim.data)
    xnew = GetSimState(sim)
    cost = intermediateCost(xnew, ui, sim)
    return (xnew, cost)
end

#compute jacobian and hessian along each point in the given trajectory
#time invariant, can drop I??
function mjDynamicsFx(x, u, I, sim, costx, costu, costxx, costuu, linear=false)
    #compute derivatives:
    nx = sim.state.nx
    #fx, fu, fxx, fuu, fxu, cx, cu, cxx, cxu, cuu
    fx = zeros(nx, nx, length(I) + 1)
    fu = zeros(nx, sim.controls.nu, length(I) + 1)
    cx = zeros(nx, length(I) + 1)
    cu = zeros(sim.controls.nu, length(I))
    cxx = zeros(nx, nx)
    cxu = zeros(nx, sim.controls.nu)
    cuu = zeros(sim.controls.nu, sim.controls.nu)

    fxx = fxu = fuu = []

    if !linear
        fxx = zeros(nx, nx, nx, length(I) + 1)
        fuu = zeros(nx, sim.controls.nu, sim.controls.nu, length(I) + 1)
        fxu = zeros(nx, nx, sim.controls.nu, length(I) + 1)
    end

    #for every point on trajectory
    for t in I
        #set states
        SetSimState(sim, x[:, t])
        SetSimControls(sim, u[:, t])
        mj_forward(sim.model, sim.data)

        dSim = MJSimEnv{SimState, AltControls}(sim.model, s -> (0), s -> (false), SimState(), AltControls())
        SetSimState(dSim, x[:, t])
        SetSimControls(dSim, u[:, t])
        mj_forward(dSim.model, dSim.data)
        for i=1:3
            mj_forward(dSim.model, dSim.data)
        end

        ddSim = MJSimEnv{SimState, AltControls}(sim.model, s -> (0), s -> (false), SimState(), AltControls())
        SetSimState(ddSim, x[:, t])
        SetSimControls(ddSim, u[:, t])
        mj_forward(ddSim.model, ddSim.data)
        for i=1:3
            mj_forward(ddSim.model, ddSim.data)
        end


        ###DYNAMICS MODEL DERIVATIVES these top level loops can be parallelized

        #fu & fuu
        for nu1=1:dSim.controls.nu

            dSim.controls.ctrl[nu1] += 1.0e-4

            #warmstart acc
            icxx"""
            mju_copy($(dSim.data.d)->qacc_warmstart, $(sim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            """

            #step to get difference
            for step=1:1
                mj_forward(dSim.model, dSim.data)
            end

            #calc fu
            for nx3=1:sim.state.nx
                fu[nx3, nu1, t] = (dSim.state[nx3] - sim.state[nx3]) / 2.0 / 1.0e-4
            end

            if !linear
                for nu2=1:ddSim.controls.nu

                    ddSim.controls.ctrl[nu2] -= 1.0e-4

                    #warmstart acc
                    icxx"""
                    mju_copy($(ddSim.data.d)->qacc_warmstart, $(sim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
                    """

                    #step to get difference
                    for step=1:1
                        mj_forward(ddSim.model, ddSim.data)
                    end

                    #calc fuu
                    for nx3=1:sim.state.nx
                        fuu[nx3, nu1, nu2, t] = (dSim.state[nx3] + ddSim.state[nx3] - 2 * sim.state[nx3]) / (2.0 * 1.0e-4)^2
                    end

                    #reset
                    icxx"""
                    mju_copy($(ddSim.data.d)->qpos, $(sim.data.d)->qpos, $(sim.model.m)->nq);
                    mju_copy($(ddSim.data.d)->qvel, $(sim.data.d)->qvel, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->qacc, $(sim.data.d)->qacc, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->qacc_warmstart, $(sim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->qfrc_applied, $(sim.data.d)->qfrc_applied, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->xfrc_applied, $(sim.data.d)->xfrc_applied, 6*$(sim.model.m)->nsite);
                    mju_copy($(ddSim.data.d)->ctrl, $(sim.data.d)->ctrl, $(sim.model.m)->nu);
                    mj_forward($(ddSim.model.m), $(sim.data.d));
                    """
                end
            end


            #reset
            icxx"""
            mju_copy($(dSim.data.d)->qpos, $(sim.data.d)->qpos, $(sim.model.m)->nq);
            mju_copy($(dSim.data.d)->qvel, $(sim.data.d)->qvel, $(sim.model.m)->nv);
            mju_copy($(dSim.data.d)->qacc, $(sim.data.d)->qacc, $(sim.model.m)->nv);
            mju_copy($(dSim.data.d)->qacc_warmstart, $(sim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mju_copy($(dSim.data.d)->qfrc_applied, $(sim.data.d)->qfrc_applied, $(sim.model.m)->nv);
            mju_copy($(dSim.data.d)->xfrc_applied, $(sim.data.d)->xfrc_applied, 6*$(sim.model.m)->nsite);
            mju_copy($(dSim.data.d)->ctrl, $(sim.data.d)->ctrl, $(sim.model.m)->nu);
            mj_forward($(dSim.model.m), $(dSim.data.d));
            """
        end

        #fx,fxx, and fxu
        for nx1=1:dSim.state.nx

            dSim.state[nx1] += 1.0e-4

            #warmstart acc
            icxx"""
            mju_copy($(dSim.data.d)->qacc_warmstart, $(sim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            """

            #step to get difference
            for step=1:1
                mj_forward(dSim.model, dSim.data)
            end

            #calc fx
            for nx3=1:sim.state.nx
                fx[nx3, nx1, t] = (dSim.state[nx3] - sim.state[nx3]) / 2.0 / 1.0e-4
            end

            if !linear
                #fxx
                for nx2=1:ddSim.state.nx

                    ddSim.state[nx2] -= 1.0e-4

                    #warmstart acc
                    icxx"""
                    mju_copy($(ddSim.data.d)->qacc_warmstart, $(sim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
                    """

                    #step to get difference
                    for step=1:1
                        mj_forward(ddSim.model, ddSim.data)
                    end

                    #calc fxx
                    for nx3=1:sim.state.nx
                        fxx[nx3, nx1, nx2, t] = (dSim.state[nx3] + ddSim.state[nx3] - 2 * sim.state[nx3]) / (2.0 * 1.0e-4)^2
                    end

                    #reset
                    icxx"""
                    mju_copy($(ddSim.data.d)->qpos, $(sim.data.d)->qpos, $(sim.model.m)->nq);
                    mju_copy($(ddSim.data.d)->qvel, $(sim.data.d)->qvel, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->qacc, $(sim.data.d)->qacc, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->qacc_warmstart, $(sim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->qfrc_applied, $(sim.data.d)->qfrc_applied, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->xfrc_applied, $(sim.data.d)->xfrc_applied, 6*$(sim.model.m)->nsite);
                    mju_copy($(ddSim.data.d)->ctrl, $(sim.data.d)->ctrl, $(sim.model.m)->nu);
                    mj_forward($(ddSim.model.m), $(sim.data.d));
                    """
                end

                #fxu
                for nu2=1:ddSim.controls.nu

                    ddSim.controls.ctrl[nu2] -= 1.0e-4

                    #warmstart acc
                    icxx"""
                    mju_copy($(ddSim.data.d)->qacc_warmstart, $(sim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
                    """

                    #step to get difference
                    for step=1:1
                        mj_forward(ddSim.model, ddSim.data)
                    end

                    #calc fxu
                    for nx3=1:sim.state.nx
                        fxu[nx3, nx1, nu2, t] = (dSim.state[nx3] + ddSim.state[nx3] - 2 * sim.state[nx3]) / (2.0 * 1.0e-4)^2
                    end

                    #reset
                    icxx"""
                    mju_copy($(ddSim.data.d)->qpos, $(sim.data.d)->qpos, $(sim.model.m)->nq);
                    mju_copy($(ddSim.data.d)->qvel, $(sim.data.d)->qvel, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->qacc, $(sim.data.d)->qacc, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->qacc_warmstart, $(sim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->qfrc_applied, $(sim.data.d)->qfrc_applied, $(sim.model.m)->nv);
                    mju_copy($(ddSim.data.d)->xfrc_applied, $(sim.data.d)->xfrc_applied, 6*$(sim.model.m)->nsite);
                    mju_copy($(ddSim.data.d)->ctrl, $(sim.data.d)->ctrl, $(sim.model.m)->nu);
                    mj_forward($(ddSim.model.m), $(sim.data.d));
                    """
                end
            end

            #reset
            icxx"""
            mju_copy($(dSim.data.d)->qpos, $(sim.data.d)->qpos, $(sim.model.m)->nq);
            mju_copy($(dSim.data.d)->qvel, $(sim.data.d)->qvel, $(sim.model.m)->nv);
            mju_copy($(dSim.data.d)->qacc, $(sim.data.d)->qacc, $(sim.model.m)->nv);
            mju_copy($(dSim.data.d)->qacc_warmstart, $(sim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mju_copy($(dSim.data.d)->qfrc_applied, $(sim.data.d)->qfrc_applied, $(sim.model.m)->nv);
            mju_copy($(dSim.data.d)->xfrc_applied, $(sim.data.d)->xfrc_applied, 6*$(sim.model.m)->nsite);
            mju_copy($(dSim.data.d)->ctrl, $(sim.data.d)->ctrl, $(sim.model.m)->nu);
            mj_forward($(dSim.model.m), $(dSim.data.d));
            """
        end

        ###COST FUNCTION DERIVATIVES
        cu[:, t] = costu(u[:, t], sim)
        cx[:, t] = costx(x[:, t], sim)
    end

    cuu = costuu(u, sim)
    cxx = costxx(x, sim)
    cxu = cx * transpose(hcat(cu, ones(sim.controls.nu)))

#    println(sum(abs, fx))
#    println(sum(abs, fu))
#    println(sum(abs, cx))
#    println(sum(abs, cu))
#    if !linear
#        println(sum(abs, fxx))
#        println(sum(abs, fxu))
#        println(sum(abs, fuu))
#    end
#    println(sum(abs, cxx))
#    println(sum(abs, cxu))
#    println(sum(abs, cuu))

    return (fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu)
end

#get cost of being in a state
function mjDynamicsFT(xi, sim, finalCost)
    return finalCost(xi, sim)
end
