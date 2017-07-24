import Cxx
import DifferentialDynamicProgramming
import MuJoCo
using Cxx
using MuJoCo

include("simenvi.jl")

function SetSimState(sim::MJSimEnv, xi)
    offset = 1
    for i=1:sim.data.nq
        sim.data.qpos.mat[i] = xi[offset]
        offset += 1
    end
    for i=1:sim.data.nv
        sim.data.qvel.mat[i] = xi[offset]
        offset += 1
    end
    for i=1:sim.data.na
        sim.data.act.mat[i] = xi[offset]
        offset += 1
    end
end

function SetSimControls(sim::MJSimEnv, ui)
    for i=1:sim.data.nu
        sim.data.ctrl.mat[i] = ui[i]
    end
end

function GetSimState(sim::MJSimEnv)
    xi = Vector{mjtNum}(sim.data.nq + sim.data.nv + sim.data.na)
    offset = 1
    for i=1:sim.data.nq
        xi[offset] = sim.data.qpos[i, 1]
        offset += 1
    end
    for i=1:sim.data.nv
        xi[offset] = sim.data.qvel[i, 1]
        offset += 1
    end
    for i=1:sim.data.na
        xi[offset] = sim.data.act[i, 1]
        offset += 1
    end
    return xi
end

function GetSimControls(sim::MJSimEnv)
    u = zeros(sim.data.nu, 1)
    for i=1:sim.data.nu
        u[i] = sim.data.ctrl[i, 1]
    end
    return u
end

#transition
function mjDynamicsF(xi, ui, i, sim, intermediateCost)
    SetSimState(sim, xi)
    SetSimControls(sim, ui)
    mj_forward(sim.model, sim.data)
    xnew = GetSimState(sim)
    cost = intermediateCost(xnew, ui)
    return (xnew, cost)
end

#compute jacobian and hessian along each point in the given trajectory
#time invariant, can drop I??
function mjDynamicsFx(x, u, I, sim, costx, costu, costxx, costuu, costxu)
    #compute derivatives:
    #fx, fu, fxx, fuu, fxu, cx, cu, cxx, cxu, cuu
    fx = zeros((sim.data.nq + sim.data.nv + sim.data.na), (sim.data.nq + sim.data.nv + sim.data.na), length(I) + 1)
    fu = zeros((sim.data.nq + sim.data.nv + sim.data.na), sim.data.nu, length(I) + 1)
    cx = zeros((sim.data.nq + sim.data.nv + sim.data.na), length(I) + 1)
    cu = zeros(sim.data.nu, length(I))
    cxx = zeros((sim.data.nq + sim.data.nv + sim.data.na), (sim.data.nq + sim.data.nv + sim.data.na))
    cxu = zeros((sim.data.nq + sim.data.nv + sim.data.na), sim.data.nu)
    cuu = zeros(sim.data.nu, sim.data.nu)

    #for every point on trajectory
    for t in I
        #set states
        SetSimState(sim, x[:, t])
        SetSimControls(sim, u[:, t])
        mj_forward(sim.model, sim.data)

        defaultSim = MJSimEnv(sim.model, s -> (0), s -> (false))
        SetSimState(defaultSim, x[:, t])
        SetSimControls(defaultSim, u[:, t])
        mj_forward(defaultSim.model, defaultSim.data)
        for i=1:3
            mj_forwardSkip(defaultSim.model, defaultSim.data, icxx"mjSTAGE_VEL;", 1)
        end

        ###DYNAMICS MODEL DERIVATIVES these top level loops can be parallelized

        #fu
        offset1 = 1
        for nu1=1:sim.data.nu
            icxx"""
            $(sim.data.d)->ctrl[$nu1] += 1.0e-4;

            //observe the updated values in sim.data, will compare to defaultSim.data
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mj_step($(sim.model.m), $(sim.data.d));
            """

            #set column values (observed derivs wrt modified value in the row)
            offset2 = 1
            for nq2=1:sim.data.nq
                val = Float64(icxx"($(sim.data.d)->qpos[$nq2 - 1] - $(defaultSim.data.d)->qpos[$nq2 - 1]) / 2 / 1.0e-4;")
                fu[offset2, offset1, t] = val
                offset2 += 1
            end
            for nv2=1:sim.data.nv
                val = Float64(icxx"($(sim.data.d)->qvel[$nv2 - 1] - $(defaultSim.data.d)->qvel[$nv2 - 1]) / 2 / 1.0e-4;")
                fu[offset2, offset1, t] = val
                offset2 += 1
            end
            for na2=1:sim.data.na
                val = Float64(icxx"($(sim.data.d)->act[$na2 - 1] - $(defaultSim.data.d)->act[$na2 - 1]) / 2 / 1.0e-4;")
                fu[offset2, offset1, t] = val
                offset2 += 1
            end

            icxx"""
            //reset
            mju_copy($(sim.data.d)->qpos, $(defaultSim.data.d)->qpos, $(sim.model.m)->nq);
            mju_copy($(sim.data.d)->qvel, $(defaultSim.data.d)->qvel, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc, $(defaultSim.data.d)->qacc, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qfrc_applied, $(defaultSim.data.d)->qfrc_applied, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->xfrc_applied, $(defaultSim.data.d)->xfrc_applied, 6*$(sim.model.m)->nbody);
            mju_copy($(sim.data.d)->ctrl, $(defaultSim.data.d)->ctrl, $(sim.model.m)->nu);
            mj_forward($(sim.model.m), $(sim.data.d));
            """
            offset1 += 1
        end

        #fx
        offset1 = 1
        for nq1=1:sim.data.nq
            icxx"""
            $(sim.data.d)->qpos[$nq1] += 1.0e-4;

            //observe the updated values in sim.data, will compare to defaultSim.data
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mj_step($(sim.model.m), $(sim.data.d));
            """

            #set column values (observed derivs wrt modified value in the row)
            offset2 = 1
            for nq2=1:sim.data.nq
                fx[offset2, offset1, t] = (sim.data.qpos[nq2, 1] - defaultSim.data.qpos[nq2, 1]) / 2 / 1.0e-4
                offset2 += 1
            end
            for nv2=1:sim.data.nv
                fx[offset2, offset1, t] = (sim.data.qvel[nv2, 1] - defaultSim.data.qvel[nv2, 1]) / 2 / 1.0e-4
                offset2 += 1
            end
            for na2=1:sim.data.na
                fx[offset2, offset1, t] = (sim.data.act[na2, 1] - defaultSim.data.act[na2, 1]) / 2 / 1.0e-4
                offset2 += 1
            end

            icxx"""
            //reset
            mju_copy($(sim.data.d)->qpos, $(defaultSim.data.d)->qpos, $(sim.model.m)->nq);
            mju_copy($(sim.data.d)->qvel, $(defaultSim.data.d)->qvel, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc, $(defaultSim.data.d)->qacc, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qfrc_applied, $(defaultSim.data.d)->qfrc_applied, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->xfrc_applied, $(defaultSim.data.d)->xfrc_applied, 6*$(sim.model.m)->nbody);
            mju_copy($(sim.data.d)->ctrl, $(defaultSim.data.d)->ctrl, $(sim.model.m)->nu);
            mj_forward($(sim.model.m), $(sim.data.d));
            """
            offset1 += 1
        end
        for nv1=1:sim.data.nv
            icxx"""
            $(sim.data.d)->qvel[$nv1] += 1.0e-4;

            //observe the updated values in sim.data, will compare to defaultSim.data
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mj_step($(sim.model.m), $(sim.data.d));
            """

            #set column values (observed derivs wrt modified value in the row)
            offset2 = 1
            for nq2=1:sim.data.nq
                fx[offset2, offset1, t] = (sim.data.qpos[nq2, 1] - defaultSim.data.qpos[nq2, 1]) / 2 / 1.0e-4
                offset2 += 1
            end
            for nv2=1:sim.data.nv
                fx[offset2, offset1, t] = (sim.data.qvel[nv2, 1] - defaultSim.data.qvel[nv2, 1]) / 2 / 1.0e-4
                offset2 += 1
            end
            for na2=1:sim.data.na
                fx[offset2, offset1, t] = (sim.data.act[na2, 1] - defaultSim.data.act[na2, 1]) / 2 / 1.0e-4
                offset2 += 1
            end

            icxx"""
            //reset
            mju_copy($(sim.data.d)->qpos, $(defaultSim.data.d)->qpos, $(sim.model.m)->nq);
            mju_copy($(sim.data.d)->qvel, $(defaultSim.data.d)->qvel, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc, $(defaultSim.data.d)->qacc, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qfrc_applied, $(defaultSim.data.d)->qfrc_applied, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->xfrc_applied, $(defaultSim.data.d)->xfrc_applied, 6*$(sim.model.m)->nbody);
            mju_copy($(sim.data.d)->ctrl, $(defaultSim.data.d)->ctrl, $(sim.model.m)->nu);
            mj_forward($(sim.model.m), $(sim.data.d));
            """
            offset1 += 1
        end
        for na1=1:sim.data.na
            icxx"""
            $(sim.data.d)->act[$na1] += 1.0e-4;

            //observe the updated values in sim.data, will compare to defaultSim.data
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mj_step($(sim.model.m), $(sim.data.d));
            """

            #set column values (observed derivs wrt modified value in the row)
            offset2 = 1
            for nq2=1:sim.data.nq
                fx[offset2, offset1, t] = (sim.data.qpos[nq2, 1] - defaultSim.data.qpos[nq2, 1]) / 2 / 1.0e-4
                offset2 += 1
            end
            for nv2=1:sim.data.nv
                fx[offset2, offset1, t] = (sim.data.qvel[nv2, 1] - defaultSim.data.qvel[nv2, 1]) / 2 / 1.0e-4
                offset2 += 1
            end
            for na2=1:sim.data.na
                fx[offset2, offset1, t] = (sim.data.act[na2, 1] - defaultSim.data.act[na2, 1]) / 2 / 1.0e-4
                offset2 += 1
            end

            icxx"""
            //reset
            mju_copy($(sim.data.d)->qpos, $(defaultSim.data.d)->qpos, $(sim.model.m)->nq);
            mju_copy($(sim.data.d)->qvel, $(defaultSim.data.d)->qvel, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc, $(defaultSim.data.d)->qacc, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qfrc_applied, $(defaultSim.data.d)->qfrc_applied, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->xfrc_applied, $(defaultSim.data.d)->xfrc_applied, 6*$(sim.model.m)->nbody);
            mju_copy($(sim.data.d)->ctrl, $(defaultSim.data.d)->ctrl, $(sim.model.m)->nu);
            mj_forward($(sim.model.m), $(sim.data.d));
            """
            offset1 += 1
        end

        ###COST FUNCTION DERIVATIVES - quadratic simple intermediate costs ergo simple to calculate
        cx[:, t] = zeros((sim.data.nq + sim.data.nv + sim.data.na))

    end

    for t in I
        if t < length(I)
            cu[:, t] = costu(u[:, t])
        end
        #cx[:, t] = costx(x[:, t])
    end

    cuu = costuu(u)
    cxx = costxx(x)
    cxu = costxu(x, u)

    #assume linear?
    fxx = fxu = fuu = []

    return (fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu)
end

#get cost of being in a state
function mjDynamicsFT(xi, sim, finalCost)
    return finalCost(xi)
end
