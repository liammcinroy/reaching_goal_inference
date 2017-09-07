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
    for i=1:sim.data.nsite
        sim.data.site_xpos.mat[:, i] = xi[offset:(offset + 2)]
        offset += 3
    end
end

function SetSimControls(sim::MJSimEnv, ui)
    SetSimControls(sim.controls, ui)
    SetControls(sim)
end

function GetSimState(sim::MJSimEnv)
    xi = Vector{mjtNum}(sim.data.nq + sim.data.nv + sim.data.nsite * 3)
    offset = 1
    for i=1:sim.data.nq
        xi[offset] = sim.data.qpos[i, 1]
        offset += 1
    end
    for i=1:sim.data.nv
        xi[offset] = sim.data.qvel[i, 1]
        offset += 1
    end
    for i=1:sim.data.nsite
        xi[offset:(offset + 2)] = sim.data.site_xpos.mat[:, i]
        offset += 3
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
    cost = intermediateCost(xnew, ui, sim)
    return (xnew, cost)
end

#compute jacobian along each point in the given trajectory
#time invariant, can drop I??
function mjDynamicsFx(x, u, I, sim, costx, costu, costxx, costuu)
    #compute derivatives:
    nx = size(x, 1)
    #fx, fu, fxx, fuu, fxu, cx, cu, cxx, cxu, cuu
    fx = zeros(nx, nx, length(I) + 1)
    fu = zeros(nx, sim.data.nu, length(I) + 1)
    cx = zeros(nx, length(I) + 1)
    cu = zeros(sim.data.nu, length(I))
    cxx = zeros(nx, nx)
    cxu = zeros(nx, sim.data.nu)
    cuu = zeros(sim.data.nu, sim.data.nu)

    #fxx = zeros()

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
            sim.data.ctrl.mat[1, nu1] += 1.0e-4

            icxx"""
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            """

            for k in 1:1
                mj_step(sim.model, sim.data)
            end

            #set column values (observed derivs wrt modified value in the row)
            offset2 = 1
            for nq2=1:sim.data.nq
                fu[offset2, offset1, t] = (sim.data.qpos[nq2, 1] - defaultSim.data.qpos[nq2, 1]) / 2 / 1.0e-4
                offset2 += 1
            end
            for nv2=1:sim.data.nv
                fu[offset2, offset1, t] = (sim.data.qvel[nv2, 1] - defaultSim.data.qvel[nv2, 1]) / 2 / 1.0e-4
                offset2 += 1
            end
            for nsite2=1:sim.data.nsite
                for j=1:3
                    fu[offset2, offset1, t] = (sim.data.site_xpos[nsite2, j] - defaultSim.data.site_xpos[nsite2, j]) / 2 / 1.0e-4
                    offset2 += 1
                end
            end
            icxx"""
            //reset
            mju_copy($(sim.data.d)->qpos, $(defaultSim.data.d)->qpos, $(sim.model.m)->nq);
            mju_copy($(sim.data.d)->qvel, $(defaultSim.data.d)->qvel, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc, $(defaultSim.data.d)->qacc, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qfrc_applied, $(defaultSim.data.d)->qfrc_applied, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->xfrc_applied, $(defaultSim.data.d)->xfrc_applied, 6*$(sim.model.m)->nsite);
            mju_copy($(sim.data.d)->ctrl, $(defaultSim.data.d)->ctrl, $(sim.model.m)->nu);
            mj_forward($(sim.model.m), $(sim.data.d));
            """
            offset1 += 1
        end

        #fx
        offset1 = 1
        for nq1=1:sim.data.nq
            sim.data.qpos.mat[1, nq1] += 1.0e-4

            icxx"""
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            """

            for k in 1:1
                mj_step(sim.model, sim.data)
            end

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
            for nsite2=1:sim.data.nsite
                for j=1:3
                    fx[offset2, offset1, t] = (sim.data.site_xpos[nsite2, j] - defaultSim.data.site_xpos[nsite2, j]) / 2 / 1.0e-4
                    offset2 += 1
                end
            end

            icxx"""
            //reset
            mju_copy($(sim.data.d)->qpos, $(defaultSim.data.d)->qpos, $(sim.model.m)->nq);
            mju_copy($(sim.data.d)->qvel, $(defaultSim.data.d)->qvel, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc, $(defaultSim.data.d)->qacc, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qfrc_applied, $(defaultSim.data.d)->qfrc_applied, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->xfrc_applied, $(defaultSim.data.d)->xfrc_applied, 6*$(sim.model.m)->nsite);
            mju_copy($(sim.data.d)->ctrl, $(defaultSim.data.d)->ctrl, $(sim.model.m)->nu);
            mj_forward($(sim.model.m), $(sim.data.d));
            """
            offset1 += 1
        end
        for nv1=1:sim.data.nv
            sim.data.qvel.mat[1, nv1] += 1.0e-4

            icxx"""
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            """

            for k in 1:1
                mj_step(sim.model, sim.data)
            end

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
            for nsite2=1:sim.data.nsite
                for j=1:3
                    fx[offset2, offset1, t] = (sim.data.site_xpos[nsite2, j] - defaultSim.data.site_xpos[nsite2, j]) / 2 / 1.0e-4
                    offset2 += 1
                end
            end

            icxx"""
            //reset
            mju_copy($(sim.data.d)->qpos, $(defaultSim.data.d)->qpos, $(sim.model.m)->nq);
            mju_copy($(sim.data.d)->qvel, $(defaultSim.data.d)->qvel, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc, $(defaultSim.data.d)->qacc, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qfrc_applied, $(defaultSim.data.d)->qfrc_applied, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->xfrc_applied, $(defaultSim.data.d)->xfrc_applied, 6*$(sim.model.m)->nsite);
            mju_copy($(sim.data.d)->ctrl, $(defaultSim.data.d)->ctrl, $(sim.model.m)->nu);
            mj_forward($(sim.model.m), $(sim.data.d));
            """
            offset1 += 1
        end
        for nsite1=1:sim.data.nsite#don't do, doesn't make sense
            for j1=1:3

                sim.data.site_xpos.mat[j1, nsite1] += 1.0e-4

                icxx"""
                mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
                """

                for k in 1:1
                    mj_step(sim.model, sim.data)
                end

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
                for nsite2=1:sim.data.nsite
                    for j=1:3
                        fx[offset2, offset1, t] = (sim.data.site_xpos[nsite2, j] - defaultSim.data.site_xpos[nsite2, j]) / 2 / 1.0e-4
                        offset2 += 1
                    end
                end
            end

            icxx"""
            //reset
            mju_copy($(sim.data.d)->qpos, $(defaultSim.data.d)->qpos, $(sim.model.m)->nq);
            mju_copy($(sim.data.d)->qvel, $(defaultSim.data.d)->qvel, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc, $(defaultSim.data.d)->qacc, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qacc_warmstart, $(defaultSim.data.d)->qacc_warmstart, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->qfrc_applied, $(defaultSim.data.d)->qfrc_applied, $(sim.model.m)->nv);
            mju_copy($(sim.data.d)->xfrc_applied, $(defaultSim.data.d)->xfrc_applied, 6*$(sim.model.m)->nsite);
            mju_copy($(sim.data.d)->ctrl, $(defaultSim.data.d)->ctrl, $(sim.model.m)->nu);
            mj_forward($(sim.model.m), $(sim.data.d));
            """
            offset1 += 1
        end

        ###COST FUNCTION DERIVATIVES
        cu[:, t] = costu(u[:, t], sim)
        cx[:, t] = costx(x[:, t], sim)
    end

    cuu = costuu(u, sim)
    cxx = costxx(x, sim)
    cxu = cx * transpose(hcat(cu, ones(sim.data.nu)))

    #assume linear?
    fxx = fxu = fuu = []

    return (fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu)
end

#get cost of being in a state
function mjDynamicsFT(xi, sim, finalCost)
    return finalCost(xi, sim)
end
