import MuJoCo
using MuJoCo

include("simenvi.jl")

function invstep(sim::MJSimEnv, x_prev, x_curr, x_next)

    # separate out x`
    q_prev = x_prev[...,:sim.data.nq]
    q_curr = x_curr[...,:sim.data.nq]
    q_next = x_next[...,:sim.data.nq]

    # compute time derivatives
    qpos = q_curr
    qvel = (q_next - q_curr) / sim.dt
    qacc = (q_prev - 2.0 * q_curr + q_next) / (sim.dt * sim.dt)

    assert qpos.shape[-1] == sim.model.nq
    assert qvel.shape[-1] == sim.model.nv
    assert qacc.shape[-1] == sim.model.nv
    assert qpos.shape[:-1] == qvel.shape[:-1] and qvel.shape[:-1] == qacc.shape[:-1], \
        'All inputs must have the same batch size, but:\nQPOS=%s\nQVEL=%s\nU=%s' \
        % (qpos.shape[:-1], qvel.shape[:-1], qacc.shape[:-1])
    batch_shape = list(qpos.shape[:-1])

    state = np.empty(batch_shape, dtype=sim.dtype)

    for flat_i in range(np.prod(batch_shape)):
        i = np.unravel_index(flat_i, batch_shape)
        sim.model.data.qpos = qpos[i]
        sim.model.data.qvel = qvel[i]
        sim.model.data.qacc = qacc[i]

        # perform inverse dynamics step
        #mjlib.mj_inverse(sim.model.ptr, sim.model.data.ptr)
        # recreate mj_inverse:
        mjlib.mj_invPosition(sim.model.ptr, sim.model.data.ptr)
        mjlib.mj_invVelocity(sim.model.ptr, sim.model.data.ptr)
        #mjlib.mj_invConstraint(sim.model.ptr, sim.model.data.ptr)
        qfrc_inverse = sim.model.data.qfrc_inverse
        mjlib.mj_rne(sim.model.ptr, sim.model.data.ptr, 1, qfrc_inverse.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        # # override forces
        # efc_force = sim.model.data.efc_force
        # qfrc_constraint = sim.model.data.qfrc_constraint
        # mjlib.mj_mulJacTVec(sim.model.ptr, sim.model.data.ptr, \
        #     qfrc_constraint.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
        #     efc_force.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        # sim.model.data.qfrc_constraint = qfrc_constraint

        sim.model.data.qfrc_inverse = qfrc_inverse \
            + sim.model.dof_armature * sim.model.data.qacc \
            - sim.model.data.qfrc_passive# - sim.model.data.qfrc_constraint

        # record results
        for result_field, source_field, source_idx in sim.fields:
            val = getattr(sim.model.data, source_field)
            if source_idx is not None: val = val[source_idx]
            state[result_field][i] = val.flatten()

    # convert the numpy structured array type back into a dict so it has .items() etc.
    # this is fine since it just creates views and does not copy.
    state_dict = {k: state[k] for k in state.dtype.names}
    # add custom fields to dictionary
    state_dict["frc_root"] = state_dict["qfrc_inverse"][...,sim.unactuated_dofs()]
    state_dict["frc_actuator"] = state_dict["qfrc_inverse"][...,sim.actuated_dofs()]
    # penetration
    state_dict["efc_penetration"] = -np.minimum(state_dict["efc_pos"],0.0)

    return state_dict
end

function dynamics_linearization(inverse_dynamics, x_means, eps)
    f_x = []
    for i=1:length(x_means)
        xi_mean = x_means[i]
        T = size(xi_mean)[1]
        dimx = size(xi_mean)[2]

        #calculate finite difference samples
        x_samples = Vector(x_samples)
        for j=1:length(x_means)
            x_samples[j] = repeat(x_means[j], outer=[1, 1, dimx * 2])
        end
        for k=1:dimx
            x_samples[i][k, :, k] -= eps
            x_samples[i][k, :, k + dimx] += eps
        end

        #compute inverse_dynamics
        f_x_samples = inverse_dynamics(x_samples)

        #calculate finite differences from samples
        f_xi = {}
        keys = keys(f_x_samples)
        for j=1:length(f_x_samples)
            v_samples = get(f_x_samples, keys[j], nothing)
            dimv = shape(v_samples)[3]
            v_x = zeros([dimx, dimv, T])
            for k=1:dimx
                v_x[k, :, :] = (v_samples[:, :, k + dimx] - v_samples[:. :, k]) / 2 / eps
            end
            push(f_xi, keys[j], v_x)
        end
        push(f_x, f_xi)
    end

    return (inverse_dynamics(x_means), f_x)
end

function create_cost_quadratic_element(cost)
    w = cost["w"]
    r = get(cost, "r", nothing)
    c = w * VectorVectorProduct(r, r) / 2
    c_x = []
    c_xx = []
    r_x = get(cost, "r_x", nothing)
    for i=1:length(r_x)
        c_xi = w * MatrixVectorProduct(r_x[i], r)
        push!(c_x, c_xi)

        c_xix = []
        for j=1:length(r_x)
            c_xixj = w * MatrixMatrixProduct(r_x[i], r_x[j])
            push!(c_xix, c_xixj)
        end
        push!(c_xx, c_xix)
    end
    return (c, c_x, c_xx)
end

function create_cost_quadratic(costlist)
    C, C_x, C_xx = create_cost_quadratic_element(costlist[1])
    for i=2:length(costlist)
        c, c_x, c_xx = create_cost_quadratic_element(costlist[i])
        C += c
        for i=1:length(c_x)
            C_x[i] += c_x[i]

            for j=1:length(c_xx[i])
                C_xx[i][j] += c_xx[i][j]
            end
        end
    end

    return (C, C_x, C_xx)
end

function direct_solve(c, c_x, c_xx, lambda)
    T = length(c)

    #assemble sparse trajectory Hessian and dense gradient
    l_x = repeat(nothing, outer=[T])
    l_xx = repeat(nothing, outer=[T, T])

    for t=1:T
        for i=1:3
            ti = t + i - 1
            if (ti < 0 || ti > (T-1))
                continue
            end
            if l_x[ti] == nothing
                l_x[ti] = 0.0
            end
            l_x[ti] += c_x[i][:, t]
            for j=1:3
                tj = t + j - 1
                if (tj < 0 || tj > (T-1))
                    continue
                end
                if l_xx[tj, ti] == nothing
                    l_xx[tj, ti] = 0.0
                end
                l_xx[tj, ti] += c_xx[i][j][:,:, t]
            end
        end
    end


end

function direct_trajectory_optimization(qpos_init, qvel_init, T, sim::MJSimEnv, inverse_dynamics,
    num_iterations, X=nothing)
    force_init = zeros()
    if X == nothing
        force_init = zeros(size(sim.data.njmax))
        X = repeat(vcat(qpos_init, qvel_init), outer=[1, T]) + randn(size(X)[1], size(X)[2]) * .1
    end

    f = nothing
    f_x = nothing

    #Perform each iteration
    for iteration=1:num_iterations
        #collect inverse_dynamics inputs
        x_prev = vcat(X[:, [1]], X[:, 2:T]))
        x_curr = X
        x_next = vcat(X[:, 2:], X[:, [T]])

        #linearize inverse_dynamics
        f, f_x = dynamics_linearization(inverse_dynamics, [x_prev, x_curr, x_next], 1e-6)

        #calculate cost
        cost = []

        #get cost lineariation
        c, c_x, c_xx = create_cost_quadratic(cost)

        #update trajectory parameters
        dX = direct_solve(c, c_x, c_xx, .1)
        X += reshape(dX, size(X))
    end

    return f
end
