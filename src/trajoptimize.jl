import MuJoCo
using MuJoCo

include("simenvi.jl")

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
