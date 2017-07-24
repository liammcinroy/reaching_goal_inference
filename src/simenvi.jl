function copyMJStateIntoVec(data::MJData)
    mat = Vector{mjtNum}(data.nq + data.nv * 2)

    #simple copy from mjdata into raw
    offset = 1
    for i=1:data.nq
        mat[offset] = data.qpos[i, 1]
        offset += 1
    end
    for i=1:data.nv
        mat[offset] = data.qvel[i, 1]
        offset += 1
    end
    for i=1:data.nv
        mat[offset] = data.qacc[i, 1]
        offset += 1
    end

    return mat
end

#Simple state wrapper which contains the raw state plus more? TODO
type SimState
    raw::Vector{mjtNum}

    function SimState(data::MJData)
        new(copyMJStateIntoVec(data))
    end
end

type SimControls
    nu::Int64
    raw::Vector{mjtNum}
    normalized::Vector{mjtNum}
    ctrl_magnitude::Vector{mjtNum}

    function SimControls(data::MJData)
        mat = Vector{mjtNum}(data.nu)
        norm = Vector{mjtNum}(data.nu)
        rng = Vector{mjtNum}(data.nu)

        for i=1:data.nu
            mat[i] = data.ctrl[i, 1]
            norm[i] = 0
            rng[i] = data.model.actuator_ctrlrange[i, 2]
        end

        new(data.nu, mat, norm, rng)
    end
end

function SetSimControls(ctrl::SimControls, norm::Vector{mjtNum})
    for i=1:ctrl.nu
        ctrl.normalized[i] = norm[i]
        mag = ctrl.ctrl_magnitude[i]
        adj = norm[i] * mag

        if (adj > mag)
            adj = mag
        elseif (adj < -mag)
            adj = -mag
        end

        ctrl.raw[i] = adj
    end

    return ctrl
end

type MJSimEnv
    model::MJModel
    data::MJData

    state::SimState
    controls::SimControls

    reward::Function
    testTerminal::Function

    t::Int
    terminal::Bool

    function MJSimEnv(m::MJModel, r::Function, tF::Function)
        d = MJData(m)
        mj_resetData(m, d)

        new(m, d, SimState(d), SimControls(d), r, tF, 0, false)
    end
end

function GetState(env::MJSimEnv)
    env.state = SimState(env.data)
end

function GetReward(env::MJSimEnv, state=env.state)
    return env.reward(state)
end

function IsTerminalState(env::MJSimEnv, state=env.state)
    terminal = env.testTerminal(state)
    return terminal
end

function Act(env::MJSimEnv, num_steps::Int = 3, controls=env.controls)
    for i=1:num_steps
        #had to do weird set controls each frame in v1.31, TODO?
        mj_step1(env.model, env.data)
        for i=1:env.nu
            env.data.ctrl[i, 1] = controls.raw[i]
        end
        mj_step2(env.model, env.data)
    end
end

function ResetEnv(env::MJSimEnv)
    env.terminal = false
    env.t = 0
    mj_resetData(env.model, env.data)
    env.state = SimState(env.data)
    env.controls = SimControls(env.data)
end
