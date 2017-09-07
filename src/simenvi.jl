struct SimControls
    nu::Int64
    ctrl::MJMatrix
    normalized::Vector{mjtNum}
    ctrl_magnitude::MJMatrix

    function SimControls(data::MJData)
        mat = data.ctrl
        norm = zeros(data.nu)
        rng = data.model.actuator_ctrlrange

        new(data.nu, mat, norm, rng)
    end
end

function SetSimControls(ctrl::SimControls, norm)
    for i=1:ctrl.nu
        ctrl.normalized[i] = norm[i]
        mag = ctrl.ctrl_magnitude[i]
        adj = norm[i] * mag

        if (adj > mag)
            adj = mag
        elseif (adj < -mag)
            adj = -mag
        end

        ctrl.ctrl[i] = adj
    end

    return ctrl
end

function getitem(ctrl::SimControls, i)
    return ctrl.ctrl[i, 1]
end

function setitem!(ctrl::SimControls, val, i)
    ctrl.ctrl.mat[1, i] = val
end

mutable struct MJSimEnv{StateType, ControlType}
    model::MJModel
    data::MJData

    state::StateType
    controls::ControlType

    reward::Function
    testTerminal::Function

    t::Int
    terminal::Bool

    stateType
    controlType

    function MJSimEnv{StateType, ControlType}(m::MJModel, r::Function, tF::Function, s::StateType, c::ControlType) where {StateType, ControlType}
        d = MJData(m)
        mj_resetData(m, d)

        new(m, d, StateType(d), ControlType(d), r, tF, 0, false, StateType, ControlType)
    end
end

function GetState(env::MJSimEnv)
    env.state = env.stateType(env.data)
end

function GetReward(env::MJSimEnv, state=env.state)
    return env.reward(state)
end

function IsTerminalState(env::MJSimEnv, state=env.state)
    terminal = env.testTerminal(state)
    return terminal
end

#function SetControls(env::MJSimEnv, controls=env.controls)
#    for i=1:env.data.nu
#        env.data.ctrl[i] = controls.ctrl[i]
#    end
#end

#function Act(env::MJSimEnv, num_steps::Int = 3, controls=env.controls)
#    for i=1:num_steps
#        #had to do weird set controls each frame in v1.31, TODO?
#        mj_step1(env.model, env.data)
#        for i=1:env.data.nu
#            env.data.ctrl.mat[1, i] = controls.ctrl[i]
#        end
#        mj_step2(env.model, env.data)
#    end
#end

function ResetEnv(env::MJSimEnv)
    env.terminal = false
    env.t = 0
    mj_resetData(env.model, env.data)
    env.state = env.stateType(env.data)
    env.controls = env.controlType(env.data)
end
