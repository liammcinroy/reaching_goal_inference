###This file intends to demonstrate the rendering and basically follows basic.jl
import Cxx
import GLFW
import MuJoCo

using Cxx
using MuJoCo
using MuJoCo.time

include("../ilqg.jl")

function distance(pnt1, pnt2, norm=2)
    sum = 0
    for i=1:length(pnt1)
        sum += (pnt1[i] - pnt2[i]) ^ norm
    end
    return sum ^ (1 / norm)
end

mj_activate()

reaching = false
N = 100

xmlfile = nothing
if reaching
    xmlfile = joinpath(@__DIR__, "../../reaching.xml")
else
    xmlfile = joinpath(@__DIR__, "../../cartpoleBalance.xml")
end

model = MJModel(xmlfile)
target = "2-1"

function intermediateCost(x, u, sim)
    sum = 0
    for i=1:length(u)
        sum += u[i] ^ 2
    end

    for i=1:sim.data.nv
        sum += x[i + sim.data.nq] ^ 2
    end

    if reaching
        handID = mj_name2id(sim.model, icxx"mjOBJ_BODY;", "right_hand")
        targetID = mj_name2id(sim.model, icxx"mjOBJ_BODY;", "target$target")
        sum += distance(sim.data.xpos[handID, :], sim.data.xpos[targetID, :]) ^ 2
    else
        angle = mj_name2id(sim.model, icxx"mjOBJ_JOINT;", "hinge")
        sum += (sim.data.qpos[angle, 1] - pi) ^ 2
    end
    return sum
end

function costx(x, sim)
    cx = zeros(size(x))
    for i=1:sim.data.nv
        cx[i + sim.data.nq] = 2 * x[i + sim.data.nq]
    end

    if reaching
        handID = mj_name2id(sim.model, icxx"mjOBJ_BODY;", "right_hand")
        targetID = mj_name2id(sim.model, icxx"mjOBJ_BODY;", "target$target")
        cx[handID] = 2 * distance(sim.data.xpos[handID, :], sim.data.xpos[targetID, :])
    else
        angle = mj_name2id(sim.model, icxx"mjOBJ_JOINT;", "hinge")
        cx[angle] = 2 * (sim.data.qpos[angle, 1] - pi)
    end
    return cx
end

function costu(u)
    return 2 * u
end

function costxx(x, sim)
    cxx = zeros(size(x, 1), size(x, 1))
    for i=(sim.data.nq + 1):(sim.data.nq + sim.data.nv)
        cxx[i, i] = 2
    end

    if reaching
        handID = mj_name2id(sim.model, icxx"mjOBJ_BODY;", "right_hand")
        cxx[handID, handID] = 2
    else
        angle = mj_name2id(sim.model, icxx"mjOBJ_JOINT;", "hinge")
        cxx[angle, angle] = 2
    end

    return cxx
end

function costuu(u)
    return 2 * eye(sim.data.nu)
end

function costxu(x, u, sim)
    return costx(x, sim) * transpose(hcat(costu(u), ones(sim.data.nu)))
end

function finalCost(x, sim)
    if reaching
        handID = mj_name2id(sim.model, icxx"mjOBJ_BODY;", "right_hand")
        targetID = mj_name2id(sim.model, icxx"mjOBJ_BODY;", "target$target")
        return (1000 * distance(sim.data.xpos[handID, :], sim.data.xpos[targetID, :]))
    else
        angle = mj_name2id(sim.model, icxx"mjOBJ_JOINT;", "hinge")
        return 100 * abs(sim.data.qpos[angle, 1] - pi)
    end
end


sim = MJSimEnv(model, s -> (0), s -> (false))
ResetEnv(sim)
for i in 1:50
    mj_step(sim.model, sim.data)
end
x0 = GetSimState(sim)
u0 = randn(sim.data.nu, N)

interC(x, u) = intermediateCost(x, u, sim)
finalC(x)  = finalCost(x, sim)
cx(x) = costx(x, sim)
cxx(x) = costxx(x, sim)
cxu(x, u) = costxu(x, u, sim)
f(x, u, i) = mjDynamicsF(x, u, i, sim, interC)
fT(x) = mjDynamicsFT(x, sim, finalC)
fX(x, u, I) = mjDynamicsFx(x, u, I, sim, cx, costu, cxx, costuu, cxu)

x = u = nothing

if length(ARGS) == 0
    print("Performing the optimization...")

    x, u, _, __, ___, ____, _____ =
    DifferentialDynamicProgramming.iLQG(f, fT, fX, x0, u0, tolFun=0)

    #save to file
    writecsv("x.csv", x)
    writecsv("u.csv", u)

    println("Done!")

elseif ARGS[1] == "--load"
    print("Loading the previous trajectory...")

    #load from file
    x = readcsv(open("x.csv", "r"))
    u = readcsv(open("u.csv", "r"))
end

window = GLFW.CreateWindow(640, 480, "demo")
GLFW.MakeContextCurrent(window)
GLFW.SwapInterval(1)

renderer = MJRenderer(sim.data)

width = Int32(0)
height = Int32(0)
step = 1
notStepped = true
elapsedTime = 0
ResetEnv(sim)
for i in 1:50
    mj_step(sim.model, sim.data)
end
tic()
while !GLFW.WindowShouldClose(window)

    #simulate
    while (elapsedTime < 1.0 / 5)
        elapsedTime += toq()
        tic()
        if (step < N && notStepped)
            mj_step1(sim.model, sim.data)
            SetSimControls(sim, u[:, step])
            mj_step2(sim.model, sim.data)
            step += 1
            notStepped = false
            println("Frame:", step)
        elseif (notStepped)
            #mj_step(sim.model, sim.data)
            step += 1
            notStepped = false
        end


        #Render
        width, height = GLFW.GetFramebufferSize(window);
        mjv_updateScene(renderer)
        mjr_render(renderer, width, height)

        # Swap front and back buffers
    	GLFW.SwapBuffers(window)

    	# Poll for and process events
    	GLFW.PollEvents()
    end
    notStepped = true
    elapsedTime = 0
end

GLFW.Terminate()
