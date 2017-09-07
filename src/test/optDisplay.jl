###This file intends to demonstrate the rendering and basically follows basic.jl
import Cxx
import DifferentialDynamicProgramming
import GLFW
import MuJoCo

using Cxx
using MuJoCo
using MuJoCo.time

include("../mjderivs.jl")

function distance(pnt1, pnt2, norm=2)
    sum = 0
    for i=1:length(pnt1)
        sum += (pnt1[i] - pnt2[i]) ^ norm
    end
    return sum ^ (1 / norm)
end

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


sim = MJSimEnv{SimState, AltControls}(model, s -> (0), s -> (false), SimState(), AltControls())
ResetEnv(sim)
for i in 1:50
    mj_step(sim.model, sim.data)
end
x0 = GetSimState(sim)
u0 = randn(sim.controls.nu, N)

x = u = nothing

if ARGS[1] == "--load"
    print("Loading the previous trajectory...")

    #load from file
    x = readcsv(open("x.csv", "r"))
    u = readcsv(open("u.csv", "r"))

    if length(ARGS) > 1
        x = readcsv(open("x$(ARGS[2]).csv", "r"))
        u = readcsv(open("u$(ARGS[2]).csv", "r"))
    end

    N = size(u, 2)
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
if (!reaching)
    for i in 1:50
        mj_step(sim.model, sim.data)
    end
end
tic()
while !GLFW.WindowShouldClose(window)

    #simulate
    while (elapsedTime < 1.0 / 60)
        elapsedTime += toq()
        tic()
        if (step < N && notStepped)
            #for i in 1:3
                mj_step1(sim.model, sim.data)
                #u[:, step] = zeros(size(u[:, step]))
                #SetSimState(sim, x[:, step])
                SetSimControls(sim, u[:, step])
                mj_step2(sim.model, sim.data)
                mj_forward(sim.model, sim.data)
            #end
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
