###This file intends to demonstrate the rendering and basically follows basic.jl
import MuJoCo
import GLFW

using MuJoCo
using MuJoCo.time



mj_activate()
xmlfile = joinpath(@__DIR__, "cartpoleBalance.xml")
model = MJModel(xmlfile)
data = MJData(model)

window = GLFW.CreateWindow(640, 480, "demo")
GLFW.MakeContextCurrent(window)
GLFW.SwapInterval(1)

renderer = MJRenderer(data)

width = Int32(0)
height = Int32(0)
mj_step(model, data)
while !GLFW.WindowShouldClose(window)

    #simulate
    simstart = time(data)
    while (time(data) - simstart < 1.0/60.0 )
        mj_step(model, data)
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

GLFW.Terminate()
