import GLFW
import MuJoCo
import OpenNI

using MuJoCo
using OpenNI

using Cxx

include("../openNIMuJoCo.jl")
include("../common.jl")

###Record only once skeleton is detected and waits for the user to put their right hand by torso
g_elapseSinceLastEvent = 0
g_firstTime = true
g_sought = false
g_userFound = false
g_startPlaying = false
g_stopPlaying = false
g_origin = Vector{NIFloat}(3)
g_scale = 1

function mapToSimSpace(mapper::NIMJSimMapper, jointPos)
    #changes orientation of gravity, now z->x, y->z, x->y
    #torso is uwaist
    origin_mj = sim.data.xpos[mj_name2id(sim.model, icxx"mjOBJ_BODY;", "lwaist"), :]

    #adjust kinect joint to axis based at init pose
    mjPos = jointPos - g_origin
    #mjPos[3] *= -1 #mirroring

    #shift gravity to mjc space
    temp = last(mjPos)
    for i=length(mjPos):2
        mjPos[i] = mjPos[i - 1]
    end
    mjPos[1] = temp

    #TODO scale down
    mjPos *= g_scale

    #adjust to axis in mjc
    mjPos += origin_mj

    return mjPos
end

println("Setting up OpenNI")

#Init without any preloaded nodes
print("Creating context...")

context = NIContext()
InitContext(context)

println("Done!")
print("Attaching generators...")

#Now create a UserGenerator node
usergen = NIUserGenerator()
CreateUserGenerator(usergen, context)

#Function for callbacks (new and lost user)
function userFound()
    println("User recognized. Waiting for pose.")
end

function userLost()
    g_stopPlaying = true
    println("User lost. Stopping simulation.")
end

RegisterUserCallbacks(usergen, userFound, userLost)

println("Done!")
print("Setting up MuJoCo...")

mj_activate()
sim = MJSimEnv(MJModel(joinpath(@__DIR__,"../../reaching.xml")), s -> (0), s -> (0))

window = GLFW.CreateWindow(640, 480, "demo")
GLFW.MakeContextCurrent(window)
GLFW.SwapInterval(1)

renderer = MJRenderer(sim.data)

width = Int32(0)
height = Int32(0)
mj_step(sim.model, sim.data)

println("Done!")
print("Setting up connection between the two interfaces...")

mapper = NIMJSimMapper(sim, context, usergen, Int32(1), mapToSimSpace)
println(sim.data.xpos[mj_name2id(sim.model, icxx"mjOBJ_BODY;", "target1-1"), :])

println("Done!")
print("Starting streaming...")

#Begin generators
StartGeneratingAll(context)

#init array of user IDs
userIDs = Vector{UInt32}(2)
frames = 0
tic()
#start program loop
while (!GLFW.WindowShouldClose(window))
    #each frame update, but wait for usergen to have new data
    WaitOneUpdateAll(context, usergen)

    if g_firstTime && !g_userFound
        frames += 1
    elseif !g_firstTime && !g_userFound && !g_sought
        #SeekToFrame(player, usergen, Int32(frames))
        g_sought = true
    end


    userIDs = GetUsers(usergen, userIDs)

    for i=1:length(userIDs)
        #If tracking, print head coordinates
        if (IsTrackingUser(usergen, userIDs[i]))

            skeleton = NISkeleton(usergen, userIDs[i])

            #Render MuJoCo
            #mj_step??, set positions from skeleton

            width, height = GLFW.GetFramebufferSize(window);
            mjv_updateScene(renderer)
            mjr_render(renderer, width, height)

        	GLFW.SwapBuffers(window)
        	GLFW.PollEvents()


            #PERFORM OPENNI PROCESSING
            jntRightHandID = Int(OpenNI.NI_SKEL_LEFT_HAND) #mirrored
            jntTorsoID = Int(OpenNI.NI_SKEL_TORSO)
            dist = distance(skeleton.jointPos[jntRightHandID, :], skeleton.jointPos[jntTorsoID, :])

            g_elapseSinceLastEvent += toq()
            tic()

            if (!g_userFound && dist < 250)
                g_userFound = true
                print("User struck pose, beginning motion in 2 seconds...")

                #calculate origin
                g_origin = deepcopy(skeleton.jointPos[jntTorsoID, :])
                println("Origin:", g_origin)

                g_scale = .16 / distance(skeleton.jointPos[jntRightHandID, :], skeleton.jointPos[Int(OpenNI.NI_SKEL_RIGHT_ELBOW), :])

                g_elapseSinceLastEvent = 0
                tic()
            end

            if (g_userFound && !g_startPlaying && g_elapseSinceLastEvent > 2)
                println("Starting motion...")
                g_startPlaying = true
                g_elapseSinceLastEvent = 0
                tic()
            end

            if (g_startPlaying)
                convertSkeletonToMJ(sim, skeleton, g_origin)
                mj_forward(sim.model, sim.data)
            end

            if (g_startPlaying && g_stopPlaying)
                println("Finished playback, restarting")
                g_elapseSinceLastEvent = 0
                g_firstTime = false
                g_sought = false
                g_userFound = false
                g_startPlaying = false
                g_stopPlaying = false
                g_origin = Vector{NIFloat}(3)
                g_scale = 1
                ResetEnv(sim)
            end
        end #IsTrackingUser
    end #for
end
GLFW.Terminate()
