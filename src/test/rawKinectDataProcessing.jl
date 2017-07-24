import OpenNI
using OpenNI

function distance(pnt1, pnt2, norm=2)
    sum = 0
    for i=1:length(pnt1)
        sum += (pnt1[i] - pnt2[i]) ^ norm
    end
    return sum ^ (1 / norm)
end

function angleBetweenVectors(pnt1, pnt2, pnt3)
    c = distance(pnt3, pnt1)
    a = distance(pnt1, pnt2)
    b = distance(pnt2, pnt3)

    try
        return acos((a ^ 2 + b ^ 2 - c ^ 2) /
         (a * b * 2))
    catch
        return pi
    end
end

function convertKinectPntToSimPnt(kinectPnt::Vector{NIFloat}, origin::Vector{NIFloat})
    for i=1:length(origin)
        kinectPnt[i] -= origin[i]
        #now scale down, first convert to inches, then to game points
        kinectPnt[i] *= 0.0393701 / .27
    end
    kinectPnt[2] += 70 #add to y? TODO way to automate
    kinectPnt[3] *= -1 #flip z
    return kinectPnt
end

function convertSkeletonToSim(jointPos::Matrix{NIFloat}, origin::Vector{NIFloat})
    for i=1:length(jointPos[:, 1])
        jointPos[i, :] = convertKinectPntToSimPnt(jointPos[i, :], origin)
    end
end

function delimVector(pnt)
    str = ""
    for i=1:length(pnt)
        str = "$str$(pnt[i]),"
    end
    return str
end


###Record only once skeleton is detected and then the user puts their right hand by torso

g_elapseSinceLastEvent = 0
g_userFound = false
g_startPlaying = false
g_stopPlaying = false
g_origin = Vector{NIFloat}()

#g_outputStream = STDOUT
g_outputStream = open(joinpath(@__DIR__, "../../data/csv/$(ARGS[1]).csv"), true, true, true, true, false)

#Init without any preloaded nodes
print("Creating context...")

context = NIContext()
InitContext(context)

println("Done!")
print("Creating Player...")

player = NIPlayer()
OpenContextFileRecording(player, context,  joinpath(@__DIR__, "../../data/$(ARGS[1]).oni"))

println("Done!")
print("Attaching generators...")

#Now create a UserGenerator node
usergen = NIUserGenerator()
CreateUserGenerator(usergen, context)

#Function for callbacks (new and lost user)
function userFound()
    println("User recognized registered!")
end

function userLost()
    g_stopPlaying = true
    println("User lost. Stopping recording")
end

RegisterUserCallbacks(usergen, userFound, userLost)

println("Done!")

#Begin generators
StartGeneratingAll(context)

print("Starting playback...")

#init array of user IDs
userIDs = Vector{UInt32}(2)

tic()
#start program loop
while !g_stopPlaying
    #each frame update, but wait for usergen to have new data
    WaitOneUpdateAll(context, usergen)

    userIDs = GetUsers(usergen, userIDs)

    for i=1:length(userIDs)
        #If tracking, print head coordinates
        if (IsTrackingUser(usergen, userIDs[i]))

            skeleton = NISkeleton(usergen, userIDs[i])
            jntHandID = Int(OpenNI.NI_SKEL_LEFT_HAND) #mirrored
            jntTorsoID = Int(OpenNI.NI_SKEL_TORSO)
            dist = distance(skeleton.jointPos[jntHandID, :], skeleton.jointPos[jntTorsoID, :])

            g_elapseSinceLastEvent += toq()
            tic()

            if (!g_userFound && dist < 250)
                g_userFound = true
                print("User struck pose, beginning motion in 2 seconds...")

                #calculate origin
                g_origin = deepcopy(skeleton.jointPos[jntHandID, :])

                #adjust the y? TODO
                #adjust the z
                g_origin[3] -= 55 / 0.0393701 * .27

                println("Origin:", g_origin)

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
                println(g_outputStream, delimVector(convertKinectPntToSimPnt(deepcopy(skeleton.jointPos[jntHandID, :]), g_origin)))
            end

            if (g_startPlaying && g_elapseSinceLastEvent > 5)
                println("Stopping playback")
                g_stopPlaying = true
            end
        end #IsTrackingUser
    end #for
end

println("Finished playback")
