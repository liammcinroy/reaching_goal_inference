import OpenNI
using OpenNI

function distance(pnt1, pnt2, norm=2)
    sum = 0
    for i=1:length(pnt1)
        sum += (pnt1[i] - pnt2[i]) ^ norm
    end
    return sum ^ (1 / norm)
end

###Record only once skeleton is detected and then the user puts their right hand by torso

g_elapseSinceLastEvent = 0
g_addedNodes = false
g_userFound = false
g_startedRecording = false
g_stoppedRecording = false

#Init without any preloaded nodes
print("Creating context...")

context = NIContext()
InitContext(context)

println("Done!")
print("Attaching generators...")

#Create an ImageGenerator node
imagegen = NIImageGenerator()
CreateImageGenerator(imagegen, context)

#Create a DepthGenerator node
depthgen = NIDepthGenerator()
CreateDepthGenerator(depthgen, context)

#Now create a UserGenerator node
usergen = NIUserGenerator()
CreateUserGenerator(usergen, context)

#Function for callbacks (new and lost user)
function userFound()
    println("User recognized registered!")
end

function userLost()
    g_stoppedRecording = true
    println("User lost. Stopping recording")
end

RegisterUserCallbacks(usergen, userFound, userLost)

println("Done!")
print("Creating Recorder...")

#Create recorder
recorder = NIRecorder()
CreateRecorder(recorder, context)

#specify the output of the recorder
SetRecorderDestination(recorder,  joinpath(@__DIR__, "../../output.oni"))

println("Done!")

#Begin generators
StartGeneratingAll(context)

print("Starting device and recording...")

#init array of user IDs
userIDs = Vector{UInt32}(2)

tic()
#start program loop
while !g_stoppedRecording
    #each frame update, but wait for usergen to have new data
    WaitOneUpdateAll(context, usergen)

    userIDs = GetUsers(usergen, userIDs)

    for i=1:length(userIDs)
        #If tracking, print head coordinates
        if (IsTrackingUser(usergen, userIDs[i]))

            #attach generators to recorder, do this now so that the skeleton is detected
            if (!g_addedNodes)
                print("Adding nodes to record...")
                AddNodeToRecording(recorder, imagegen)
                AddNodeToRecording(recorder, depthgen)
                g_addedNodes = true
                println("Done!")
            end

            skeleton = NISkeleton(usergen, userIDs[i])
            jntHandID = Int(OpenNI.NI_SKEL_LEFT_HAND) #mirrored
            jntTorsoID = Int(OpenNI.NI_SKEL_TORSO)
            dist = distance(skeleton.jointPos[jntHandID, :], skeleton.jointPos[jntTorsoID, :])

            g_elapseSinceLastEvent += toq()
            tic()

            if (!g_userFound && dist < 250)
                g_userFound = true
                print("User struck pose, begin motion in 2 seconds...")
                g_elapseSinceLastEvent = 0
                tic()
            end

            if (g_userFound && !g_startedRecording && g_elapseSinceLastEvent > 2)
                println("Starting recording...")
                g_startedRecording = true
                g_elapseSinceLastEvent = 0
                tic()
            end

            if (g_startedRecording && g_elapseSinceLastEvent > 5)
                println("Stopping recording and saving to file")
                g_stoppedRecording = true
            end
        end #IsTrackingUser
    end #for

    #record the frame
    Record(recorder)
end

#cleanup recorder
RemoveNodeFromRecording(recorder, imagegen)
RemoveNodeFromRecording(recorder, depthgen)

#force cleanup to print to file?
finalize(recorder)

println("Finished recording")
