import Gen
@everywhere using Gen;

import Images
import Colors
using Images

#The locations of the objects
ObjectsPositions = [[54., 5, -35], [54., 5, -10], [54., 5, 15], [54., 5, 40],
                    [18., 5, -35], [18., 5, -10], [18., 5, 15], [18., 5, 40],
                    [-18., 5, -35], [-18., 5, -10], [-18., 5, 15], [-18., 5, 40],
                    [-54., 5, -35], [-54., 5, -10], [-54., 5, 15], [-54., 5, 40]]

function plan_straightline_path(startPos, destPos)
    #assume make it 1/4 of the way (not constant speed!!)
    #colinearPnt = Vector{Float64}(length(startPos))
    #for i=1:length(colinearPnt)
    #    colinearPnt[i] = (3 * destPos[i] + startPos[i]) / 4
    #end
    #return colinearPnt

    dir = destPos - startPos
    speed = 5
    pnt = dir / norm(dir) * speed + startPos

    if speed > norm(dir)
        pnt = destPos
    end

    return pnt
end

#Define the probabilistic program
@program move_hand_to_destination(path_length = 50) begin

    mv_covar = [1. 0 0 ; 0 1. 0; 0 0 1.]

    startPos = @tag(mvnormal([0, 50., -55.], mv_covar), "startPos")

    destinationObject = @tag(uniform_discrete(1, 16), "destinationObject")
    destinationPos = @tag(mvnormal(ObjectsPositions[destinationObject], mv_covar), "destinationPos")

    #move along path (plus some noise)
    lastPos = startPos
    for i=1:path_length
        nextPos = @tag(mvnormal(plan_straightline_path(lastPos, destinationPos), mv_covar), "pathPos$i")
    end
end

#Sample likely goals with SIR and simulations from the simple planner
function sample_linear_planner(num_samples::Int, path_length = 50, prior_trace = nothing)

    #dictionary of the traces and their scores
    traces = Vector{Trace}(num_samples)
    scores = Vector{Float64}(num_samples)

    #collect a sample
    for i=1:num_samples

        #create the trace
        t = Trace()

        #add priors if know
        if (prior_trace != nothing)
            for k in keys(prior_trace.constraints)
                constrain!(t, k, prior_trace.constraints[k])
            end
            for k in keys(prior_trace.interventions)
                intervene!(t, k, prior_trace.interventions[k])
            end
            #for i in keys(prior_trace.proposals)
            #    propose!(t, i, prior_trace.proposals[i])
            #end
        end

        #generate the sample
        scores[i] = @generate(t, move_hand_to_destination(path_length))

        #score and store
        traces[i] = t
    end
    weights = exp.(scores - logsumexp(scores))
    weights = weights / sum(weights)
    return traces[rand(Distributions.Categorical(weights))]
end

function score_of_sampled_goal(goalID, num_samples, path_length = 50, prior_trace = nothing)
    #dictionary of the traces and their scores
    traces = Vector{Trace}(num_samples)
    scores = Vector{Float64}(num_samples)

    #collect a sample
    for i=1:num_samples

        #create the trace
        t = Trace()

        #add priors if know
        if (prior_trace != nothing)
            for k in keys(prior_trace.constraints)
                constrain!(t, k, prior_trace.constraints[k])
            end
            for k in keys(prior_trace.interventions)
                intervene!(t, k, prior_trace.interventions[k])
            end
            #for i in keys(prior_trace.proposals)
            #    propose!(t, i, prior_trace.proposals[i])
            #end
        end
        constrain!(t, "destinationObject", goalID)
        constrain!(t, "destinationPos", ObjectsPositions[goalID])

        #generate the sample
        scores[i] = @generate(t, move_hand_to_destination(path_length))

        #score and store
        traces[i] = t
    end

    return mean(scores)
end

function manhattanDistance(objID1, objID2)
    return abs((objID1 % 4) - (objID2 % 4)) + abs(Int(floor(objID1 / 4) - floor(objID2 / 4)))
end

function getMatrixLocation(objID)
    return (Int(ceil(objID / 4)), (objID % 4) + 1)
end

sampledStepExperiments = [.1, .2, .5, 1]
SIRsampleSizes = [32]
#For each object, load sample linear file, take varying amount of steps from it and
#attempt to infer the goal node via the framework

distributionsOverSamples = Dict()
for steps in sampledStepExperiments
    distributionsOverSIR = Dict()
    for size in SIRsampleSizes
        distributionsOverSIR[size] = zeros(4, 4)
    end
    distributionsOverSamples[steps] = distributionsOverSIR
end


entropyOverSamples = Dict()
for steps in sampledStepExperiments
    entropyOverSIR = Dict()
    for size in SIRsampleSizes
        entropyOverSIR[size] = zeros(4, 4)
    end
    entropyOverSamples[steps] = entropyOverSIR
end

distribution = Dict()
for object in 1:16
    distribution[object] = 0
end

#Task configuration
print("Task configuration:\n\t")
for obj in 1:16
    print(obj, "\t")
    if (obj % 4 == 0)
        print("\n\t")
    end
end

linearData = false
customObj = false

for object in 1:16

    for i in ARGS
        try
            object = parse(Int, i)
            customObj = true
            println("Testing on user specified object:")
        catch
        end
        if i == "-l"
            linearData = true
            println("Testing on linear data set")
        end
    end
    println("Testing on object: $object")

    file = nothing
    if linearData
        file = open(joinpath(@__DIR__, "csv/linear/$(object + 16).csv"), "r")
    else
        file = open(joinpath(@__DIR__, "csv/$(object).csv"), "r")
    end
    humanPoints = []
    line = readline(file)
    while line != ""
        pnt = float(split(line, ",")[1:3])
        if linearData
            shift = [20, -20, 20]
            pnt += shift
        end
        push!(humanPoints, pnt)
        line = readline(file)
    end

    #number of points to skip each frame
    #skip = Int(floor(length(humanPoints) / 50))
    skip = 1

    for sampledSteps in sampledStepExperiments

        println("\tTesting with $sampledSteps % observed points")

        #set the observed values
        prior_trace = Trace()
        intervene!(prior_trace, "startPos", humanPoints[1])
        for i in 1:Int(floor(sampledSteps * length(humanPoints)))
            constrain!(prior_trace, "pathPos$i", humanPoints[skip * i])
        end

        #now collect the distribution of goals for a given sample size
        for SIRsampleSize in SIRsampleSizes
            println("\t\tDistribution for SIR with $SIRsampleSize samples (goal = $object, $sampledSteps % observed steps):")

            #create the distribution from 50 SIR runs
            for sirIt in 1:100
                distribution[value(sample_linear_planner(SIRsampleSize, length(humanPoints),
                prior_trace), "destinationObject")] += 1
            end

            #get cumulative
            print("\n\t")
            for obj in 1:16
                print(distribution[obj] / 100, "\t")
                if (obj % 4 == 0)
                    print("\n\t")
                end
            end
            print("\n")

            expectedDist = 0.0
            for obj in 1:16
                expectedDist += distribution[obj] / 100 * manhattanDistance(object, obj)
            end
            distributionsOverSamples[sampledSteps][SIRsampleSize][getMatrixLocation(object)...] = expectedDist

            entropy = 0.0
            for obj in 1:16
                if distribution[obj] > 0
                    entropy -= distribution[obj] / 100 * log(distribution[obj] / 100)
                end
            end
            entropyOverSamples[sampledSteps][SIRsampleSize][getMatrixLocation(object)...] = entropy


            #get cumulative
            #for obj in 1:16
            #    print(obj, "\t")
            #end
            #print("\n")
            #for obj in 1:16
            #    print(score_of_sampled_goal(obj, SIRsampleSize, length(humanPoints), prior_trace), "\t")
            #end
            #print("\n")

            #reset distribution for next run
            for obj in 1:16
                distribution[obj] = 0
            end
        end
    end

    if customObj
        break
    end
end

#print expected distances
for sampledSteps in sampledStepExperiments
    for SIRsampleSize in SIRsampleSizes
        println("\tExpected Distance for SIR with $SIRsampleSize samples ($sampledSteps % observed steps):")

        print("Exp. Dist.:\n\t")
        for obj in 1:16
            print(distributionsOverSamples[sampledSteps][SIRsampleSize][getMatrixLocation(obj)...], "\t")
            if (obj % 4 == 0)
                print("\n\t")
            end
        end
        save("images/edist$sampledSteps-$SIRsampleSize.png", Gray.(distributionsOverSamples[sampledSteps][SIRsampleSize]))

        max = 0
        print("Entropy:\n\t")
        for obj in 1:16
            print(entropyOverSamples[sampledSteps][SIRsampleSize][getMatrixLocation(obj)...], "\t")
            if (obj % 4 == 0)
                print("\n\t")
            end
            if (entropyOverSamples[sampledSteps][SIRsampleSize][getMatrixLocation(obj)...] > max)
                max = entropyOverSamples[sampledSteps][SIRsampleSize][getMatrixLocation(obj)...]
            end
        end

        save("images/ent$sampledSteps-$SIRsampleSize.png", Gray.(entropyOverSamples[sampledSteps][SIRsampleSize] / max))
    end
end
