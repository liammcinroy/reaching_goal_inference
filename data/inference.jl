import Gen
import HypothesisTests
import Images
import Colors

using HypothesisTests
using Images

@everywhere using Gen;


#The locations of the objects
ObjectsPositions = [[54., 5, -35], [54., 5, -10], [54., 5, 15], [54., 5, 40],
                    [18., 5, -35], [18., 5, -10], [18., 5, 15], [18., 5, 40],
                    [-18., 5, -35], [-18., 5, -10], [-18., 5, 15], [-18., 5, 40],
                    [-54., 5, -35], [-54., 5, -10], [-54., 5, 15], [-54., 5, 40]]

function linear_planner_constant(startPos, destPos, num_pnts)
    dir = destPos - startPos
    speed = 5

    pnts = []

    pnt = deepcopy(startPos)
    for p in 1:num_pnts
        pnt = dir / norm(dir) * speed + pnt
        if speed > norm(dir)
            pnt = destPos
        end
        push!(pnts, pnt)
    end
    return pnts
end

function linear_planner_exponential(startPos, destPos, num_pnts)
    #assume make it 1/4 of the way (not constant speed!!)
    pnts = []
    colinearPnt = deepcopy(startPos)
    for p in 1:num_pnts
        for i=1:length(colinearPnt)
            colinearPnt[i] = (3 * destPos[i] + colinearPnt[i]) / 4
        end
        push!(pnts, colinearPnt)
    end
    return pnts
end

function parabolic_planner_constant(startPos, destPos, num_pnts, a = .01, phirot = 0)
    xaxis = destPos - startPos
    xaxis /= norm(xaxis)

    thetay = atan2(xaxis[2], xaxis[1]) + pi/2 #make orthogonal
    phiy = acos(xaxis[3]) + phirot #rotate if desired

    yaxis = [cos(thetay)*sin(phiy), sin(thetay)*sin(phiy), sin(phiy)]
    yaxis /= norm(yaxis)

    zaxis = cross(xaxis, yaxis)
    zaxis /= norm(zaxis)

    np = norm(startPos)
    nd = norm(destPos - startPos)

    #actual length is less
    l_inc = (np * nd + nd ^ 3 / 3 + (1 - a) / 2 * nd^2) / num_pnts
    l = l_inc

    pnts = []
    atEnd = false
    for i in 1:num_pnts
        #rs = roots(BigFloat.([-l, np, (1 - a) / 2, 1/3]))
        k = i * nd / num_pnts
        #set = false

        #for r in rs
        #    if isreal(r)
        #        if Float64(r) > 0
        #            k = Float64(r)
        #            set = true
        #            break
        #        end
        #    end
        #end

        #if (!set || atEnd)
        #    pnt = destPos
        #    atEnd = true
        #else
            pnt = startPos + k * xaxis + a * k * (nd - k) * zaxis
        #end

        push!(pnts, pnt)
        l += l_inc
    end
    return pnts
end

#Define the probabilistic program
@program move_hand_to_destination(planner::Function, path_length = 50) begin

    mv_covar = [1. 0 0 ; 0 1. 0; 0 0 1.]

    startPos = @tag(mvnormal([0, 50., -55.], mv_covar), "startPos")

    destinationObject = @tag(uniform_discrete(1, 16), "destinationObject")
    destinationPos = @tag(mvnormal(ObjectsPositions[destinationObject], mv_covar), "destinationPos")

    #move along path (plus some noise)
    pnts = planner(startPos, destinationPos, path_length)
    for i=1:path_length
        nextPos = @tag(mvnormal(pnts[i], mv_covar), "pathPos$i")
    end
end

#Sample likely goals with Inference and simulations from the simple planner
function infer_planner_samples(planner::Function, num_samples::Int, sir, path_length = 50, prior_trace = nothing)

    if sir > 0
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
            scores[i] = @generate(t, move_hand_to_destination(planner, path_length))

            #score and store
            traces[i] = t
        end
        weights = exp.(scores - logsumexp(scores))
        weights = weights / sum(weights)
        return traces[rand(Distributions.Categorical(weights))]
    else
        #Nested inference MH

        currentTrace = Trace()

        #add priors if know
        if (prior_trace != nothing)
            for k in keys(prior_trace.constraints)
                constrain!(currentTrace, k, prior_trace.constraints[k])
            end
            for k in keys(prior_trace.interventions)
                intervene!(currentTrace, k, prior_trace.interventions[k])
            end
            #for i in keys(prior_trace.proposals)
            #    propose!(t, i, prior_trace.proposals[i])
            #end
        end


        #generate the sample
        old_score = @generate(currentTrace, move_hand_to_destination(planner, path_length))
        i = 1
        #collect a sample
        while i < num_samples

            #create the new sample trace
            t = deepcopy(currentTrace)

            #generate the sample
            new_score = @generate(t, move_hand_to_destination(planner, path_length))
            #if log(rand()) < new_score - old_score || i == 0
                old_score = new_score
                currentTrace = t
                i += 1
            #end
        end
        return currentTrace
    end
end

function manhattanDistance(objID1, objID2)
    return abs((objID1 - 1) % 4 - (objID2 - 1) % 4) + abs(Int(ceil(objID1 / 4) - ceil(objID2 / 4)))
end

function getMatrixLocation(objID)
    return (Int(ceil(objID / 4)), ((objID - 1) % 4) + 1)
end

planners = [linear_planner_constant, parabolic_planner_constant]
sampledStepExperiments = [.1, .2, .5, 1]
inferenceAlgorithms = [[1, true], [32, true], [128, true], #[512, true],
                        [1, false], [32, false]]#, [256, false], [1024, false]]
#For each object, load sample linear file, take varying amount of steps from it and
#attempt to infer the goal node via the framework

distributionsOverSamples = Dict()
for planner in planners
    distributionsOverSteps = Dict()
    for steps in sampledStepExperiments
        distributionsOverInference = Dict()
        for size in inferenceAlgorithms
            distributionsOverInference[size] = zeros(4, 4)
        end
        distributionsOverSteps[steps] = distributionsOverInference
    end
    distributionsOverSamples[planner] = distributionsOverSteps
end

entropyOverSamples = Dict()
for planner in planners
    entropyOverSteps = Dict()
    for steps in sampledStepExperiments
        entropyOverInference = Dict()
        for size in inferenceAlgorithms
            entropyOverInference[size] = zeros(4, 4)
        end
        entropyOverSteps[steps] = entropyOverInference
    end
    entropyOverSamples[planner] = entropyOverSteps
end

timeOverSamples = Dict()
for planner in planners
    timeOverSteps = Dict()
    for steps in sampledStepExperiments
        timeOverInference = Dict()
        for size in inferenceAlgorithms
            timeOverInference[size] = zeros(100,1)
        end
        timeOverSteps[steps] = timeOverInference
    end
    timeOverSamples[planner] = timeOverSteps
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

for planner in planners
    println("TESTING ON PLANNER: $planner")

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
            for inferenceAlgorithm in inferenceAlgorithms
                println("\t\t(planner = $planner, goal = $object, % observations = $sampledSteps, Inference samples = $inferenceAlgorithm):")

                #create the distribution from 50 Inference runs
                for infIt in 1:100
                    tic()
                    distribution[value(infer_planner_samples(planner, inferenceAlgorithm..., length(humanPoints),
                    prior_trace), "destinationObject")] += 1
                    timeOverSamples[planner][sampledSteps][inferenceAlgorithm][infIt] = toq()
                end

                #print cumulative
                #print("\n\t")
                #for obj in 1:16
                #    print(distribution[obj] / 100, "\t")
                #    if (obj % 4 == 0)
                #        print("\n\t")
                #    end
                #end
                #print("\n")

                expectedDist = 0.0
                for obj in 1:16
                    expectedDist += distribution[obj] / 100 * manhattanDistance(object, obj)
                end
                distributionsOverSamples[planner][sampledSteps][inferenceAlgorithm][getMatrixLocation(object)...] = expectedDist

                entropy = 0.0
                for obj in 1:16
                    if distribution[obj] > 0
                        entropy -= distribution[obj] / 100 * log(distribution[obj] / 100)
                    end
                end
                entropyOverSamples[planner][sampledSteps][inferenceAlgorithm][getMatrixLocation(object)...] = entropy


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
end

#print expected distances
for planner in planners
    for sampledSteps in sampledStepExperiments
        for inferenceAlgorithm in inferenceAlgorithms
            println("\tExpected Distance for inference with $planner, $inferenceAlgorithm samples, $sampledSteps % observed steps):")

            print("Exp. Dist.:\n\t")
            max = 0
            for obj in 1:16
                print(distributionsOverSamples[planner][sampledSteps][inferenceAlgorithm][getMatrixLocation(obj)...], "\t")
                if (obj % 4 == 0)
                    print("\n\t")
                end
                if (distributionsOverSamples[planner][sampledSteps][inferenceAlgorithm][getMatrixLocation(obj)...] > max)
                    max = distributionsOverSamples[planner][sampledSteps][inferenceAlgorithm][getMatrixLocation(obj)...]
                end
            end
            save("images/eerr-$planner-$(Int(100 * sampledSteps))-$inferenceAlgorithm.png", Gray.(ones(4, 4) - distributionsOverSamples[planner][sampledSteps][inferenceAlgorithm] / max))

            max = 0
            print("Entropy:\n\t")
            for obj in 1:16
                print(entropyOverSamples[planner][sampledSteps][inferenceAlgorithm][getMatrixLocation(obj)...], "\t")
                if (obj % 4 == 0)
                    print("\n\t")
                end
                if (entropyOverSamples[planner][sampledSteps][inferenceAlgorithm][getMatrixLocation(obj)...] > max)
                    max = entropyOverSamples[planner][sampledSteps][inferenceAlgorithm][getMatrixLocation(obj)...]
                end
            end

            save("images/ent-$planner-$(Int(100 * sampledSteps))-$inferenceAlgorithm.png", Gray.(ones(4, 4) - entropyOverSamples[planner][sampledSteps][inferenceAlgorithm] / max))

            print("Average Time:$(sum(timeOverSamples[planner][sampledSteps][inferenceAlgorithm][:] / 100)) sec\n\t")
        end
    end
end
println("")

function printSigTable(mat, disp, savefilename)
    print(disp)
    for i in 1:size(mat,1)
        print("\n\t")
        for j in 1:size(mat,2)
            print(mat[i, j],"\t")
        end
    end
    println("\n")
    writecsv(joinpath("/home/liam/Documents/RSI/Probcomp/reaching_goal_inference/data", savefilename), mat)
end

println("Signficance for different planners (Wilcoxon signed rank test):")
planSigsErrs = ones(length(planners), length(planners))
planSigsEnts = ones(length(planners), length(planners))
planSigsTimes = ones(length(planners), length(planners))

for i in 1:length(planners)
    for j in 1:length(planners)
        if i == j
            continue
        end
        planner1 = planners[i]
        planner2 = planners[j]

        errs1 = distributionsOverSamples[planner1][.5][[32,1]]
        errs2 = distributionsOverSamples[planner2][.5][[32,1]]
        planSigsErrs[i, j] = pvalue(SignedRankTest(reshape(errs1, length(errs1)), reshape(errs2, length(errs2))))

        ents1 = entropyOverSamples[planner1][.5][[32,1]]
        ents2 = entropyOverSamples[planner2][.5][[32,1]]
        planSigsEnts[i, j] = pvalue(SignedRankTest(reshape(ents1, length(ents1)), reshape(ents2, length(ents2))))

        times1 = timeOverSamples[planner1][.5][[32,1]]
        times2 = timeOverSamples[planner2][.5][[32,1]]
        planSigsTimes[i, j] = pvalue(SignedRankTest(reshape(times1, length(times1)), reshape(times2, length(times2))))
    end
end

printSigTable(planSigsErrs, "Expected Error Significance (wrt planner):", "sig/plansigerr.csv")
printSigTable(planSigsEnts, "Entropy Significance (wrt planner):", "sig/plansigent.csv")
printSigTable(planSigsTimes, "Time Significance (wrt planner):", "sig/plansigt.csv")

println("Signficance for varying observation sizes (Wilcoxon signed rank test):")
obsSigsErrs = ones(length(sampledStepExperiments), length(sampledStepExperiments))
obsSigsEnts = ones(length(sampledStepExperiments), length(sampledStepExperiments))
obsSigsTimes = ones(length(sampledStepExperiments), length(sampledStepExperiments))

for i in 1:length(sampledStepExperiments)
    for j in 1:length(sampledStepExperiments)
        if i == j
            continue
        end

        sampledSteps1 = sampledStepExperiments[i]
        sampledSteps2 = sampledStepExperiments[j]

        errs1 = distributionsOverSamples[linear_planner_constant][sampledSteps1][[32,1]]
        errs2 = distributionsOverSamples[linear_planner_constant][sampledSteps2][[32,1]]
        obsSigsErrs[i, j] = pvalue(SignedRankTest(reshape(errs1, length(errs1)), reshape(errs2, length(errs2))))
        ents1 = entropyOverSamples[linear_planner_constant][sampledSteps1][[32,1]]
        ents2 = entropyOverSamples[linear_planner_constant][sampledSteps2][[32,1]]
        obsSigsEnts[i, j] = pvalue(SignedRankTest(reshape(ents1, length(ents1)), reshape(ents2, length(ents2))))

        times1 = timeOverSamples[linear_planner_constant][sampledSteps1][[32,1]]
        times2 = timeOverSamples[linear_planner_constant][sampledSteps2][[32,1]]
        obsSigsTimes[i, j] = pvalue(SignedRankTest(reshape(times1, length(times1)), reshape(times2, length(times2))))
    end
end

printSigTable(obsSigsErrs, "Expected Error Significance (wrt observation sample size):", "sig/obssigerr.csv")
printSigTable(obsSigsEnts, "Entropy Significance (wrt observation sample size):", "sig/obssigent.csv")
printSigTable(obsSigsTimes, "Time Significance (wrt observation sample size):", "sig/obssigt.csv")


println("Signficance for varying Inference sample sizes (Wilcoxon signed rank test):")
infSigsErrs = ones(length(inferenceAlgorithms), length(inferenceAlgorithms))
infSigsEnts = ones(length(inferenceAlgorithms), length(inferenceAlgorithms))
infSigsTimes = ones(length(inferenceAlgorithms), length(inferenceAlgorithms))

for i in 1:length(inferenceAlgorithms)
    for j in 1:length(inferenceAlgorithms)
        if i == j
            continue
        end

        inferenceAlgorithm1 = inferenceAlgorithms[i]
        inferenceAlgorithm2 = inferenceAlgorithms[j]

        errs1 = distributionsOverSamples[linear_planner_constant][.5][inferenceAlgorithm1]
        errs2 = distributionsOverSamples[linear_planner_constant][.5][inferenceAlgorithm2]
        infSigsErrs[i, j] = pvalue(SignedRankTest(reshape(errs1, length(errs1)), reshape(errs2, length(errs2))))

        ents1 = entropyOverSamples[linear_planner_constant][.5][inferenceAlgorithm1]
        ents2 = entropyOverSamples[linear_planner_constant][.5][inferenceAlgorithm2]
        infSigsEnts[i, j] = pvalue(SignedRankTest(reshape(ents1, length(ents1)), reshape(ents2, length(ents2))))

        times1 = timeOverSamples[linear_planner_constant][.5][inferenceAlgorithm1]
        times2 = timeOverSamples[linear_planner_constant][.5][inferenceAlgorithm2]
        infSigsTimes[i, j] = pvalue(SignedRankTest(reshape(times1, length(times1)), reshape(times2, length(times2))))
    end
end

printSigTable(infSigsErrs, "Expected Error Significance (wrt Inference sample size):", "sig/infsigerr.csv")
printSigTable(infSigsEnts, "Entropy Significance (wrt Inference sample size):", "sig/infsigent.csv")
printSigTable(infSigsTimes, "Time Significance (wrt Inference sample size):", "sig/infsigt.csv")
