# Goal inference through Probabilistic Programs
A project demonstrating goal inference using a PPL to build an end-to-end, modular framework.
This framework can be applied to a variety of tasks, but is used specifically on a reaching task
 where 16 objects are placed on a table and a human user begins to reach for one of the objects.
 Input is supplied by a Kinect sensor, translated into MuJoCo, where a iLQG planner creates an 
approximation for the humans behavior. A probabilistic program encapsulates this planning, collecting
 the likelihood of the observed data given the model. This likelihood density can then be sampled to
 infer the goal object using any inference algorithm, such as SIR.


There are several key files in this repo.

# Interacting with Kinect
`capture.jl` records the movement of a user once they put their fist in front of their chest into a 
.oni file. `rawKinectDataProcessing.jl` can then read this file and output a .csv with the joint position 
 of the right hand over time. Note that the Kinect skeleton stores the user's right side as the left side, 
so the proper mirror conversion must be applied.


`inference.jl` can take the .csv and perform inference on this data set using a linear trajectory with constant
 speed. The expected error and entropy of each will be saved to heatmaps as well as outputted to the command line.


# Interacting with MuJoCo
`nimjctest.jl` reads a .oni file recorded from kinect and plays it out on the humanoid model supplied in reaching.xml. 
`nimjlive.jl` can actually translate the skeleton in real time from the Kinect to the simulation model.


`ilqg.jl` calculates the optimal control inputs in MuJoCo using finite differences and the interfaces defined in
 `simenvi.jl` which is a basic wrapper class for MuJoCo. User must supply cost function and call `iLQG(...)`, but
 it should do the rest. `opt.jl` is an example usage. `optDisplay.jl` will read the saved activations out from a file
 render them.


