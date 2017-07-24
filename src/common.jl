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

function convertSkeletonToMJ(sim::MJSimEnv, skeleton::NISkeleton, origin)
    jntLHandID = Int(OpenNI.NI_SKEL_RIGHT_HAND)
    jntLElbowID = Int(OpenNI.NI_SKEL_RIGHT_ELBOW)
    jntLShoulderID = Int(OpenNI.NI_SKEL_RIGHT_SHOULDER)
    jntRHandID = Int(OpenNI.NI_SKEL_LEFT_HAND)
    jntRElbowID = Int(OpenNI.NI_SKEL_LEFT_ELBOW)
    jntRShoulderID = Int(OpenNI.NI_SKEL_LEFT_SHOULDER)
    jntNeckID = Int(OpenNI.NI_SKEL_NECK)
    jntTorsoID = Int(OpenNI.NI_SKEL_TORSO)
    jntLHipID = Int(OpenNI.NI_SKEL_RIGHT_HIP)
    jntRHipID = Int(OpenNI.NI_SKEL_LEFT_HIP)
    jntLKneeID = Int(OpenNI.NI_SKEL_RIGHT_KNEE)
    jntRKneeID = Int(OpenNI.NI_SKEL_LEFT_KNEE)

    #get positions and offset to simplify computations
    LHandPos = skeleton.jointPos[jntLHandID, :] - origin
    LElbowPos = skeleton.jointPos[jntLElbowID, :] - origin
    LShoulderPos = skeleton.jointPos[jntLShoulderID, :] - origin
    RHandPos = skeleton.jointPos[jntRHandID, :] - origin
    RElbowPos = skeleton.jointPos[jntRElbowID, :] - origin
    RShoulderPos = skeleton.jointPos[jntRShoulderID, :] - origin
    neckPos = skeleton.jointPos[jntNeckID, :] - origin
    torsoPos = skeleton.jointPos[jntTorsoID, :] - origin
    hipMidPos = (skeleton.jointPos[jntLHipID, :] + skeleton.jointPos[jntRHipID, :]) / 2 - origin
    kneeMidPos = (skeleton.jointPos[jntLKneeID, :] + skeleton.jointPos[jntRKneeID, :]) / 2 - origin

    #compute the reference angles via law of cosines
    refAngleShoulder1 = angleBetweenVectors([0 -0.17 .12], [0 -0.17 .06], [.18 -0.35 -.12])
    refAngleShoulder2 = angleBetweenVectors([-.01 0 .06], [0 -0.17 .06], [.18 -0.35 -.12])
    refAngleElbow = angleBetweenVectors([0 -0.17 .06], [.18 -0.35 -.12], [.36 -.17 .06])
    refTorsoZ = pi / 2
    refTorsoY = pi

    #compute the actual angles via law of cosines
    thetaLShoulder1 = Float64(angleBetweenVectors(deepcopy(LShoulderPos) + [0, 0, 100], LShoulderPos, LElbowPos) - refAngleShoulder2)
    thetaLShoulder2 = -Float64(angleBetweenVectors(neckPos, LShoulderPos, LElbowPos) - refAngleShoulder1)
    thetaLElbow = -Float64(angleBetweenVectors(LShoulderPos, LElbowPos, LHandPos) - refAngleElbow)
    thetaRShoulder1 = -Float64(angleBetweenVectors(deepcopy(RShoulderPos) + [0, 0, 100], RShoulderPos, RElbowPos) - refAngleShoulder2)
    thetaRShoulder2 = -Float64(angleBetweenVectors(neckPos, RShoulderPos, RElbowPos) - refAngleShoulder1)
    thetaRElbow = -Float64(angleBetweenVectors(RShoulderPos, RElbowPos, RHandPos) - refAngleElbow)
    thetaTorsoZ = Float64(angleBetweenVectors(deepcopy(hipMidPos) + [0, 0, 100], hipMidPos, skeleton.jointPos[jntRHipID, :] - origin) - refTorsoZ)
    thetaTorsoY = -Float64(angleBetweenVectors(kneeMidPos, hipMidPos, torsoPos) - refTorsoY)

    #set the angles
    icxx"""
    $(sim.data.d)->qpos[$(sim.model.m)->jnt_dofadr[mj_name2id($(sim.model.m), mjOBJ_JOINT, "left_shoulder1")]] = $thetaLShoulder1;
    $(sim.data.d)->qpos[$(sim.model.m)->jnt_dofadr[mj_name2id($(sim.model.m), mjOBJ_JOINT, "left_shoulder2")]] = $thetaLShoulder2;
    $(sim.data.d)->qpos[$(sim.model.m)->jnt_dofadr[mj_name2id($(sim.model.m), mjOBJ_JOINT, "left_elbow")]] = $thetaLElbow;
    $(sim.data.d)->qpos[$(sim.model.m)->jnt_dofadr[mj_name2id($(sim.model.m), mjOBJ_JOINT, "right_shoulder1")]] = $thetaRShoulder1;
    $(sim.data.d)->qpos[$(sim.model.m)->jnt_dofadr[mj_name2id($(sim.model.m), mjOBJ_JOINT, "right_shoulder2")]] = $thetaRShoulder2;
    $(sim.data.d)->qpos[$(sim.model.m)->jnt_dofadr[mj_name2id($(sim.model.m), mjOBJ_JOINT, "right_elbow")]] = $thetaRElbow;
    $(sim.data.d)->qpos[$(sim.model.m)->jnt_dofadr[mj_name2id($(sim.model.m), mjOBJ_JOINT, "abdomen_z")]] = $thetaTorsoZ;
    $(sim.data.d)->qpos[$(sim.model.m)->jnt_dofadr[mj_name2id($(sim.model.m), mjOBJ_JOINT, "abdomen_y")]] = $thetaTorsoY;
    """
end

function delimVector(pnt)
    str = ""
    for i=1:length(pnt)
        str = "$str$(pnt[i]),"
    end
    return str
end
