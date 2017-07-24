import OpenNI
using OpenNI

include("simenvi.jl")

type NIMJSimMapper
    mj_sim::MJSimEnv

    ni_context::NIContext
    ni_usergen::NIUserGenerator
    ni_userID::OpenNI.NIUserID

    pointMapper::Function

    function NIMJSimMapper(sim::MJSimEnv, c::NIContext, gen::NIUserGenerator, uid, mapper)
        new(sim, c, gen, uid, mapper)
    end
end

function MapNIJointsToMJSim(mapper::NIMJSimMapper, jointPos)
    mjPos = deepcopy(jointPos[:, 1])
    for i=1:length(jointPos[:, 1])
        mjPos[i, :] = mapper.pointMapper(mapper, jointPos[i, :])
    end
    return mjPos
end
