__precompile__()

module SDEModels
using StaticArrays

include("base.jl")
include("codegen.jl")
include("models.jl")
include("schemes.jl")
include("simulation.jl")
include("bridge.jl")
include("multilevel.jl")

export @sde_model, dim, drift, diffusion
export ModifiedBridge, EulerMaruyama, Milstein, sample, sample!, simulate, simulate!, subsample
export Multilevel, multilevel, coarse, fine

end # module
