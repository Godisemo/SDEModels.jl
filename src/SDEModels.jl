__precompile__()

module SDEModels
using StaticArrays
using Parameters

include("base.jl")
include("schemes.jl")
include("models/models.jl")
include("simulation.jl")
include("bridge.jl")
include("multilevel.jl")
include("inference.jl")
include("recipes.jl")

export @sde_model, dim, drift, diffusion, variables
export Exact, ModifiedBridge, EulerMaruyama, Milstein,
       EulerExponential1, EulerExponential2, EulerExponential3
export sample, sample!, simulate, simulate!, subsample
export multilevel, coarse, fine
export transition, logtransition

end # module
