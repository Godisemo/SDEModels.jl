__precompile__()

module SDEModels
using StaticArrays
using Parameters

abstract AbstractSDE{D,M}
abstract StateIndependentDiffusion{D,M} <: AbstractSDE{D,M}

abstract AbstractScheme
abstract ConditionalScheme{T} <: AbstractScheme
abstract UnconditionalScheme <: AbstractScheme

include("states.jl")
include("schemes/schemes.jl")
include("models/models.jl")
include("simulation.jl")
include("multilevel.jl")
include("recipes.jl")

export @sde_model, dim, drift, diffusion, variables, subdivide
export sample, sample!, simulate, simulate!
export Multilevel
export transition, logtransition

end # module
