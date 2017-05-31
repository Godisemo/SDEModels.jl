__precompile__()

module SDEModels
using StaticArrays
using Parameters

abstract AbstractSDE{D,M}
abstract StateIndependentDiffusion{D,M} <: AbstractSDE{D,M}

abstract AbstractScheme
abstract ExplicitScheme <: AbstractScheme
abstract ImplicitScheme <: AbstractScheme
abstract ConditionalScheme{T} <: ExplicitScheme
abstract UnconditionalScheme <: ExplicitScheme

include("states.jl")
include("schemes/schemes.jl")
include("models/models.jl")
include("simulation.jl")
include("multilevel.jl")
include("recipes.jl")

export @sde_model, dim, drift, diffusion, corrected_drift, variables, subdivide
export sample, sample!, simulate, simulate!
export Multilevel
export transition, logtransition

end # module
