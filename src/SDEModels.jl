__precompile__()

module SDEModels
using StaticArrays
using Parameters

abstract type AbstractSDE{D,M} end
abstract type StateIndependentDiffusion{D,M} <: AbstractSDE{D,M} end

abstract type AbstractScheme end
abstract type ExplicitScheme <: AbstractScheme end
abstract type ImplicitScheme <: AbstractScheme end
abstract type ConditionalScheme{T} <: ExplicitScheme end
abstract type UnconditionalScheme <: ExplicitScheme end

include("schemes/schemes.jl")
include("models/models.jl")
include("simulation.jl")
include("multilevel.jl")

export @sde_model, dim, drift, diffusion, corrected_drift, variables, subdivide
export sample, sample!, simulate, simulate!
export MultilevelScheme, npaths
export transition, logtransition

end # module
