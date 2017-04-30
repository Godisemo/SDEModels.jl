__precompile__()

module SDEModels

include("base.jl")
include("codegen.jl")
include("models.jl")
include("schemes.jl")
include("simulation.jl")

export @sde_model, dim, drift, diffusion
export EulerMaruyama, Milstein, sample, sample!, simulate, simulate!, subsample

end # module
