abstract AbstractSDE{D,M}
abstract StateIndependentDiffusion{D,M} <: AbstractSDE{D,M}

dim{D,M}(::AbstractSDE{D,M}) = (D,M)
dim{D,M}(::Type{AbstractSDE{D,M}}) = (D,M)
dim{T<:AbstractSDE}(::Type{T}) = dim(supertype(T))

model_dim{D,M}(::AbstractSDE{D,M}) = D
noise_dim{D,M}(::AbstractSDE{D,M}) = M

immutable SDEState{D,S,T}
  x::T
end

SDEState{T<:Number}(x::T) = SDEState{1,T,T}(x)
SDEState{D,T}(x::SVector{D,T}) = SDEState{D,T,SVector{D,T}}(x)
SDEState(x::AbstractVector) = SDEState(convert(SVector{length(x)}, x))

state(x) = SDEState(x)

statevalue(state::SDEState) = state.x

statevalue(A::AbstractArray) = reshape([ statevalue(x) for x in A ], size(A))

export TimeDependentState, SDEState, state, statevalue, statetime

Base.show(io::IO, s::SDEState) = show(io, s.x)

drift{D}(::AbstractSDE{D}, t::Number, x::SDEState{D}) = error("drift is not implemented for this model")
drift_jacobian{D}(::AbstractSDE{D}, t::Number, x::SDEState{D}) = error("drift jacobian is not implemented for this model")
diffusion{D}(::AbstractSDE{D}, t::Number, x::SDEState{D}) = error("diffusion is not implemented for this model")
variables(::AbstractSDE) = error("variables is not implemented for this model")
