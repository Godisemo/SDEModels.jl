abstract AbstractSDE{D,M}

dim{D,M}(::AbstractSDE{D,M}) = (D,M)
dim{D,M}(::Type{AbstractSDE{D,M}}) = (D,M)
dim{T<:AbstractSDE}(::Type{T}) = dim(supertype(T))

model_dim{D,M}(::AbstractSDE{D,M}) = D
noise_dim{D,M}(::AbstractSDE{D,M}) = M

abstract AbstractState{D,S,T}

immutable TimeHomogeneousState{D,S,T} <: AbstractState{D,S,T}
  x::T
end

TimeHomogeneousState{T<:Number}(x::T) = TimeHomogeneousState{1,T,T}(x)
TimeHomogeneousState{D,T}(x::SVector{D,T}) = TimeHomogeneousState{D,T,SVector{D,T}}(x)
TimeHomogeneousState(x::AbstractVector) = TimeHomogeneousState(convert(SVector{length(x)}, x))

state(x) = TimeHomogeneousState(x)

statevalue(state::AbstractState) = state.x

statevalue(A::AbstractArray) = reshape([ statevalue(x) for x in A ], size(A))

export TimeDependentState, TimeHomogeneousState, state, statevalue, statetime

Base.show(io::IO, s::TimeHomogeneousState) = show(io, s.x)

drift{D}(::AbstractSDE{D}, t::Number, x::AbstractState{D}) = error("drift is not implemented for this model")
diffusion{D}(::AbstractSDE{D}, t::Number, x::AbstractState{D}) = error("diffusion is not implemented for this model")
variables(::AbstractSDE) = error("variables is not implemented for this model")
