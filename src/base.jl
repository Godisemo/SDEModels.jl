abstract AbstractSDE{D,M}

dim{D,M}(::AbstractSDE{D,M}) = (D,M)
dim{D,M}(::Type{AbstractSDE{D,M}}) = (D,M)
dim{T<:AbstractSDE}(::Type{T}) = dim(supertype(T))

model_dim{D,M}(::AbstractSDE{D,M}) = D
noise_dim{D,M}(::AbstractSDE{D,M}) = M

drift(::AbstractSDE, x) = nothing
diffusion(::AbstractSDE, x) = nothing



abstract AbstractState{S,T}

immutable TimeDependentState{S,T} <: AbstractState{S,T}
  x::T
  t::Float64
end

TimeDependentState(x, t) = TimeDependentState{eltype(x),typeof(x)}(x, float(t))

immutable TimeHomogeneousState{S,T} <: AbstractState{S,T}
  x::T
end

TimeHomogeneousState(x) = TimeHomogeneousState{eltype(x),typeof(x)}(x)


state(x) = TimeHomogeneousState(x)
state(x, t) = TimeDependentState(x, t)

Base.eltype{T<:AbstractState}(::Type{T}) = T

statevalue(state::AbstractState) = state.x
statetime(state::TimeDependentState) = state.t
statetime(state::TimeHomogeneousState) = nothing

statevalue(A::AbstractArray) = reshape([ statevalue(x) for x in A ], size(A))
statetime(A::AbstractArray) = reshape([ statetime(x) for x in A ], size(A))

export TimeDependentState, TimeHomogeneousState, state, statevalue, statetime

Base.show(io::IO, s::TimeDependentState) = show(io, (s.x, s.t))
Base.show(io::IO, s::TimeHomogeneousState) = show(io, s.x)
