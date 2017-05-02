abstract AbstractSDE{D,M}

dim{D,M}(::AbstractSDE{D,M}) = (D,M)
dim{D,M}(::Type{AbstractSDE{D,M}}) = (D,M)
dim{T<:AbstractSDE}(::Type{T}) = dim(supertype(T))

model_dim{D,M}(::AbstractSDE{D,M}) = D
noise_dim{D,M}(::AbstractSDE{D,M}) = M

drift(::AbstractSDE, x) = nothing
diffusion(::AbstractSDE, x) = nothing



abstract AbstractState{D,S,T}

immutable TimeDependentState{D,S,T} <: AbstractState{D,S,T}
  x::T
  t::Float64
end

TimeDependentState{T<:Number}(x::T, t::Float64) = TimeDependentState{1,T,T}(x, t)
TimeDependentState{D,T}(x::SVector{D,T}, t::Float64) = TimeDependentState{D,T,SVector{D,T}}(x, t)
TimeDependentState(x::AbstractVector, t::Float64) = TimeDependentState(convert(SVector{length(x)}, x), t)

immutable TimeHomogeneousState{D,S,T} <: AbstractState{D,S,T}
  x::T
end

TimeHomogeneousState{T<:Number}(x::T) = TimeHomogeneousState{1,T,T}(x)
TimeHomogeneousState{D,T}(x::SVector{D,T}) = TimeHomogeneousState{D,T,SVector{D,T}}(x)
TimeHomogeneousState(x::AbstractVector) = TimeHomogeneousState(convert(SVector{length(x)}, x))

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
