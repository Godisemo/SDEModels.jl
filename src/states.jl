immutable SDEState{D,S,T}
  x::T
end

SDEState{T<:Number}(x::T) = SDEState{1,T,T}(x)
SDEState{D,T}(x::SVector{D,T}) = SDEState{D,T,SVector{D,T}}(x)
SDEState(x::AbstractVector) = SDEState(convert(SVector{length(x)}, x))

state(x) = SDEState(x)

statevalue(state::SDEState) = state.x

statevalue(A::AbstractArray) = reshape([ statevalue(x) for x in A ], size(A))

export SDEState, state, statevalue

Base.show(io::IO, s::SDEState) = show(io, s.x)
