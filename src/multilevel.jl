
# heavily inspired by base/complex.jl

immutable Multilevel{T<:Number}
  coarse::T
  fine::T
end

Multilevel(x::Number, y::Number) = Multilevel(promote(x,y)...)
Multilevel(x::Number) = Multilevel(x, x)

Base.convert{T<:Number}(::Type{Multilevel{T}}, x::Number) = Multilevel{T}(x,x)

multilevel(x::Number) = Multilevel(x, x)
multilevel(x::Number, y::Number) = Multilevel(x, y)
multilevel{T}(A::AbstractArray{T}) = Base.convert(AbstractArray{typeof(Multilevel(zero(T)))}, A)
function multilevel{S<:Number,T<:Number}(A::AbstractArray{S}, B::AbstractArray{T})
    if size(A) != size(B); throw(DimensionMismatch()); end
    F = similar(A, typeof(multilevel(zero(S),zero(T))))
    for (iF, iA, iB) in zip(eachindex(F), eachindex(A), eachindex(B))
        @inbounds F[iF] = multilevel(A[iA], B[iB])
    end
    return F
end

Base.eltype{T<:Multilevel}(::Type{T}) = T
Base.eltype{T}(::Multilevel{T}) = Multilevel{T}

coarse(x::Multilevel) = x.coarse
fine(x::Multilevel) = x.fine

coarse(A::AbstractArray) = reshape([ coarse(x) for x in A ], size(A))
fine(A::AbstractArray) = reshape([ fine(x) for x in A ], size(A))

Base.show(io::IO, x::Multilevel) = show(io, (x.coarse, x.fine))


function sample{T<:Multilevel}(model, scheme, state0::TimeDependentState{T}; nsubsteps=2)
  subscheme = subdivide(scheme, nsubsteps)
  xcprev = state(coarse(statevalue(state0)), statetime(state0))
  xfprev = state(fine(statevalue(state0)), statetime(state0))
  Δw = 0.0
  for j in 1:nsubsteps
    Δw += δw = wiener(model, subscheme)
    xfprev = step(model, subscheme, xfprev, δw)
  end
  xcprev = step(model, scheme, xcprev, Δw)
  state(multilevel(statevalue(xcprev), statevalue(xfprev)), statetime(xcprev))
end
