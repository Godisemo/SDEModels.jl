immutable MultilevelState{D,S,T} <: AbstractState{D,S,T}
  xc::T
  xf::T
  t::Float64
end

MultilevelState{T<:Number}(xc::T, xf::T, t::Float64) =
  MultilevelState{1,T,T}(xc, xf, t)
MultilevelState{D,T}(xc::SVector{D,T}, xf::SVector{D,T}, t::Float64) =
  MultilevelState{D,T,SVector{D,T}}(xc, xf, t)
MultilevelState(xc::AbstractVector, xf::AbstractVector, t::Float64) =
  MultilevelState(convert(SVector{length(xc)}, xc), convert(SVector{length(xf)}, xf), t)

coarse(state::MultilevelState) = TimeDependentState(state.xc, state.t)
fine(state::MultilevelState) = TimeDependentState(state.xf, state.t)

coarse(A::AbstractArray) = reshape([ coarse(x) for x in A ], size(A))
fine(A::AbstractArray) = reshape([ fine(x) for x in A ], size(A))

multilevel(xc, xf, t) = MultilevelState(xc, xf, t)
multilevel(x, t) = MultilevelState(x, x, t)

function sample(model, scheme, state0::MultilevelState; nsubsteps=2)
  subscheme = subdivide(scheme, nsubsteps)
  prevstate_coarse = coarse(state0)
  prevstate_fine = fine(state0)
  Δw = 0.0
  for j in 1:nsubsteps
    Δw += δw = wiener(model, subscheme)
    prevstate_fine = step(model, subscheme, prevstate_fine, δw)
  end
  prevstate_coarse = step(model, scheme, prevstate_coarse, Δw)
  MultilevelState(statevalue(prevstate_coarse), statevalue(prevstate_fine), statetime(prevstate_coarse))
end
