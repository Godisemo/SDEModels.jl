type MultilevelState{T<:AbstractState}
  xc::T
  xf::T
end

coarse(state::MultilevelState) = state.xc
fine(state::MultilevelState) = state.xf

coarse(A::AbstractArray) = reshape([ coarse(x) for x in A ], size(A))
fine(A::AbstractArray) = reshape([ fine(x) for x in A ], size(A))

multilevel(xc, xf) = MultilevelState(xc, xf)
multilevel(x) = MultilevelState(x, x)

Base.copy(s::MultilevelState) = MultilevelState(s.xc, s.xf)

function sample(model, scheme, state0::MultilevelState; nsubsteps=2)
  subscheme = subdivide(scheme, nsubsteps)
  prevstate = copy(state0)
  Δw = 0.0
  for j in 1:nsubsteps
    Δw += δw = wiener(model, subscheme)
    prevstate.xf = step(model, subscheme, prevstate.xf, δw)
  end
  prevstate.xc = step(model, scheme, prevstate.xc, Δw)
  prevstate
end
