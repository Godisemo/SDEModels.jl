immutable ModifiedBridge{D,T,S} <: AbstractScheme
  Δt::Float64
  t1::Float64
  state1::TimeHomogeneousState{D,T,S}
end

subdivide(scheme::ModifiedBridge, nsubsteps) = ModifiedBridge(scheme.Δt / nsubsteps, scheme.state1)

function _step(model, scheme::ModifiedBridge, t, state0::AbstractState, Δw)
  x = statevalue(state0)
  x1 = statevalue(scheme.state1)
  t1 = scheme.t1
  # TODO numerical issues when subdividing
  μ = (x1 - x) / (t1 - t)
  σ = sqrt((t1 - (t + scheme.Δt)) / (t1 - t)) * diffusion(model, state0)
  x + μ * scheme.Δt + σ * Δw
end
