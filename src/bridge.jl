immutable ModifiedBridge{D,T,S} <: AbstractScheme
  Δt::Float64
  state1::TimeDependentState{D,T,S}
end

subdivide(scheme::ModifiedBridge, nsubsteps) = ModifiedBridge(scheme.Δt / nsubsteps, scheme.state1)

function _step(model, scheme::ModifiedBridge, state0::TimeDependentState, Δw)
  x = statevalue(state0)
  t = statetime(state0)
  x1 = statevalue(scheme.state1)
  t1 = statetime(scheme.state1)
  # TODO numerical issues when subdividing
  μ = (x1 - x) / (t1 - t)
  σ = sqrt((t1 - (t + scheme.Δt)) / (t1 - t)) * diffusion(model, state0)
  x + μ * scheme.Δt + σ * Δw
end
