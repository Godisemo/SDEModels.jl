immutable ModifiedBridge{D,T,S} <: AbstractScheme
  Δt::Float64
  t1::Float64
  state1::SDEState{D,T,S}
end

subdivide(scheme::ModifiedBridge, nsubsteps) = ModifiedBridge(scheme.Δt / nsubsteps, scheme.state1)

function step(model, scheme::ModifiedBridge, t, state0::SDEState, Δw)
  x0 = statevalue(state0)
  x1 = statevalue(scheme.state1)
  t1 = scheme.t1
  # TODO numerical issues when subdividing
  μ = (x1 - x0) / (t1 - t)
  σ = sqrt((t1 - (t + scheme.Δt)) / (t1 - t)) * diffusion(model, state0)
  x = x0 + μ * scheme.Δt + σ * Δw
  SDEState(x)
end
