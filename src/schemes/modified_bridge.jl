function step(model, scheme::ModifiedBridge, t0, s0, Δw)
  x0 = statevalue(s0)
  x1 = statevalue(scheme.s)
  t1 = scheme.t
  μ = (x1 - x0) / (t1 - t0)
  σ = sqrt((t1 - (t0 + scheme.Δt)) / (t1 - t0)) * diffusion(model, t0, s0)
  x = x0 + μ * scheme.Δt + σ * Δw
  SDEState(x)
end

function _normal_transition_params(model, scheme::ModifiedBridge, t0, s0, s1)
  x0 = statevalue(s0)
  x1 = statevalue(scheme.s)
  t1 = scheme.t
  μ = x0 + (x1 - x0) / (t1 - t0) * scheme.Δt
  σ = diffusion(model, t0, s0)
  Σ = (t1 - (t0 + scheme.Δt)) / (t1 - t0) * scheme.Δt * σ * σ'
  μ, Σ
end
