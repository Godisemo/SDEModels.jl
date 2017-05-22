function step(model::AbstractSDE, scheme::EulerMaruyama, t0, s0, Δw)
  μ = drift(model, t0, s0)
  σ = diffusion(model, t0, s0)
  x = statevalue(s0) + μ * scheme.Δt + σ * Δw
  SDEState(x)
end

function _normal_transition_params(model, scheme::EulerMaruyama, t0, s0, s1)
  x0 = statevalue(s0)
  x1 = statevalue(s1)
  μ = x0 + drift(model, t0, s0) * scheme.Δt
  σ = diffusion(model, t0, s0)
  Σ = scheme.Δt * σ * σ'
  z = x1 - μ
  z, Σ
end
