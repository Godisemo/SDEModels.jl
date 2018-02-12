function step(model::AbstractSDE, scheme::EulerMaruyama, t0, x0, Δw)
  μ = drift(model, t0, x0)
  σ = diffusion(model, t0, x0)
  x = x0 + μ * scheme.Δt + σ * Δw
  x
end

function _normal_transition_params(model, scheme::EulerMaruyama, t0, x0, x1)
  μ = x0 + drift(model, t0, x0) * scheme.Δt
  σ = diffusion(model, t0, x0)
  Σ = scheme.Δt * σ * σ'
  μ, Σ
end
