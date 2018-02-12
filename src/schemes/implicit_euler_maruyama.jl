function _normal_transition_params(model, scheme::ImplicitEulerMaruyama, t0, x0, x1)
  t1 = t0 + scheme.Δt
  μ = x0 + corrected_drift(model, t1, x1, 1.0) * scheme.Δt
  σ = diffusion(model, t1, x1)
  Σ = scheme.Δt * σ * σ'
  μ, Σ
end
