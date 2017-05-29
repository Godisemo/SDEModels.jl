function _normal_transition_params(model, scheme::ImplicitEulerMaruyama, t0, s0, s1)
  t1 = t0 + scheme.Δt
  μ = statevalue(s0) + corrected_drift(model, t1, s1, 1.0) * scheme.Δt
  σ = diffusion(model, t1, s1)
  Σ = scheme.Δt * σ * σ'
  μ, Σ
end
