function step(model, scheme::ModifiedBridge, t0, x0, Δw)
  x1 = scheme.s
  t1 = scheme.t
  μ = (x1 - x0) / (t1 - t0)
  σ = sqrt((t1 - (t0 + scheme.Δt)) / (t1 - t0)) * diffusion(model, t0, x0)
  x = x0 + μ * scheme.Δt + σ * Δw
  x
end

function _normal_transition_params(model, scheme::ModifiedBridge, t0, x0, s1)
  x1 = scheme.s
  t1 = scheme.t
  μ = x0 + (x1 - x0) / (t1 - t0) * scheme.Δt
  σ = diffusion(model, t0, x0)
  Σ = (t1 - (t0 + scheme.Δt)) / (t1 - t0) * scheme.Δt * σ * σ'
  μ, Σ
end
