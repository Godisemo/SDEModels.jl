function step(model::AbstractSDE{D,1}, scheme::RungeKutta, t0, x0, Δw) where D
  μ = drift(model, t0, x0)
  σ = diffusion(model, t0, x0)
  x0hat = x0 + μ * scheme.Δt + σ * sqrt(scheme.Δt)
  σpred = diffusion(model, t0, x0hat)
  x = x0 + μ * scheme.Δt + σ * Δw + (σpred - σ) * (Δw^2 - scheme.Δt) / (2 * sqrt(scheme.Δt))
  x
end
