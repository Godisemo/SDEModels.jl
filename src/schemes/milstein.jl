import ForwardDiff

function step(model::AbstractSDE{1,M}, scheme::Milstein, t0, x0, Δw) where M
  μ = drift(model, t0, x0)
  # computes the drift and diffusion at the same time efficiently using dual numbers
  dual = ForwardDiff.Dual(x0, one(x0))
  σ∂σ = diffusion(model, t0, dual)
  σ = ForwardDiff.value(σ∂σ)
  ∂σ, = ForwardDiff.partials(σ∂σ)
  x = x0 + μ * scheme.Δt + σ * Δw + σ * ∂σ * (Δw^2 - scheme.Δt) / 2
  x
end
