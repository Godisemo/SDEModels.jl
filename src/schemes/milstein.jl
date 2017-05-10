import ForwardDiff

function step{M}(model::AbstractSDE{1,M}, scheme::Milstein, t0, s0, Δw)
  μ = drift(model, t0, s0)
  # computes the drift and diffusion at the same time efficiently using dual numbers
  dual = ForwardDiff.Dual(statevalue(s0), one(statevalue(s0)))
  σ∂σ = diffusion(model, t0, SDEState(dual))
  σ = ForwardDiff.value(σ∂σ)
  ∂σ, = ForwardDiff.partials(σ∂σ)
  x = statevalue(s0) + μ * scheme.Δt + σ * Δw + σ * ∂σ * (Δw^2 - scheme.Δt) / 2
  SDEState(x)
end
