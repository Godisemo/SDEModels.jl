import ForwardDiff

function step{M}(model::AbstractSDE{1,M}, scheme::Milstein, t, current_state::SDEState, Δw)
  μ = drift(model, t, current_state)
  # computes the drift and diffusion at the same time efficiently using dual numbers
  dual = ForwardDiff.Dual(statevalue(current_state), one(statevalue(current_state)))
  σ∂σ = diffusion(model, t, SDEState(dual))
  σ = ForwardDiff.value(σ∂σ)
  ∂σ, = ForwardDiff.partials(σ∂σ)
  x = statevalue(current_state) + μ * scheme.Δt + σ * Δw + σ * ∂σ * (Δw^2 - scheme.Δt) / 2
  SDEState(x)
end
