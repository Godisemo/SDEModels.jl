# ===================================================
# The following schemes are described in the paper
#   Weak exponential schemes for stochastic
#   differential equations with additive noise
#   CM Mora - IMA journal of numerical analysis, 2005
# ===================================================

# TODO temporary convenience method, should be removed when expm is in StaticArrays.jl
@inline Base.expm{T<:StaticMatrix}(x::T) = T(expm(Array(x)))

function step(model::StateIndependentDiffusion, scheme::EulerExponential1, t, current_state::SDEState, Δw)
  Jμ = drift_jacobian(model, t, current_state)
  μ = drift(model, t, current_state)
  σ = diffusion(model, t, current_state)
  xprev = statevalue(current_state)
  x = expm(Jμ * scheme.Δt) * (xprev + (μ - Jμ * xprev) * scheme.Δt + σ * Δw)
  SDEState(x)
end

function step(model::StateIndependentDiffusion, scheme::EulerExponential2, t, current_state::SDEState, Δw)
  Jμ = drift_jacobian(model, t, current_state)
  μ = drift(model, t, current_state)
  σ = diffusion(model, t, current_state)
  xprev = statevalue(current_state)
  x = (I - Jμ * scheme.Δt) \ (xprev + (μ - Jμ * xprev) * scheme.Δt + σ * Δw)
  SDEState(x)
end

function step(model::StateIndependentDiffusion, scheme::EulerExponential3, t, current_state::SDEState, Δw)
  Jμ = drift_jacobian(model, t, current_state)
  expmJμ = expm(0.5 * Jμ * scheme.Δt)
  μ = drift(model, t, current_state)
  σ = diffusion(model, t, current_state)
  xprev = statevalue(current_state)
  x = expmJμ * (expmJμ * xprev + (μ - Jμ * xprev) * scheme.Δt + σ * Δw)
  SDEState(x)
end
