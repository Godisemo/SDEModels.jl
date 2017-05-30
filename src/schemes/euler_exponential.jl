# ===================================================
# The following schemes are described in the paper
#   Weak exponential schemes for stochastic
#   differential equations with additive noise
#   CM Mora - IMA journal of numerical analysis, 2005
# ===================================================

function step(model::StateIndependentDiffusion, scheme::EulerExponential1, t0, s0, Δw)
  Jμ = drift_jacobian(model, t0, s0)
  μ = drift(model, t0, s0)
  σ = diffusion(model, t0, s0)
  xprev = statevalue(s0)
  x = expm(Jμ * scheme.Δt) * (xprev + (μ - Jμ * xprev) * scheme.Δt + σ * Δw)
  SDEState(x)
end

function step(model::StateIndependentDiffusion, scheme::EulerExponential2, t0, s0, Δw)
  Jμ = drift_jacobian(model, t0, s0)
  μ = drift(model, t0, s0)
  σ = diffusion(model, t0, s0)
  xprev = statevalue(s0)
  x = (I - Jμ * scheme.Δt) \ (xprev + (μ - Jμ * xprev) * scheme.Δt + σ * Δw)
  SDEState(x)
end

function step(model::StateIndependentDiffusion, scheme::EulerExponential3, t0, s0, Δw)
  Jμ = drift_jacobian(model, t0, s0)
  expmJμ = expm(0.5 * Jμ * scheme.Δt)
  μ = drift(model, t0, s0)
  σ = diffusion(model, t0, s0)
  xprev = statevalue(s0)
  x = expmJμ * (expmJμ * xprev + (μ - Jμ * xprev) * scheme.Δt + σ * Δw)
  SDEState(x)
end

function _normal_transition_params(model, scheme::EulerExponential3, t0, s0, s1)
  x0 = statevalue(s0)
  Jμ = drift_jacobian(model, t0, s0)
  expmJμ = expm(0.5 * Jμ * scheme.Δt)
  μ = expmJμ * (expmJμ * x0 + (drift(model, t0, s0) - Jμ * x0) * scheme.Δt)
  σ = expmJμ * diffusion(model, t0, s0)
  Σ = scheme.Δt * σ * σ'
  μ, Σ
end
