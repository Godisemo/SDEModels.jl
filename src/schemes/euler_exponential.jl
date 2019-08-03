import LinearAlgebra: I

# ===================================================
# The following schemes are described in the paper
#   Weak exponential schemes for stochastic
#   differential equations with additive noise
#   CM Mora - IMA journal of numerical analysis, 2005
# ===================================================

_drift_jacobian(model::AbstractSDE{D,M}, t0, x0) where {D,M} = ForwardDiff.jacobian(x->drift(model, t0, x), x0)
_drift_jacobian(model::AbstractSDE{1,M}, t0, x0) where {M} = ForwardDiff.derivative(x->drift(model, t0, x), x0)

function step(model::StateIndependentDiffusion, scheme::EulerExponential1, t0, x0, Δw)
  Jμ = _drift_jacobian(model, t0, x0)
  μ = drift(model, t0, x0)
  σ = diffusion(model, t0, x0)
  x = exp(Jμ * scheme.Δt) * (x0 + (μ - Jμ * x0) * scheme.Δt + σ * Δw)
  x
end

function step(model::StateIndependentDiffusion, scheme::EulerExponential2, t0, x0, Δw)
  Jμ = _drift_jacobian(model, t0, x0)
  μ = drift(model, t0, x0)
  σ = diffusion(model, t0, x0)
  x = (I - Jμ * scheme.Δt) \ (x0 + (μ - Jμ * x0) * scheme.Δt + σ * Δw)
  x
end

function step(model::StateIndependentDiffusion, scheme::EulerExponential3, t0, x0, Δw)
  Jμ = _drift_jacobian(model, t0, x0)
  expmJμ = exp(0.5 * Jμ * scheme.Δt)
  μ = drift(model, t0, x0)
  σ = diffusion(model, t0, x0)
  x = expmJμ * (expmJμ * x0 + (μ - Jμ * x0) * scheme.Δt + σ * Δw)
  x
end

function _normal_transition_params(model, scheme::EulerExponential3, t0, x0, x1)
  Jμ = _drift_jacobian(model, t0, x0)
  expmJμ = exp(0.5 * Jμ * scheme.Δt)
  μ = expmJμ * (expmJμ * x0 + (drift(model, t0, x0) - Jμ * x0) * scheme.Δt)
  σ = expmJμ * diffusion(model, t0, x0)
  Σ = scheme.Δt * σ * σ'
  μ, Σ
end
