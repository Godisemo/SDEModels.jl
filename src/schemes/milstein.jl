_diffusion_jacobian(model::AbstractSDE{D,M}, t0, x0) where {D,M} = ForwardDiff.jacobian(x->diffusion(model, t0, x), x0)
_diffusion_jacobian(model::AbstractSDE{1,M}, t0, x0) where {M} = ForwardDiff.derivative(x->diffusion(model, t0, x), x0)

function step(model::AbstractSDE{D,1}, scheme::Milstein, t0, x0, Δw) where D
  μ = drift(model, t0, x0)
  σ = diffusion(model, t0, x0)
  ∂σ = _diffusion_jacobian(model, t0, x0)
  x = x0 + μ * scheme.Δt + σ * Δw + ∂σ * σ * (Δw.^2 - scheme.Δt) / 2
  x
end
