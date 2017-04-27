import ForwardDiff

abstract AbstractScheme

immutable EulerMaruyama <: AbstractScheme
  Δt::Float64
end

immutable Milstein <: AbstractScheme
  Δt::Float64
end

function sample{D,M}(model::AbstractSDE{D,M}, scheme::EulerMaruyama, x)
  Δw = sqrt(scheme.Δt) * randn(M)
  μ = drift(model, x)
  σ = diffusion(model, x)
  x + μ * scheme.Δt + σ * Δw
end

function sample{D,M}(model::AbstractSDE{D,M}, scheme::Milstein, x)
  Δw = sqrt(scheme.Δt) * randn(M)
  μ = drift(model, x)
  # computes the drift and diffusion at the same time efficiently using dual numbers
  σ∂σ = diffusion(model, ForwardDiff.Dual(x, one(x)))
  σ = ForwardDiff.value(σ∂σ)
  ∂σ, = ForwardDiff.partials(σ∂σ)
  x + μ * scheme.Δt + σ * Δw + σ * ∂σ * (Δw.^2 - scheme.Δt) / 2
end

simulate{D,M}(model::AbstractSDE{D,M}, scheme, x0, N) = simulate!(Array(Float64, D, N), model, scheme, x0, N)
function simulate!{D,M}(x::AbstractArray, model::AbstractSDE{D,M}, scheme::AbstractScheme, x0, N)
  xprev = x0
  for i in 1:N
    xprev = x[:,i] = sample(model, scheme, xprev)
  end
  x
end
