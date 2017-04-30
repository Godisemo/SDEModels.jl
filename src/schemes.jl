import ForwardDiff

immutable Scheme{T}
  Δt::Float64
end

typealias EulerMaruyama Scheme{:EulerMaruyama}
typealias Milstein Scheme{:Milstein}

subdivide{T}(scheme::Scheme{T}, nsubsteps) = Scheme{T}(scheme.Δt / nsubsteps)

wiener(::AbstractSDE{TypeVar(:D),0}, scheme::Scheme) = 0.0
wiener(::AbstractSDE{TypeVar(:D),1}, scheme::Scheme) = sqrt(scheme.Δt) * randn()
wiener{M}(::AbstractSDE{TypeVar(:D),M}, scheme::Scheme) = sqrt(scheme.Δt) * randn(M)

function step(model::AbstractSDE, scheme::EulerMaruyama, x, Δw)
  μ = drift(model, x)
  σ = diffusion(model, x)
  x + μ * scheme.Δt + σ * Δw
end

function step{M}(model::AbstractSDE{1,M}, scheme::Milstein, x, Δw)
  μ = drift(model, x)
  # computes the drift and diffusion at the same time efficiently using dual numbers
  σ∂σ = diffusion(model, ForwardDiff.Dual(x, one(x)))
  σ = ForwardDiff.value(σ∂σ)
  ∂σ, = ForwardDiff.partials(σ∂σ)
  x + μ * scheme.Δt + σ * Δw + σ * ∂σ * (Δw^2 - scheme.Δt) / 2
end
