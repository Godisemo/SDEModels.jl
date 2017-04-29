import ForwardDiff

abstract AbstractScheme

immutable EulerMaruyama <: AbstractScheme
  Δt::Float64
end

immutable Milstein <: AbstractScheme
  Δt::Float64
end

wiener{D}(::AbstractSDE{D,0}, Δt) = 0.0
wiener{D}(::AbstractSDE{D,1}, Δt) = sqrt(Δt) * randn()
wiener{D,M}(::AbstractSDE{D,M}, Δt) = sqrt(Δt) * randn(M)

function step(model::AbstractSDE, ::EulerMaruyama, x, Δt, Δw)
  μ = drift(model, x)
  σ = diffusion(model, x)
  x + μ * Δt + σ * Δw
end

function step{M}(model::AbstractSDE{1,M}, ::Milstein, x, Δt, Δw)
  μ = drift(model, x)
  # computes the drift and diffusion at the same time efficiently using dual numbers
  σ∂σ = diffusion(model, ForwardDiff.Dual(x, one(x)))
  σ = ForwardDiff.value(σ∂σ)
  ∂σ, = ForwardDiff.partials(σ∂σ)
  x + μ * Δt + σ * Δw + σ * ∂σ * (Δw^2 - Δt) / 2
end

simulate{D,M}(model::AbstractSDE{D,M}, scheme, x0, N; substeps=1) =
  simulate!(Array(Float64, D, N), model, scheme, x0; substeps=substeps)
function simulate!(x, model::AbstractSDE, scheme::AbstractScheme, x0; substeps=1)
  xprev = x0
  for i in 1:size(x, 2)
    xprev = x[:,i] = sample(model, scheme, xprev; substeps=substeps)
  end
  x
end

function sample(model::AbstractSDE, scheme::AbstractScheme, x0; substeps=1)
  δt = scheme.Δt / substeps
  xprev = x0
  for i in 1:substeps
    xprev = step(model, scheme, xprev, δt, wiener(model, δt))
  end
  xprev
end
