import ForwardDiff

abstract AbstractScheme

immutable Scheme{T} <: AbstractScheme
  Δt::Float64
end

typealias Exact Scheme{:Exact}
typealias EulerMaruyama Scheme{:EulerMaruyama}
typealias Milstein Scheme{:Milstein}

subdivide{T}(scheme::Scheme{T}, nsubsteps) = Scheme{T}(scheme.Δt / nsubsteps)

wiener_type(::AbstractSDE{0}) = Float64
wiener_type(::AbstractSDE{1}) = Float64
wiener_type{D}(::AbstractSDE{D}) = SVector{D,Float64}

wiener{D}(::AbstractSDE{D,0}, scheme::AbstractScheme) = 0.0
wiener{D}(::AbstractSDE{D,1}, scheme::AbstractScheme) = sqrt(scheme.Δt) * randn()
wiener{D,M}(::AbstractSDE{D,M}, scheme::AbstractScheme) = sqrt(scheme.Δt) * randn(SVector{M})

function step(model::AbstractSDE, scheme::EulerMaruyama, t, current_state::SDEState, Δw)
  μ = drift(model, t, current_state)
  σ = diffusion(model, t, current_state)
  x = statevalue(current_state) + μ * scheme.Δt + σ * Δw
  SDEState(x)
end

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

# ===================================================
# The following schemes are described in the paper
#   Weak exponential schemes for stochastic
#   differential equations with additive noise
#   CM Mora - IMA journal of numerical analysis, 2005
# ===================================================

typealias EulerExponential1 Scheme{:EulerExponential1}
typealias EulerExponential2 Scheme{:EulerExponential2}
typealias EulerExponential3 Scheme{:EulerExponential3}

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
