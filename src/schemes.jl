import ForwardDiff

abstract AbstractScheme

immutable Scheme{T} <: AbstractScheme
  Δt::Float64
end

typealias EulerMaruyama Scheme{:EulerMaruyama}
typealias Milstein Scheme{:Milstein}

subdivide{T}(scheme::Scheme{T}, nsubsteps) = Scheme{T}(scheme.Δt / nsubsteps)

wiener{D}(::AbstractSDE{D,0}, scheme::AbstractScheme) = 0.0
wiener{D}(::AbstractSDE{D,1}, scheme::AbstractScheme) = sqrt(scheme.Δt) * randn()
wiener{D,M}(::AbstractSDE{D,M}, scheme::AbstractScheme) = sqrt(scheme.Δt) * randn(SVector{M})

step(model, scheme, state0::TimeDependentState, Δw) =
  TimeDependentState(_step(model, scheme, state0, Δw), statetime(state0) + scheme.Δt)

step(model, scheme, state0::TimeHomogeneousState, Δw) =
  TimeHomogeneousState(_step(model, scheme, state0, Δw))

function _step(model::AbstractSDE, scheme::EulerMaruyama, current_state::AbstractState, Δw)
  μ = drift(model, current_state)
  σ = diffusion(model, current_state)
  statevalue(current_state) + μ * scheme.Δt + σ * Δw
end

function _step{M}(model::AbstractSDE{1,M}, scheme::Milstein, current_state::AbstractState, Δw)
  μ = drift(model, current_state)
  # computes the drift and diffusion at the same time efficiently using dual numbers
  σ∂σ = diffusion(model, state(ForwardDiff.Dual(statevalue(current_state), one(statevalue(current_state)))))
  σ = ForwardDiff.value(σ∂σ)
  ∂σ, = ForwardDiff.partials(σ∂σ)
  statevalue(current_state) + μ * scheme.Δt + σ * Δw + σ * ∂σ * (Δw^2 - scheme.Δt) / 2
end
