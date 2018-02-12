macro unconditional_scheme(name)
  quote
    struct $name <: UnconditionalScheme
      Δt::Float64
    end
    $name(Δt, t, s) = $name(Δt)
    export $name
  end |> esc
end

macro implicit_scheme(name)
  quote
    struct $name <: ImplicitScheme
      Δt::Float64
    end
    $name(Δt, t, s) = $name(Δt)
    export $name
  end |> esc
end

macro conditional_scheme(name)
  quote
    struct $name{T} <: ConditionalScheme{T}
      Δt::Float64
      t::Float64
      s::T
    end
    export $name
  end |> esc
end

@unconditional_scheme Exact
@unconditional_scheme EulerMaruyama
@implicit_scheme      ImplicitEulerMaruyama
@unconditional_scheme Milstein
@conditional_scheme   ModifiedBridge
@unconditional_scheme EulerExponential1
@unconditional_scheme EulerExponential2
@unconditional_scheme EulerExponential3

include("euler_maruyama.jl")
include("implicit_euler_maruyama.jl")
include("milstein.jl")
include("modified_bridge.jl")
include("euler_exponential.jl")

subdivide{T<:ConditionalScheme}(scheme::T, nsubsteps) = T(scheme.Δt / nsubsteps, scheme.t, scheme.s)
subdivide{T<:UnconditionalScheme}(scheme::T, nsubsteps) = T(scheme.Δt / nsubsteps)

const NormalScheme = Union{EulerMaruyama,ImplicitEulerMaruyama,ModifiedBridge,EulerExponential3}

function transition{D}(model::AbstractSDE{D}, scheme::NormalScheme, t0, x0, x1)
  μ, Σ = _normal_transition_params(model, scheme, t0, x0, x1)
  z = μ - x1
  1.0 / sqrt(det(2pi*Σ)) * exp(-0.5*dot(z, Σ\z))
end

function logtransition{D}(model::AbstractSDE{D}, scheme::NormalScheme, t0, x0, x1)
  μ, Σ = _normal_transition_params(model, scheme, t0, x0, x1)
  z = μ - x1
  -0.5*(D*log(2pi) + log(det(Σ)) + dot(z, Σ\z))
end

function transition(model, scheme, times, data)
  [transition(model, scheme, times[i-1], data[i-1], data[i]) for i in 2:length(data)]
end

function logtransition(model, scheme, times, data)
  [logtransition(model, scheme, times[i-1], data[i-1], data[i]) for i in 2:length(data)]
end
