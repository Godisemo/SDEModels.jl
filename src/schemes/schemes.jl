macro unconditional_scheme(name)
  quote
    immutable $name <: UnconditionalScheme
      Δt::Float64
    end
    export $name
  end |> esc
end

macro conditional_scheme(name)
  quote
    immutable $name{T} <: ConditionalScheme{T}
      Δt::Float64
      t::Float64
      s::T
    end
    export $name
  end |> esc
end

@unconditional_scheme Exact
@unconditional_scheme EulerMaruyama
@unconditional_scheme Milstein
@conditional_scheme   ModifiedBridge
@unconditional_scheme EulerExponential1
@unconditional_scheme EulerExponential2
@unconditional_scheme EulerExponential3

include("euler_maruyama.jl")
include("milstein.jl")
include("modified_bridge.jl")
include("euler_exponential.jl")

subdivide{T<:ConditionalScheme}(scheme::T, nsubsteps) = T(scheme.Δt / nsubsteps, scheme.t, scheme.s)
subdivide{T<:UnconditionalScheme}(scheme::T, nsubsteps) = T(scheme.Δt / nsubsteps)

function transition(model, scheme, times, data)
  [transition(model, scheme, times[i-1], data[i-1], data[i]) for i in 2:length(data)]
end

function logtransition(model, scheme, times, data)
  [logtransition(model, scheme, times[i-1], data[i-1], data[i]) for i in 2:length(data)]
end
