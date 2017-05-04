immutable ModifiedBridge{T} <: AbstractScheme
  Δt::Float64
  t1::Float64
  s1::T
end

subdivide(scheme::ModifiedBridge, nsubsteps) = ModifiedBridge(scheme.Δt / nsubsteps, scheme.t1, scheme.s1)

function step(model, scheme::ModifiedBridge, t0, s0::SDEState, Δw)
  x0 = statevalue(s0)
  x1 = statevalue(scheme.s1)
  t1 = scheme.t1
  μ = (x1 - x0) / (t1 - t0)
  σ = sqrt((t1 - (t0 + scheme.Δt)) / (t1 - t0)) * diffusion(model, t0, s0)
  x = x0 + μ * scheme.Δt + σ * Δw
  SDEState(x)
end
