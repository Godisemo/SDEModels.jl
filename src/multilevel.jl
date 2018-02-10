struct MultilevelScheme{T<:AbstractScheme}
  scheme::T
  substeps::Int64
  levels::UnitRange{Int64}
end

function schemes(s::MultilevelScheme)
  subdivide.(s.scheme, s.substeps.^s.levels)
end

function npaths(s::MultilevelScheme, max_budget)
  budget = max_budget
  nlevels = length(s.levels)
  steps = s.substeps.^s.levels
  n_total = zeros(Int64, nlevels)
  for i in nlevels:-1:1
    level_budget = minimum(floor.(Int64, budget / length(s.levels[1:i]) ./ steps[1:i]) .* steps[1:i])
    n_partial = div.(level_budget, steps[1:i])
    n_total[1:i] += n_partial
    budget -= dot(n_partial, steps[1:i])
  end
  n_total
end

function cost(s::MultilevelScheme, npaths)
  steps = s.substeps.^s.levels
  s.substeps.^s.levels .* npaths
end

simulate(model, scheme::MultilevelScheme, t0, s0, nsteps, npaths::Int64) =
  simulate(model, scheme::MultilevelScheme, t0, s0, nsteps, fill(npaths, length(scheme.levels)))

sample(model, scheme::MultilevelScheme, t0, s0, nsteps, npaths::Int64) =
  sample(model, scheme::MultilevelScheme, t0, s0, nsteps, fill(npaths, length(scheme.levels)))

function simulate(model, scheme::MultilevelScheme, t0, s0, nsteps, npaths)
  nlevels = length(scheme.levels)
  ml_schemes = schemes(scheme)
  ml_steps = scheme.substeps.^scheme.levels
  x1, t1 = simulate(model, ml_schemes[1], t0, s0, nsteps*ml_steps[1], npaths[1])
  x = Array{typeof(x1)}(2*nlevels-1)
  t = Array{typeof(t1)}(2*nlevels-1)
  x[1] = x1
  t[1] = t1
  for i in 2:nlevels
    x[2i-2], x[2i-1], t[2i-2], t[2i-1] = _multilevel_simulate(model, ml_schemes[i-1], ml_schemes[i], scheme.substeps, t0, s0, nsteps*ml_steps[i-1], npaths[i])
  end
  x, t
end

function sample(model, scheme::MultilevelScheme, t0, s0, nsteps, npaths)
  nlevels = length(scheme.levels)
  ml_schemes = schemes(scheme)
  ml_steps = scheme.substeps.^scheme.levels
  x1 = sample(model, ml_schemes[1], t0, s0, nsteps*ml_steps[1], npaths[1])
  x = Array{typeof(x1)}(2*nlevels-1)
  x[1] = x1
  for i in 2:nlevels
    x[2i-2], x[2i-1] = _multilevel_sample(model, ml_schemes[i-1], ml_schemes[i], scheme.substeps, t0, s0, nsteps*ml_steps[i-1], npaths[i])
  end
  x
end

function _multilevel_sample{T}(model, coarse_scheme, fine_scheme, substeps, t0, s0::T, nsteps, npaths)
  xc = Array{T}(npaths)
  xf = Array{T}(npaths)
  for i in 1:npaths
    xc[i], xf[i] = _multilevel_sample(model, coarse_scheme, fine_scheme, substeps, t0, s0, nsteps)
  end
  xc, xf
end

function _multilevel_sample{D,T,S}(model, coarse_scheme, fine_scheme, substeps, t0, s0::SDEState{D,T,S}, nsteps)
  xc = s0
  xf = s0
  σ = sqrt(fine_scheme.Δt)
  for n in 1:nsteps
    Δw = zero(S)
    tc = t0 + (n - 1) * coarse_scheme.Δt
    for i in 1:substeps
      tf = tc + (i - 1) * fine_scheme.Δt
      Δw += δw = σ * _randn(S)
      xf = step(model, fine_scheme, tf, xf, δw)
    end
    xc = step(model, coarse_scheme, tc, xc, Δw)
  end
  xc, xf
end

function _multilevel_simulate{T,N}(model, coarse_scheme, fine_scheme, substeps, t0, s0::T, nsteps, npaths::Vararg{Integer,N})
  tf = t0 + fine_scheme.Δt * (0:nsteps*substeps)
  tc = tf[1:substeps:end]
  xf = Array{T}(nsteps*substeps+1, npaths...)
  wiener!(xf, fine_scheme.Δt)
  xc = xf[1:substeps:end,:]
  simulate!(xf, model, fine_scheme, t0, s0, xf)
  simulate!(xc, model, coarse_scheme, t0, s0, xc)
  xc, xf, tc, tf
end
