sample(model, scheme, t0, state0::SDEState) =
  step(model, scheme, t0, state0, wiener(model, scheme))

function sample(model, scheme, t0, state0, nsteps)
  prevstate = state0
  for i in 1:nsteps
    ti = t0 + (i - 1) * scheme.Δt
    prevstate = sample(model, scheme, ti, prevstate)
  end
  prevstate
end

sample{T}(model, scheme, t0, state0::T, nsteps, npaths) =
  sample!(Array(T, npaths), model, scheme, t0, state0, nsteps)

function sample!{T}(x::AbstractArray{T,1}, model, scheme, t0, state0, nsteps)
  for i in 1:length(x)
    x[i] = sample(model, scheme, t0, state0, nsteps)
  end
  x
end



function simulate{T}(model, scheme, t0, state0::T, nsteps; includestart=false)
  t = t0 + scheme.Δt * (Int64(!includestart):nsteps)
  x = Array(T, nsteps + Int64(includestart))
  simulate!(x, model, scheme, t0, state0; includestart=includestart)
  x, t
end

function simulate{T}(model, scheme, t0, state0::T, nsteps, npaths; includestart=false)
  t = t0 + scheme.Δt * (Int64(!includestart):nsteps)
  x = Array(T, nsteps + Int64(includestart), npaths)
  simulate!(x, model, scheme, t0, state0; includestart=includestart)
  x, t
end

function simulate!{T}(x::AbstractArray{T,1}, model::AbstractSDE, scheme, t0, state0; includestart=false)
  x[1] = prevstate = state0
  start = includestart ? 2 : 1
  for i in start:length(x)
    ti = t0 + (i - 1) * scheme.Δt
    x[i] = prevstate = sample(model, scheme, ti, prevstate)
  end
  x
end

function simulate!{T}(x::AbstractArray{T,2}, model, scheme, t0, state0; includestart=false)
  start = includestart ? 2 : 1
  for i in 1:size(x, 2)
    x[1,i] = prevstate = state0
    for j in start:size(x, 1)
      tj = t0 + (j - 1) * scheme.Δt
      x[j,i] = prevstate = sample(model, scheme, tj, prevstate)
    end
  end
  x
end



function subsample(model::AbstractSDE, scheme, t0, state0, nsubsteps)
  subscheme = subdivide(scheme, nsubsteps)
  prevstate = state0
  for i in 1:nsubsteps
    ti = t0 + (i - 1) * subscheme.Δt
    prevstate = sample(model, subscheme, ti, prevstate)
  end
  prevstate
end
