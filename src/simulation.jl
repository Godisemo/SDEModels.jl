sample(model, scheme, t, state0::TimeDependentState) =
  TimeDependentState(_sample(model, scheme, t, state0), statetime(state0) + scheme.Δt)

sample(model, scheme, t, state0::TimeHomogeneousState) =
  TimeHomogeneousState(_sample(model, scheme, t, state0))

_sample(model, scheme, t, state0) = _step(model, scheme, t, state0, wiener(model, scheme))

function sample(model, scheme, t, state0, nsteps)
  prevstate = state0
  for i in 1:nsteps
    ti = t + (i - 1) * scheme.Δt
    prevstate = sample(model, scheme, ti, prevstate)
  end
  prevstate
end

sample{T}(model, scheme, t, state0::T, nsteps, npaths) =
  sample!(Array(T, npaths), model, scheme, t, state0, nsteps)

function sample!{T}(x::AbstractArray{T,1}, model, scheme, t, state0, nsteps)
  for i in 1:length(x)
    x[i] = sample(model, scheme, t, state0, nsteps)
  end
  x
end



simulate{T}(model, scheme, t, state0::T, nsteps; includestart=false) =
  simulate!(Array(T, nsteps + Int64(includestart)), model, scheme, t, state0; includestart=includestart)

simulate{T}(model, scheme, t, state0::T, nsteps, npaths; includestart=false) =
  simulate!(Array(T, nsteps + Int64(includestart), npaths), model, scheme, t, state0; includestart=includestart)

function simulate!{T}(x::AbstractArray{T,1}, model::AbstractSDE, scheme, t, state0; includestart=false)
  x[1] = prevstate = state0
  start = includestart ? 2 : 1
  for i in start:length(x)
    ti = t + (i - 1) * scheme.Δt
    x[i] = prevstate = sample(model, scheme, ti, prevstate)
  end
  x
end

function simulate!{T}(x::AbstractArray{T,2}, model, scheme, t, state0; includestart=false)
  for i in 1:size(x, 2)
    simulate!(view(x, :, i), model, scheme, t, state0; includestart=includestart)
  end
  x
end



function subsample(model::AbstractSDE, scheme, t, state0, nsubsteps)
  subscheme = subdivide(scheme, nsubsteps)
  prevstate = state0
  for i in 1:nsubsteps
    ti = t + (i - 1) * subscheme.Δt
    prevstate = sample(model, subscheme, ti, prevstate)
  end
  prevstate
end
