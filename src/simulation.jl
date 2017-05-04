sample(model, scheme, state0::TimeDependentState) =
  TimeDependentState(_sample(model, scheme, state0), statetime(state0) + scheme.Δt)

sample(model, scheme, state0::TimeHomogeneousState) =
  TimeHomogeneousState(_sample(model, scheme, state0))

_sample(model, scheme, state0) = _step(model, scheme, state0, wiener(model, scheme))

function sample(model, scheme, state0, nsteps)
  prevstate = state0
  for i in 1:nsteps
    prevstate = sample(model, scheme, prevstate)
  end
  prevstate
end

sample{T}(model, scheme, state0::T, nsteps, npaths) =
  sample!(Array(T, npaths), model, scheme, state0, nsteps)

function sample!{T}(x::AbstractArray{T,1}, model, scheme, state0, nsteps)
  for i in 1:length(x)
    x[i] = sample(model, scheme, state0, nsteps)
  end
  x
end



simulate{T}(model, scheme, state0::T, nsteps; includestart=false) =
  simulate!(Array(T, nsteps + Int64(includestart)), model, scheme, state0; includestart=includestart)

simulate{T}(model, scheme, state0::T, nsteps, npaths; includestart=false) =
  simulate!(Array(T, nsteps + Int64(includestart), npaths), model, scheme, state0; includestart=includestart)

function simulate!{T}(x::AbstractArray{T,1}, model::AbstractSDE, scheme, state0; includestart=false)
  x[1] = prevstate = state0
  start = includestart ? 2 : 1
  for i in start:length(x)
    x[i] = prevstate = sample(model, scheme, prevstate)
  end
  x
end

function simulate!{T}(x::AbstractArray{T,2}, model, scheme, state0; includestart=false)
  for i in 1:size(x, 2)
    simulate!(view(x, :, i), model, scheme, state0; includestart=includestart)
  end
  x
end



function subsample(model::AbstractSDE, scheme, state0, nsubsteps)
  subscheme = subdivide(scheme, nsubsteps)
  prevstate = state0
  for i in 1:nsubsteps
    prevstate = sample(model, subscheme, prevstate)
  end
  prevstate
end
