sample(model, scheme, state0) = step(model, scheme, state0, wiener(model, scheme))

function sample(model, scheme, state0, nsteps; args...)
  prevstate = state0
  for i in 1:nsteps
    prevstate = sample(model, scheme, prevstate; args...)
  end
  prevstate
end

sample(model, scheme, state0, nsteps, npaths; args...) =
  sample!(Array(eltype(state0), npaths), model, scheme, state0, nsteps; args...)

function sample!{T}(x::AbstractArray{T,1}, model, scheme, state0, nsteps; args...)
  for i in 1:length(x)
    x[i] = sample(model, scheme, state0, nsteps; args...)
  end
  x
end



simulate(model, scheme, state0, nsteps; includestart=false, args...) =
  simulate!(Array(eltype(state0), nsteps + Int64(includestart)), model, scheme, state0; includestart=includestart, args...)

simulate(model, scheme, state0, nsteps, npaths; includestart=false, args...) =
  simulate!(Array(eltype(state0), nsteps + Int64(includestart), npaths), model, scheme, state0; includestart=includestart, args...)

function simulate!{T}(x::AbstractArray{T,1}, model::AbstractSDE, scheme, state0; includestart=false, args...)
  x[1] = prevstate = state0
  start = includestart ? 2 : 1
  for i in start:length(x)
    x[i] = prevstate = sample(model, scheme, prevstate; args...)
  end
  x
end

function simulate!{T}(x::AbstractArray{T,2}, model, scheme, state0; args...)
  for i in 1:size(x, 2)
    simulate!(view(x, :, i), model, scheme, state0; args...)
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
