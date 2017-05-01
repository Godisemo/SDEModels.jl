# TODO option to include startpoint
# TODO move npaths and nsteps to key-value args


sample(model, scheme, state0) = step(model, scheme, state0, wiener(model, scheme))

sample(model, scheme, state0, npaths; args...) =
  sample!(Array(eltype(state0), npaths), model, scheme, state0; args...)

function sample!(x, model, scheme, state0; args...)
  for i in 1:length(x)
    x[i] = sample(model, scheme, state0; args...)
  end
  x
end



simulate(model, scheme, state0, nsteps; args...) =
  simulate!(Array(eltype(state0), nsteps), model, scheme, state0; args...)

simulate(model, scheme, state0, nsteps, npaths; args...) =
  simulate!(Array(eltype(state0), nsteps, npaths), model, scheme, state0; args...)

function simulate!{T}(x::AbstractArray{T,1}, model::AbstractSDE, scheme, state0; args...)
  prevstate = state0
  for i in 1:length(x)
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
