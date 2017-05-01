
sample(model, scheme, x0) = step(model, scheme, x0, wiener(model, scheme))

sample(model, scheme, x0, npaths) =
  sample!(Array(eltype(x0), model_dim(model), npaths), model, scheme, x0)

function sample!(x, model, scheme, x0)
  for i in 1:size(x, 2)
    x[:,i] = sample(model, scheme, x0)
  end
  x
end



simulate(model, scheme, x0, nsteps) =
  simulate!(Array(eltype(x0), model_dim(model), nsteps), model, scheme, x0)

simulate(model, scheme, x0, nsteps, npaths) =
  simulate!(Array(eltype(x0), model_dim(model), nsteps, npaths), model, scheme, x0)

function simulate!{T}(x::AbstractArray{T,2}, model::AbstractSDE, scheme, x0)
  xprev = x0
  for i in 1:size(x, 2)
    x[:,i] = xprev = sample(model, scheme, xprev)
  end
  x
end

function simulate!{T}(x::AbstractArray{T,3}, model, scheme, x0)
  for i in 1:size(x, 3)
    simulate!(view(x, :, :,i), model, scheme, x0)
  end
  x
end



function subsample(model::AbstractSDE, scheme, x0, nsubsteps)
  subscheme = subdivide(scheme, nsubsteps)
  xprev = x0
  for i in 1:nsubsteps
    xprev = sample(model, subscheme, xprev)
  end
  xprev
end
