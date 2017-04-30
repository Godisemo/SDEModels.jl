
sample(model, scheme, x0) = step(model, scheme, x0, wiener(model, scheme))

sample{D}(model::AbstractSDE{D}, scheme, x0, npaths) =
  sample!(Array(Float64, D, npaths), model, scheme, x0)

function sample!(x, model, scheme, x0)
  for i in 1:size(x, 2)
    x[:,i] = sample(model, scheme, x0)
  end
  x
end



simulate{D}(model::AbstractSDE{D}, scheme, x0, nsteps) =
  simulate!(Array(Float64, D, nsteps), model, scheme, x0)

simulate{D}(model::AbstractSDE{D}, scheme, x0, nsteps, npaths) =
  simulate!(Array(Float64, D, nsteps, npaths), model, scheme, x0)

function simulate!(x::AbstractArray{TypeVar(:T),2}, model, scheme, x0)
  xprev = x0
  for i in 1:size(x, 2)
    x[:,i] = xprev = sample(model, scheme, xprev)
  end
  x
end

function simulate!(x::AbstractArray{TypeVar(:T),3}, model, scheme, x0)
  for i in 1:size(x, 3)
    simulate!(view(x, :, :,i), model, scheme, x0)
  end
  x
end



function subsample(model, scheme, x0, nsubsteps)
  subscheme = subdivide(scheme, nsubsteps)
  xprev = x0
  for i in 1:nsubsteps
    xprev = sample(model, subscheme, xprev)
  end
  xprev
end
