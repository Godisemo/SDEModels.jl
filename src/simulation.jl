
# specialized versions of `randn(S)` which give better performance
@inline _randn(S) = randn(S)
@inline _randn(::Type{Float64}) = randn()
@generated function _randn{S<:StaticArray}(::Type{S})
  quote
    $(Expr(:meta, :inline))
    $(Expr(:call, S, repeated(:(randn()), length(S))...))
  end
end

## wiener process simulation ##

function wiener!{D,T,S}(w::AbstractVecOrMat{SDEState{D,T,S}}, Δt)
  σ = sqrt(Δt)
  nsteps = size(w, 1)
  npaths = size(w, 2)
  for k in 1:npaths
    sum = zero(S)
    @inbounds w[1,k] = SDEState(sum)
    for i in 2:nsteps
      sum += σ * _randn(S)
      @inbounds w[i,k] = SDEState(sum)
    end
  end
  w
end

## simulate and discard everything except for the endpoint ##

@inline function sample{D,T,S}(model, scheme, t0, s0::SDEState{D,T,S}, nsteps)
  σ = sqrt(scheme.Δt)
  x = s0
  for i in 1:nsteps
    ti = t0 + (i - 1) * scheme.Δt
    Δw = σ * _randn(S)
    x = step(model, scheme, ti, x, Δw)
  end
  x
end

@inline sample{T}(model, scheme, t0, s0::T, nsteps, npaths) =
  sample!(Array{T}(npaths), model, scheme, t0, s0, nsteps)

@inline function sample!(x, model, scheme, t0, s0, nsteps)
  for k in 1:length(x)
    x[k] = sample(model, scheme, t0, s0, nsteps)
  end
  x
end

## simulate and save the entire path, start point included ##

function simulate{T,N}(model, scheme, t0, s0::T, nsteps, npaths::Vararg{Integer,N})
  t = t0 + scheme.Δt * (0:nsteps)
  x = Array{T}(nsteps+1, npaths...)
  simulate!(x, model, scheme, t0, s0)
  x, t
end

@inline function simulate!{D,T,S}(x, model, scheme, t0, s0::SDEState{D,T,S})
  nsteps = size(x, 1)
  npaths = size(x, 2)
  σ = sqrt(scheme.Δt)
  for k in 1:npaths
    @inbounds x[1,k] = s0
    for i in 2:nsteps
      t = t0 + (i-2) * scheme.Δt
      Δw = σ * _randn(S)
      @inbounds x[i,k] = step(model, scheme, t, x[i-1,k], Δw)
    end
  end
  x
end

@inline function simulate!{T}(x, model, scheme, t0, s0::T, w)
  # note that x === w is allowed
  nsteps = size(x, 1)
  npaths = size(x, 2)
  for k in 1:npaths
    @inbounds wprev = statevalue(w[1,k])
    @inbounds x[1,k] = s0
    for i in 2:nsteps
      t = t0 + (i-2) * scheme.Δt
      @inbounds wnext = statevalue(w[i,k])
      @inbounds x[i,k] = step(model, scheme, t, x[i-1,k], wnext - wprev)
      wprev = wnext
    end
  end
  x
end
