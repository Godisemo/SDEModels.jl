
# specialized versions of `randn(S)` which give better performance
# @inline _randn(::S) where S = randn(S)
@inline _randn(::Type{Float64}) = randn()
@generated function _randn(::Type{SVector{D,Float64}}) where D
   quote
     $(Expr(:meta, :inline))
     $(Expr(:call, SVector, Iterators.repeated(:(randn()), D)...))
   end
 end

## wiener process simulation ##

function wiener!(w::AbstractVecOrMat{T}, Δt) where T
  σ = sqrt(Δt)
  nsteps = size(w, 1)
  npaths = size(w, 2)
  for k in 1:npaths
    sum = zero(T)
    @inbounds w[1,k] = sum
    for i in 2:nsteps
      sum += σ * _randn(T)
      @inbounds w[i,k] = sum
    end
  end
  w
end

## simulate and discard everything except for the endpoint ##

@inline function sample(model, scheme, t0, x0::T, nsteps) where T
  σ = sqrt(scheme.Δt)
  x = x0
  for i in 1:nsteps
    ti = t0 + (i - 1) * scheme.Δt
    Δw = σ * _randn(T)
    x = step(model, scheme, ti, x, Δw)
  end
  x
end

@inline function sample(model, scheme::Exact, t0, x0, nsteps)
  x = x0
  for i in 1:nsteps
    ti = t0 + (i - 1) * scheme.Δt
    x = sample(model, scheme, ti, x)
  end
  x
end

@inline sample(model, scheme, t0, x0::T, nsteps, npaths) where T =
  sample!(Array{T}(npaths), model, scheme, t0, x0, nsteps)

@inline function sample!(x, model, scheme, t0, x0, nsteps)
  for k in 1:length(x)
    x[k] = sample(model, scheme, t0, x0, nsteps)
  end
  x
end

## simulate and save the entire path, start point included ##

function simulate(model, scheme, t0, x0::T, nsteps, npaths::Vararg{Integer,N}) where {T,N}
  t = t0 + scheme.Δt * (0:nsteps)
  x = Array{T}(nsteps+1, npaths...)
  simulate!(x, model, scheme, t0, x0)
  x, t
end

@inline function simulate!(x, model, scheme::Exact, t0, x0)
  nsteps = size(x, 1)
  npaths = size(x, 2)
  for k in 1:npaths
    @inbounds x[1,k] = x0
    for i in 2:nsteps
      t = t0 + (i-2) * scheme.Δt
      @inbounds x[i,k] = sample(model, scheme, t, x[i-1,k])
    end
  end
  x
end

@inline function simulate!(x, model, scheme, t0, x0::T) where T
  nsteps = size(x, 1)
  npaths = size(x, 2)
  σ = sqrt(scheme.Δt)
  for k in 1:npaths
    @inbounds x[1,k] = x0
    for i in 2:nsteps
      t = t0 + (i-2) * scheme.Δt
      Δw = σ * _randn(T)
      @inbounds x[i,k] = step(model, scheme, t, x[i-1,k], Δw)
    end
  end
  x
end

@inline function simulate!(x, model, scheme, t0, x0::T, w) where T
  # note that x === w is allowed
  nsteps = size(x, 1)
  npaths = size(x, 2)
  for k in 1:npaths
    @inbounds wprev = w[1,k]
    @inbounds x[1,k] = x0
    for i in 2:nsteps
      t = t0 + (i-2) * scheme.Δt
      @inbounds wnext = w[i,k]
      @inbounds x[i,k] = step(model, scheme, t, x[i-1,k], wnext - wprev)
      wprev = wnext
    end
  end
  x
end

## wrapper methods to make sure StaticArrays are used internaly ##

function sample(model, scheme::AbstractScheme, t0, x0::Array{Float64,1}, nsteps)
  x = sample(model, scheme, t0, SVector(x0...), nsteps)
  data = Array{Float64}(x)
end

function sample(model, scheme::AbstractScheme, t0, x0::Array{Float64,1}, nsteps, npaths::Vararg{Integer,N}) where N
  D = length(x0)
  x = sample(model, scheme, t0, SVector{D}(x0), nsteps, npaths...)
  data = reinterpret(Float64, x, (D, size(x)...))
end

function simulate(model, scheme::AbstractScheme, t0, x0::Array{Float64,1}, nsteps, npaths::Vararg{Integer,N}) where N
  D = length(x0)
  x, t = simulate(model, scheme, t0, SVector{D}(x0), nsteps, npaths...)
  data = reinterpret(Float64, x, (D, size(x)...))
  data, t
end
