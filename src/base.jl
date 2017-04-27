import Base.show

abstract AbstractSDE{D,M}

dim{D,M}(::AbstractSDE{D,M}) = (D,M)
dim{D,M}(::Type{AbstractSDE{D,M}}) = (D,M)
dim{T<:AbstractSDE}(::Type{T}) = dim(supertype(T))

drift(::AbstractSDE, x) = nothing
diffusion(::AbstractSDE, x) = nothing

function Base.show(io::IO, model::AbstractSDE)
  name = typeof(model)
  n = nfields(model)
  parameters = fieldnames(model)
  w = maximum(length.(string.(fieldnames(model))))
  println("$name with $n parameters:")
  for p in parameters
    println("  $(rpad(p, w)) => $(getfield(model, p))")
  end
end
