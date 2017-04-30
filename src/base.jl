abstract AbstractSDE{D,M}

dim{D,M}(::AbstractSDE{D,M}) = (D,M)
dim{D,M}(::Type{AbstractSDE{D,M}}) = (D,M)
dim{T<:AbstractSDE}(::Type{T}) = dim(supertype(T))

model_dim{D,M}(::AbstractSDE{D,M}) = D
noise_dim{D,M}(::AbstractSDE{D,M}) = M

drift(::AbstractSDE, x) = nothing
diffusion(::AbstractSDE, x) = nothing
