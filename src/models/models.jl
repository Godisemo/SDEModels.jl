import Distributions: pdf, logpdf, NoncentralChisq, LogNormal, Normal

dim(::AbstractSDE{D,M}) where {D,M} = (D,M)
dim(::Type{AbstractSDE{D,M}}) where {D,M} = (D,M)
dim(::Type{T}) where {T<:AbstractSDE} = dim(supertype(T))

model_dim(::AbstractSDE{D,M}) where {D,M} = D
noise_dim(::AbstractSDE{D,M}) where {D,M} = M

function drift end
# function drift_jacobian end
function diffusion end
function jump end
function mark_distribution end
function variables end

_reshape_or_nothing(x::Number, dims) = x
_reshape_or_nothing(x::AbstractArray, dims) = reshape(x, dims)

function corrected_drift(model::AbstractSDE{D,M}, t0, x0, α) where {D,M}
  μ = drift(model, t0, x0)
  σ = diffusion(model, t0, x0)
  corr = @MArray zeros(D)
  ∂σ = _reshape_or_nothing(_diffusion_jacobian(model, t0, x0), (D, M, D))
  # μ - α*vec(sum(permutedims(∂σ, (3,2,1)).*σ, dims=(1,2)))
  for j=1:M, i=1:D, k=1:D
    corr[i] += σ[k,j] * ∂σ[i,j,k]
  end
  μ - α*corr
end

include("codegen.jl")

include("black_scholes.jl")
include("cox_ingersoll_ross.jl")
include("merton.jl")
include("ornstein_uhlenbeck.jl")

@sde_model FitzHughNagumo begin
  dX = ɛ*(X-X^3-Y+s)*dt + σ1*dW1
  dY =     (γ*X-Y+β)*dt + σ2*dW2
end

@sde_model Heston begin
  dS =     r*S*dt + sqrt(V)*S*dW1
  dV = κ*(θ-V)*dt + σ*sqrt(V)*dW2
  dW1*dW2 = ρ*dt
end

@sde_model Bates begin
  dS =     α*S*dt + sqrt(V)*S*dW1 + S*(ξ-1.0)*dN
  dV = κ*(θ-V)*dt + σ*sqrt(V)*dW2
  dW1*dW2 = ρ*dt
  ξ ~ LogNormal(μ, δ)
end α κ θ σ ρ λ μ δ

@sde_model CompoundPoisson begin
  dS = S*(ξ-1.0)*dN
  ξ ~ LogNormal(μ, δ)
end λ μ δ
