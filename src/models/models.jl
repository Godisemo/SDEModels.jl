import Distributions: pdf, logpdf, NoncentralChisq, LogNormal, Normal

dim{D,M}(::AbstractSDE{D,M}) = (D,M)
dim{D,M}(::Type{AbstractSDE{D,M}}) = (D,M)
dim{T<:AbstractSDE}(::Type{T}) = dim(supertype(T))

model_dim{D,M}(::AbstractSDE{D,M}) = D
noise_dim{D,M}(::AbstractSDE{D,M}) = M

function drift end
function corrected_drift end
function drift_jacobian end
function diffusion end
function variables end

include("codegen.jl")

include("black_scholes.jl")
include("cox_ingersoll_ross.jl")
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
