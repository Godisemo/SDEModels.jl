import Distributions: pdf, logpdf, NoncentralChisq, LogNormal, Normal

dim{D,M}(::AbstractSDE{D,M}) = (D,M)
dim{D,M}(::Type{AbstractSDE{D,M}}) = (D,M)
dim{T<:AbstractSDE}(::Type{T}) = dim(supertype(T))

model_dim{D,M}(::AbstractSDE{D,M}) = D
noise_dim{D,M}(::AbstractSDE{D,M}) = M

drift(::AbstractSDE, t, x) = error("drift is not implemented for this model")
drift_jacobian(::AbstractSDE, t, x) = error("drift jacobian is not implemented for this model")
diffusion(::AbstractSDE, t, x) = error("diffusion is not implemented for this model")
variables(::AbstractSDE) = error("variables is not implemented for this model")

include("codegen.jl")

include("black_scholes.jl")
include("cox_ingersoll_ross.jl")
include("ornstein_uhlenbeck.jl")

@sde_model FitzHughNagumo begin
  dX = ɛ*(X-X^3-Y+s)*dt + σ1*dW1
  dY =     (γ*X-Y+β)*dt + σ2*dW2
end

@sde_model Heston begin
  dS =     r*S*dt + √V*S*dW1
  dV = κ*(θ-V)*dt + σ*√V*(ρ*dW1 + √(1-ρ^2)*dW2)
end
