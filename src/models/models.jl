import Distributions: pdf, logpdf, NoncentralChisq, LogNormal

include("black_scholes.jl")
include("cox_ingersoll_ross.jl")

@sde_model OrnsteinUhlenbeck dx = θ*(μ-x)*dt + σ*dW

@sde_model FitzHughNagumo begin
  dX = ɛ*(X-X^3-Y+s)*dt + σ1*dW1
  dY =     (γ*X-Y+β)*dt + σ2*dW2
end

@sde_model Heston begin
  dS =     r*S*dt + √V*S*dW1
  dV = κ*(θ-V)*dt + σ*√V*(ρ*dW1 + √(1-ρ^2)*dW2)
end
