
@sde_model BlackScholes dS = r*S*dt + σ*S*dW

@sde_model CoxIngersollRoss dr = κ*(θ-r)*dt + σ*√r*dW

@sde_model FitzHughNagumo begin
  dX = ɛ*(X-X^3-Y+s)*dt + σ1*dW1
  dY =     (γ*X-Y+β)*dt + σ2*dW2
end

@sde_model Heston begin
  dS =     r*S*dt + √V*S*dW1
  dV = κ*(θ-V)*dt + σ*√V*(ρ*dW1 + √(1-ρ^2)*dW2)
end