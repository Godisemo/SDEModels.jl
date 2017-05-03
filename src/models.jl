import Distributions: pdf, logpdf, NoncentralChisq

@sde_model BlackScholes dS = r*S*dt + σ*S*dW

@sde_model OrnsteinUhlenbeck dx = θ*(μ-x)*dt + σ*dW

@sde_model CoxIngersollRoss dr = κ*(θ-r)*dt + σ*√r*dW

function _cir_transition_params(m, s, s0, s1)
  x0 = statevalue(s0)
  x1 = statevalue(s1)
  c = 2 * m.κ / (m.σ^2 * (1 - exp(-m.κ * s.Δt)))
  df = 4 * m.κ * m.θ / m.σ^2
  nc = x0 * 2c * exp(-m.κ * s.Δt)
  c, df, nc
end

function transition(model::CoxIngersollRoss, scheme::Exact, s0::AbstractState{1}, s1::AbstractState{1})
  c, df, nc = _cir_transition_params(model, scheme, s0, s1)
  2c * pdf(NoncentralChisq(df, nc), statevalue(s1) * 2c)
end

function logtransition(model::CoxIngersollRoss, scheme::Exact, s0::AbstractState{1}, s1::AbstractState{1})
  c, df, nc = _cir_transition_params(model, scheme, s0, s1)
  logpdf(NoncentralChisq(df, nc), statevalue(s1) * 2c) + log(2c)
end

@sde_model FitzHughNagumo begin
  dX = ɛ*(X-X^3-Y+s)*dt + σ1*dW1
  dY =     (γ*X-Y+β)*dt + σ2*dW2
end

@sde_model Heston begin
  dS =     r*S*dt + √V*S*dW1
  dV = κ*(θ-V)*dt + σ*√V*(ρ*dW1 + √(1-ρ^2)*dW2)
end
