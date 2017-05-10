@sde_model OrnsteinUhlenbeck dx = θ*(μ-x)*dt + σ*dW

function _ou_transition_distr(m, s, s0)
  @unpack θ, μ, σ = m
  x0 = statevalue(s0)
  m = x0 * exp(-θ * s.Δt) + μ * (1.0 - exp(-θ * s.Δt))
  d = sqrt(1.0 - exp(-2θ * s.Δt)) * σ / sqrt(2θ)
  Normal(m, d)
end

function sample(model::OrnsteinUhlenbeck, scheme::Exact, t0, s0::SDEState{1})
  d = _ou_transition_distr(model, scheme, s0)
  SDEState(rand(d))
end

function transition(model::OrnsteinUhlenbeck, scheme::Exact, t0, s0::SDEState{1}, s1::SDEState{1})
  d = _ou_transition_distr(model, scheme, s0)
  pdf(d, statevalue(s1))
end

function logtransition(model::OrnsteinUhlenbeck, scheme::Exact, t0, s0::SDEState{1}, s1::SDEState{1})
  d = _ou_transition_distr(model, scheme, s0)
  logpdf(d, statevalue(s1))
end
