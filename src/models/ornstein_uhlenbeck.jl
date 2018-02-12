@sde_model OrnsteinUhlenbeck dx = θ*(μ-x)*dt + σ*dW

function _ou_transition_distr(m, s, x0)
  @unpack θ, μ, σ = m
  m = x0 * exp(-θ * s.Δt) + μ * (1.0 - exp(-θ * s.Δt))
  d = sqrt(1.0 - exp(-2θ * s.Δt)) * σ / sqrt(2θ)
  Normal(m, d)
end

function sample(model::OrnsteinUhlenbeck, scheme::Exact, t0, x0)
  d = _ou_transition_distr(model, scheme, x0)
  rand(d)
end

function transition(model::OrnsteinUhlenbeck, scheme::Exact, t0, x0, x1)
  d = _ou_transition_distr(model, scheme, x0)
  pdf(d, x1)
end

function logtransition(model::OrnsteinUhlenbeck, scheme::Exact, t0, x0, x1)
  d = _ou_transition_distr(model, scheme, x0)
  logpdf(d, x1)
end
