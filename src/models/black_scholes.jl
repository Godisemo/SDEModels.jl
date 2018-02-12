@sde_model BlackScholes dS = r*S*dt + σ*S*dW

function _bs_transition_distr(m, s, x0)
  @unpack r, σ = m
  m = log(x0) + (r-0.5*σ^2)*s.Δt
  d = sqrt(s.Δt)*σ
  LogNormal(m, d)
end

function sample(model::BlackScholes, scheme::Exact, t0, x0)
  d = _bs_transition_distr(model, scheme, x0)
  rand(d)
end

function transition(model::BlackScholes, scheme::Exact, t0, x0, x1)
  d = _bs_transition_distr(model, scheme, x0)
  pdf(d, x1)
end

function logtransition(model::BlackScholes, scheme::Exact, t0, x0, x1)
  d = _bs_transition_distr(model, scheme, x0)
  logpdf(d, x1)
end
