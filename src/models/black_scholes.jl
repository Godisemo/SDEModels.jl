@sde_model BlackScholes dS = r*S*dt + σ*S*dW

function _bs_transition_distr(m, s, s0)
  @unpack r, σ = m
  x0 = statevalue(s0)
  m = log(x0) + (r-0.5*σ^2)*s.Δt
  d = sqrt(s.Δt)*σ
  LogNormal(m, d)
end

function _sample(model::BlackScholes, scheme::Exact, s0::AbstractState{1})
  d = _bs_transition_distr(model, scheme, s0)
  rand(d)
end

function transition(model::BlackScholes, scheme::Exact, s0::AbstractState{1}, s1::AbstractState{1})
  d = _bs_transition_distr(model, scheme, s0)
  pdf(d, statevalue(s1))
end

function logtransition(model::BlackScholes, scheme::Exact, s0::AbstractState{1}, s1::AbstractState{1})
  d = _bs_transition_distr(model, scheme, s0)
  logpdf(d, statevalue(s1))
end
