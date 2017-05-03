import Distributions: pdf, logpdf, NoncentralChisq

@sde_model CoxIngersollRoss dr = κ*(θ-r)*dt + σ*√r*dW

function _cir_transition_params(m, s, s0)
  @unpack κ, θ, σ = m
  x0 = statevalue(s0)
  c = 2 * κ / (σ^2 * (1 - exp(-κ * s.Δt)))
  df = 4 * κ * θ / σ^2
  nc = x0 * 2c * exp(-κ * s.Δt)
  c, df, nc
end

function _sample(model::CoxIngersollRoss, scheme::Exact, s0::AbstractState{1})
  c, df, nc = _cir_transition_params(model, scheme, s0)
  rand(NoncentralChisq(df, nc)) / 2c
end

function transition(model::CoxIngersollRoss, scheme::Exact, s0::AbstractState{1}, s1::AbstractState{1})
  c, df, nc = _cir_transition_params(model, scheme, s0)
  2c * pdf(NoncentralChisq(df, nc), statevalue(s1) * 2c)
end

function logtransition(model::CoxIngersollRoss, scheme::Exact, s0::AbstractState{1}, s1::AbstractState{1})
  c, df, nc = _cir_transition_params(model, scheme, s0)
  logpdf(NoncentralChisq(df, nc), statevalue(s1) * 2c) + log(2c)
end
