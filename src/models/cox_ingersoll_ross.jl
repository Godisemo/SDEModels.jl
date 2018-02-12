@sde_model CoxIngersollRoss dr = κ*(θ-r)*dt + σ*sqrt(r)*dW

function _cir_transition_params(m, s, x0)
  @unpack κ, θ, σ = m
  c = 2 * κ / (σ^2 * (1 - exp(-κ * s.Δt)))
  df = 4 * κ * θ / σ^2
  nc = x0 * 2c * exp(-κ * s.Δt)
  c, df, nc
end

function sample(model::CoxIngersollRoss, scheme::Exact, t0, x0)
  c, df, nc = _cir_transition_params(model, scheme, x0)
  rand(NoncentralChisq(df, nc)) / 2c
end

function transition(model::CoxIngersollRoss, scheme::Exact, t0, x0, x1)
  c, df, nc = _cir_transition_params(model, scheme, x0)
  2c * pdf(NoncentralChisq(df, nc), x1 * 2c)
end

function logtransition(model::CoxIngersollRoss, scheme::Exact, t0, x0, x1)
  c, df, nc = _cir_transition_params(model, scheme, x0)
  logpdf(NoncentralChisq(df, nc), x1 * 2c) + log(2c)
end
