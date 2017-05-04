function _euler_transition_params(model, scheme, s0, s1)
    x0 = statevalue(s0)
    x1 = statevalue(s1)
    μ = x0 + drift(model, s0) * scheme.Δt
    σ = diffusion(model, s0)
    Σ = scheme.Δt * σ * σ'
    z = x1 - μ
    z, Σ
end

function transition{D}(model::AbstractSDE{D}, scheme::EulerMaruyama, s0::SDEState{D}, s1::SDEState{D})
  z, Σ = _euler_transition_params(model, scheme, s0, s1)
  1.0 / sqrt(det(2pi*Σ)) * exp(-0.5*dot(z, Σ\z))
end

function logtransition{D}(model::AbstractSDE{D}, scheme::EulerMaruyama, s0::SDEState{D}, s1::SDEState{D})
  z, Σ = _euler_transition_params(model, scheme, s0, s1)
  -0.5*(D*log(2pi) + log(det(Σ)) + dot(z, Σ\z))
end
