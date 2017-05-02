function zeromeanvar{D}(model::AbstractSDE{D}, s0::TimeDependentState{D}, s1::TimeDependentState{D})
    x0 = statevalue(s0)
    x1 = statevalue(s1)
    t0 = statetime(s0)
    t1 = statetime(s1)
    Δt = t1 - t0
    μ = x0 + drift(model, s0) * Δt
    σ = diffusion(model, s0)
    Σ = Δt * σ * σ'
    z = x1 - μ
    z, Σ
end

function pdf{D}(model::AbstractSDE{D}, s0::TimeDependentState{D}, s1::TimeDependentState{D})
  z, Σ = zeromeanvar(model, s0, s1)
  1.0 / sqrt(det(2pi*Σ)) * exp(-0.5*dot(z, Σ\z))
end

function logpdf{D}(model::AbstractSDE{D}, s0::TimeDependentState{D}, s1::TimeDependentState{D})
  z, Σ = zeromeanvar(model, s0, s1)
  -0.5*(D*log(2pi) + log(det(Σ)) + dot(z, Σ\z))
end
