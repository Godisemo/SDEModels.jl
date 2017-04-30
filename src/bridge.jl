immutable ModifiedBridge
  model::AbstractSDE
  x1::Union{Array{Float64},Float64}
  t1::Float64
end

model_dim(bridge::ModifiedBridge) = model_dim(bridge.model)

function sample(bridge::ModifiedBridge, scheme::EulerMaruyama, x0)
  xprev = x0
  tprev = 0.0
    μ = (bridge.x1 - xprev) / bridge.t1
    σ = sqrt((bridge.t1 - scheme.Δt) / bridge.t1) * diffusion(bridge.model, xprev)
    xprev + μ * scheme.Δt + σ * wiener(bridge.model, scheme)
end

function simulate!{T}(x::AbstractArray{T,2}, bridge::ModifiedBridge, scheme::EulerMaruyama, x0)
  xprev = x0
  tprev = 0.0
  for i in 1:size(x, 2)
    μ = (bridge.x1 - xprev) / (bridge.t1 - tprev)
    σ = sqrt((bridge.t1 - (tprev + scheme.Δt)) / (bridge.t1 - tprev)) * diffusion(bridge.model, xprev)
    x[:,i] = xprev = xprev + μ * scheme.Δt + σ * wiener(bridge.model, scheme)
    tprev += scheme.Δt
  end
  x
end
