# for Q dynamics set α = r-λ*(exp(μ+0.5*δ^2)-1)
@sde_model Merton begin
  dS = S*α*dt + S*σ*dW + S*(ξ-1.0)*dN
  J ~ LogNormal(μ, δ)
end α σ λ μ δ

function sample(model::Merton, scheme::Exact, t0, x0)
  @unpack α, σ, λ, μ, δ = model
  N = rand(Poisson(model.λ*scheme.Δt))
  rand(LogNormal(log(x0)+(α-0.5*σ^2)*scheme.Δt+N*μ, sqrt(N*δ^2+scheme.Δt*σ^2)))
end
