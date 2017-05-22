immutable Multilevel{M,T<:AbstractScheme}
  coarse::T
  fine::T
  (::Type{Multilevel}){T}(M, scheme::T) = new{M,T}(scheme, subdivide(scheme, M))
end

function sample{M,D,T,S}(model, scheme::Multilevel{M}, t0, s0::SDEState{D,T,S}, nsteps)
  xc = s0
  xf = s0
  σ = sqrt(scheme.fine.Δt)
  for n in 1:nsteps
    Δw = zero(S)
    tc = t0 + (n - 1) * scheme.coarse.Δt
    for i in 1:M
      tf = tc + (i - 1) * scheme.fine.Δt
      Δw += δw = σ * _randn(S)
      xf = step(model, scheme.fine, tf, xf, δw)
    end
    xc = step(model, scheme.coarse, tc, xc, Δw)
  end
  xc, xf
end

function sample{T}(model, scheme::Multilevel, t0, s0::T, nsteps, npaths)
  xc = Array{T}(npaths)
  xf = Array{T}(npaths)
  for i in 1:npaths
    xc[i], xf[i] = sample(model, scheme, t0, s0, nsteps)
  end
  xc, xf
end

function simulate{M,T,N}(model, scheme::Multilevel{M}, t0, s0::T, nsteps, npaths::Vararg{Integer,N})
  tf = t0 + scheme.fine.Δt * (0:nsteps*M)
  tc = tf[1:M:end]
  xf = Array{T}(nsteps*M+1, npaths...)
  wiener!(xf, scheme.fine.Δt)
  xc = xf[1:M:end,:]
  simulate!(xf, model, scheme.fine, t0, s0, xf)
  simulate!(xc, model, scheme.coarse, t0, s0, xc)
  xc, xf, tc, tf
end
