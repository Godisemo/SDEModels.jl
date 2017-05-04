immutable MultilevelState{M,T}
  sc::T
  sf::T
end

coarse(state::MultilevelState) = state.sc
fine(state::MultilevelState) = state.sf

coarse(A::AbstractArray) = reshape([ coarse(x) for x in A ], size(A))
fine(A::AbstractArray) = reshape([ fine(x) for x in A ], size(A))

multilevel{T}(M, xc::T, xf::T) = MultilevelState{M,T}(xc, xf)
multilevel{T}(M, x::T) = MultilevelState{M,T}(x, x)

function sample{M,T}(model, scheme, t0, s0::MultilevelState{M,T})
  subscheme = subdivide(scheme, M)
  scprev = coarse(s0)
  sfprev = fine(s0)
  Δw = zero(wiener_type(model))
  for i in 1:M
    ti = t0 + (i - 1) * subscheme.Δt
    Δw += δw = wiener(model, subscheme)
    sfprev = step(model, subscheme, ti, sfprev, δw)
  end
  scprev = step(model, scheme, t0, scprev, Δw)
  MultilevelState{M,T}(scprev, sfprev)
end
