immutable MultilevelState{M,T}
  xc::T
  xf::T
end

coarse(state::MultilevelState) = state.xc
fine(state::MultilevelState) = state.xf

coarse(A::AbstractArray) = reshape([ coarse(x) for x in A ], size(A))
fine(A::AbstractArray) = reshape([ fine(x) for x in A ], size(A))

multilevel{T}(M, xc::T, xf::T) = MultilevelState{M,T}(xc, xf)
multilevel{T}(M, x::T) = MultilevelState{M,T}(x, x)

function sample{M,T}(model, scheme, state0::MultilevelState{M,T})
  subscheme = subdivide(scheme, M)
  prevstate_coarse = coarse(state0)
  prevstate_fine = fine(state0)
  Δw = zero(wiener_type(model))
  for j in 1:M
    Δw += δw = wiener(model, subscheme)
    prevstate_fine = step(model, subscheme, prevstate_fine, δw)
  end
  prevstate_coarse = step(model, scheme, prevstate_coarse, Δw)
  MultilevelState{M,T}(prevstate_coarse, prevstate_fine)
end
