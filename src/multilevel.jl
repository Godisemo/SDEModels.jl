type MultilevelState{M,T<:AbstractState}
  xc::T
  xf::T
end

coarse(state::MultilevelState) = state.xc
fine(state::MultilevelState) = state.xf

coarse(A::AbstractArray) = reshape([ coarse(x) for x in A ], size(A))
fine(A::AbstractArray) = reshape([ fine(x) for x in A ], size(A))

multilevel{T}(M, xc::T, xf::T) = MultilevelState{M,T}(xc, xf)
multilevel{T}(M, x::T) = MultilevelState{M,T}(x, x)

Base.copy{M,T}(s::MultilevelState{M,T}) = MultilevelState{M,T}(s.xc, s.xf)

function sample{M}(model, scheme, state0::MultilevelState{M})
  subscheme = subdivide(scheme, M)
  prevstate = copy(state0)
  Δw = 0.0
  for j in 1:M
    Δw += δw = wiener(model, subscheme)
    prevstate.xf = step(model, subscheme, prevstate.xf, δw)
  end
  prevstate.xc = step(model, scheme, prevstate.xc, Δw)
  prevstate
end
