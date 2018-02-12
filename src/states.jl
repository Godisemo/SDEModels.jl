state(x::Real) = x
state(x::SVector) = x
state(x::AbstractArray) = SVector(x...)
export state
