# using RecipesBase
#
# @recipe function f{D}(model::SDEModels.AbstractSDE{D}, args...)
#   variable_names = reshape([variables(model)...], 1, D)
#   labels --> variable_names
#   args
# end
#
# @recipe function f{T<:SDEState}(y::AbstractArray{T})
#   1:size(y, 1), y
# end
#
# @recipe function f{T<:SDEState}(x::AbstractArray, y::AbstractArray{T})
#   D = length(y[1])
#   data = reinterpret(Float64, y, (D, size(y)...))
#   x --> 1:size(y, 1)
#   for i in 1:size(data, 3)
#     @series x, permutedims(slicedim(data, 3, i), [2, 1])
#   end
#   layout --> (D,1)
#   # seriescolor --> reshape(Plots.get_color_palette(:auto, default(:bgcolor), D), 1, D)
# end
