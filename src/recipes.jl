using RecipesBase

@recipe function plot{T<:AbstractState}(series::AbstractArray{T,2})
  for i in 1:size(series,2)
    @series series[:,i]
  end
end

@recipe function plot{T<:AbstractState}(series::AbstractArray{T,1})
  t = statetime(series)
  z = statevalue(series)
  D = length(first(z))
  x = reshape(reinterpret(Float64, z), D, length(t))
  layout --> (D,1)
  # seriescolor --> reshape(Plots.get_color_palette(:auto, default(:bgcolor), D), 1, D)
  t, x'
end

@recipe function plot{D,T<:AbstractState}(model::AbstractSDE{D}, series::AbstractArray{T})
  variable_names = reshape([variables(model)...], 1, D)
  labels --> variable_names
  series
end
