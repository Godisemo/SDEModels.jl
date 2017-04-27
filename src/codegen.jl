import Calculus: simplify
import DataStructures: OrderedSet, OrderedDict

# =================================
# main codegen functions and macros
# =================================

function sde_struct(typename::Symbol, d::Integer, m::Integer, parameter_vars)
  parameter_defs = map(p -> :($p::Float64), parameter_vars)
  quote
    export $typename
    immutable $typename <: AbstractSDE{$d,$m}
      $(esc.(parameter_defs)...)
    end
  end
end

function sde_function(typename::Symbol, functionname::Symbol, model_vars, parameter_vars, ex)
  replacements = Dict{Any,Any}(0 => 0.0)
  if length(model_vars) == 1
    push!(replacements, first(model_vars) => :x)
  else
    merge!(replacements, Dict(j => :(x[$i]) for (i,j) in enumerate(model_vars)))
  end
  merge!(replacements, Dict(map(s -> s => :(model.$s), parameter_vars)))
  quote
    function $(esc(:(SDEModels.$functionname)))(model::$(esc(typename)), x)
      $(replace_symbols(ex, replacements))
    end
  end
end

macro sde_model(typename::Symbol, ex::Expr)
  if ex.head == :block
    equations = filter(e -> e.head == :(=), ex.args)
  elseif ex.head == :(=)
    equations = [ex]
  else
    throw("Expression must be block or assignment $(ex.head)")
  end

  vars = union(symbols.(equations)...)
  model_vars = OrderedDict(model_variable.(equations))
  process_vars = OrderedDict(union(process_variable.(equations)...))
  differentials = union(keys(model_vars), keys(process_vars), [:dt])
  drift_equations = [differential_expression(e.args[2], :dt, differentials) for e in equations]
  diffusion_equations = [differential_expression(e.args[2], dw, differentials) for e in equations, dw in keys(process_vars)]

  drift_parameter_vars = setdiff(union(symbols.(drift_equations)...), values(model_vars))
  diffusion_parameter_vars = setdiff(union(symbols.(diffusion_equations)...), values(model_vars))
  parameter_vars = union(drift_parameter_vars, diffusion_parameter_vars)

  blk = Expr(:block)
  append!(blk.args, sde_struct(typename, length(equations), length(process_vars), parameter_vars).args)
  append!(blk.args, sde_function(typename, :drift, values(model_vars), parameter_vars, cat_expressions(drift_equations)).args)
  append!(blk.args, sde_function(typename, :diffusion, values(model_vars), parameter_vars, cat_expressions(diffusion_equations)).args)
  blk
end

# =================================
# specific codegen helper functions
# =================================

function differential_expression(ex, target::Symbol, differentials)
  replacements = Dict(s => 0 for s in differentials)
  replacements[target] = 1
  simplify(replace_symbols(ex, replacements))
end

function differential_variable(sym::Symbol)
  str = string(sym)
  startswith(str, "d") ? sym => Symbol(str[2:end]) : nothing
end

function model_variable(ex::Expr)
  ex.head == :(=) ? differential_variable(ex.args[1]) : nothing
end

function process_variable(ex::Expr)
  if ex.head == :(=)
    f = s -> startswith(lowercase(string(s)), "dw")
    map(differential_variable, filter(f, symbols(ex.args[2])))
  end
end

# ================================
# generic codegen helper functions
# ================================

cat_expressions{T}(x::Array{T,1}) =
  length(x) == 1 ? x[1] : Expr(:vect, x...)
cat_expressions{T}(x::Array{T,2}) =
  length(x) == 1 ? x[1] : Expr(:vcat, mapslices(r -> Expr(:row, r...), x, 2)...)

replace_symbols(ex, dict) = replace_symbols!(copy(ex), dict)
replace_symbols(sym::Symbol, dict) = replace_symbols!(sym, dict)
replace_symbols!(ex, pair::Pair) = replace_symbols!(ex, Dict(pair))
replace_symbols!(ex, dict::Associative) = haskey(dict, ex) ? dict[ex] : ex
function replace_symbols!(ex::Expr, dict::Associative)
  for i in 1:length(ex.args)
    ex.args[i] = replace_symbols!(ex.args[i], dict)
  end
  ex
end

symbols(ex) = symbols!(OrderedSet{Symbol}(), ex)
symbols!(set::OrderedSet{Symbol}, ex) = set
symbols!(set::OrderedSet{Symbol}, sym::Symbol) = push!(set, sym)
function symbols!(set::OrderedSet{Symbol}, ex::Expr)
  start = ex.head == :call ? 2 : 1
  for arg in ex.args[start:end]
    symbols!(set, arg)
  end
  set
end
