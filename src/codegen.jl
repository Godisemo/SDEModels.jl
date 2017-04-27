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
  replacements = Dict()
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

  model_vars = foldl(merge!, matchdict(r"(?<=^d).*", e.args[1]) for e in equations)
  time_vars = OrderedDict([:dt => :t])
  process_vars = foldl(merge!, matchdict(r"(?<=^d)[wW].*", e.args[2]) for e in equations)
  differentials = union(keys(time_vars), keys(process_vars))
  drift = cat_expressions([factor_extract(e.args[2], :dt, differentials) for e in equations])
  diffusion = cat_expressions([factor_extract(e.args[2], dw, differentials) for e in equations, dw in keys(process_vars)])
  parameter_vars = setdiff(union(symbols(drift), symbols(diffusion)), values(model_vars))

  blk = Expr(:block)
  append!(blk.args, sde_struct(typename, length(equations), length(process_vars), parameter_vars).args)
  append!(blk.args, sde_function(typename, :drift, values(model_vars), parameter_vars, drift).args)
  append!(blk.args, sde_function(typename, :diffusion, values(model_vars), parameter_vars, diffusion).args)
  blk
end

# ================================
# generic codegen helper functions
# ================================

function matchdict(r::Regex, ex)
  syms = symbols(ex)
  dict = OrderedDict()
  for sym in syms
    str = string(sym)
    m = match(r, str)
    if m != nothing
      push!(dict, sym => Symbol(m.match))
    end
  end
  dict
end

function factor_extract(ex, one_sym::Symbol, zero_syms)
  replacements = Dict(s => 0 for s in zero_syms)
  replacements[one_sym] = 1
  factor = simplify(replace_symbols(ex, replacements))
  isa(factor, Number) ? Float64(factor) : factor
end

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
