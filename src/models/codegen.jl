import Calculus: simplify, differentiate
import DataStructures: OrderedSet, OrderedDict
import Parameters: with_kw

# =================================
# main codegen functions and macros
# =================================

function sde_struct(typename::Symbol, supertype::Symbol, d::Integer, m::Integer, parameter_vars, docstring)
  block = Expr(:block, [:($p::Float64) for p in parameter_vars]...)
  typedef =  Expr(:type, false, :($typename <: SDEModels.$supertype{$d,$m}), block)
  if length(parameter_vars) > 0
     typedef = with_kw(typedef)
  end
  quote
    @doc $docstring -> $typedef
    export $typename
  end
end

eltype_promote{T1<:Number,T2<:Number}(::Type{T1}, val::T2) = convert(promote_type(T1,T2), val)
eltype_promote{T1<:Number,T2<:Number,S}(::Type{T1}, val::SVector{S,T2}) = convert(SVector{S,promote_type(T1,T2)}, val)
eltype_promote{T1<:Number,T2<:Number,S1,S2}(::Type{T1}, val::SMatrix{S1,S2,T2}) = convert(SMatrix{S1,S2,promote_type(T1,T2)}, val)

function sde_state_function(typename::Symbol, functionname::Symbol, model_vars, parameter_vars, ex)
  docstring = "$typename: $ex"
  replacements = Dict()
  if length(model_vars) == 1
    push!(replacements, first(model_vars) => :x)
  else
    merge!(replacements, Dict(j => :(x[$i]) for (i,j) in enumerate(model_vars)))
  end
  merge!(replacements, Dict(map(s -> s => :(model.$s), parameter_vars)))
  m = length(model_vars)
  ex = replace_symbols(ex, replacements)
  quote
    @doc $docstring ->
    function (SDEModels.$functionname){S,T}(model::$typename, t::Number, state::SDEModels.SDEState{$m,S,T})
      x = statevalue(state)
      SDEModels.eltype_promote(S, $ex)
    end
  end
end

function sde_model_function(typename::Symbol, functionname::Symbol, ex)
  quote
    function (SDEModels.$functionname)(::$typename)
      $ex
    end
  end
end

function sde_model(typename::Symbol, ex::Expr)
  if ex.head == :block
    equations = filter(e -> e.head == :(=), ex.args)
  elseif ex.head == :(=)
    equations = [ex]
  else
    error("Expression must be block or assignment $(ex.head)")
  end

  model_vars = foldl(merge!, matchdict(r"(?<=^d).*", e.args[1]) for e in equations)
  time_vars = OrderedDict([:dt => :t])
  process_vars = foldl(merge!, matchdict(r"(?<=^d)[wW].*", e.args[2]) for e in equations)
  differentials = union(keys(time_vars), keys(process_vars))
  drift_expressions = [factor_extract(e.args[2], :dt, differentials) for e in equations]
  drift = cat_expressions(drift_expressions)
  drift_jacobian_expressions = [differentiate(f_i, x_j) for f_i in drift_expressions, x_j in values(model_vars)]
  drift_jacobian = isempty(drift_jacobian_expressions) ? 0 : cat_expressions(drift_jacobian_expressions)
  diffusion_expressions = [factor_extract(e.args[2], dw, differentials) for e in equations, dw in keys(process_vars)]
  diffusion = isempty(diffusion_expressions) ? 0 : cat_expressions(diffusion_expressions)
  parameter_vars = setdiff(union(symbols(drift), symbols(diffusion)), union(values(model_vars), [:t]))

  if isempty(intersect(symbols(diffusion), values(model_vars)))
    supertype = :StateIndependentDiffusion
  else
    supertype = :AbstractSDE
  end

  docstring = """
    Model variables: $(join(values(model_vars), ", "))

    Process variables: $(join(values(process_vars), ", "))

    Parameter variables: $(join(parameter_vars, ", "))

    Definition:

    $(join(string.(equations), "\n\n"))
  """
  blk = Expr(:block)
  append!(blk.args, sde_struct(typename, supertype, length(equations), length(process_vars), parameter_vars, docstring).args)
  append!(blk.args, sde_state_function(typename, :drift, values(model_vars), parameter_vars, drift).args)
  append!(blk.args, sde_state_function(typename, :drift_jacobian, values(model_vars), parameter_vars, drift_jacobian).args)
  append!(blk.args, sde_state_function(typename, :diffusion, values(model_vars), parameter_vars, diffusion).args)
  append!(blk.args, sde_model_function(typename, :variables, :($(values(model_vars)...))).args)
  blk
end

macro sde_model(typename, ex)
  esc(sde_model(typename, ex))
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
  # isa(factor, Number) ? Float64(factor) : factor
end

cat_expressions{T}(x::Array{T,1}) =
  length(x) == 1 ? x[1] : :(StaticArrays.SVector{$(size(x)...)}($(x...)))

cat_expressions{T}(x::Array{T,2}) =
  length(x) == 1 ? x[1] : :(StaticArrays.SMatrix{$(size(x)...)}($(x...)))

replace_symbols(ex, dict) = replace_symbols!(copy(ex), dict)
replace_symbols(sym::Symbol, dict) = replace_symbols!(sym, dict)
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
