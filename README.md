# SDEModels

Tools for working with stochastic differential equation models.

[![Build Status](https://travis-ci.org/Godisemo/SDEModels.jl.svg?branch=master)](https://travis-ci.org/Godisemo/SDEModels.jl)
[![Coverage Status](https://coveralls.io/repos/github/Godisemo/SDEModels.jl/badge.svg?branch=master)](https://coveralls.io/github/Godisemo/SDEModels.jl?branch=master)
[![codecov.io](http://codecov.io/github/Godisemo/SDEModels.jl/coverage.svg?branch=master)](http://codecov.io/github/Godisemo/SDEModels.jl?branch=master)

[![See the talk from JuliaCon 2018 about SDEModels.jl](https://img.youtube.com/vi/dy7tXk403bM/mq1.jpg)](https://www.youtube.com/watch?v=dy7tXk403bM)

The main feature of this package is that it allows you to define SDE models in a compact form, similar to the mathematical definition.
```julia
@sde_model BlackScholes dS = r*S*dt + σ*S*dW
```
The `@sde_model` macro figures out the properties of your model, and generates all code that is necessary.
In this case, the code generated is:
```julia
  immutable BlackScholes <: AbstractSDE{1,1} # 1 dimension and 1 wiener process
    r::Float64
    s::Float64
  end

  function drift(model::BlackScholes, x)
    model.r * x
  end

  function diffusion(model::BlackScholes, x)
    model.σ * x
  end
```
It is easy to see how defining your models can become cumbersome, especially when you are working with multidimensional SDEs. You can define multivariate models as
```julia
@sde_model Heston begin
  dS =     r*S*dt + sqrt(V)*S*dW1
  dV = κ*(θ-V)*dt + σ*sqrt(V)*dW2
  dW1*dW2 = ρ*dt
end
```
which generates the code
```julia
  # the parameters are arranged in the order of appearance
  immutable Heston <: AbstractSDE{2,2} # 2 dimension and 2 wiener process
    r::Float64
    κ::Float64
    θ::Float64
    σ::Float64
    ρ::Float64
  end

  # x is assumed to be arranged in the order of appearance, i.e [S, V]
  function drift(model::Heston, x)
    [model.r * x[1]
     model.κ*(model.θ-x[2])]
  end

  function diffusion(model::Heston, x)
    [sqrt(x[2])*x[1]        0
     model.σ*√x[2]*model.ρ  model.σ*sqrt(x[2])*sqrt(1-model.ρ^2)]
  end
```
