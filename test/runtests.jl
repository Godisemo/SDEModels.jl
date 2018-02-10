using SDEModels
using Base.Test

ex1 = :(dX = Y*a*dt + b*dW1 + dW2)

@testset "codegen" begin
  @test collect(SDEModels.symbols(ex1)) == [:dX, :Y, :a, :dt, :b, :dW1, :dW2]
  SDEModels.replace_symbols!(ex1, Dict(:dX => :(sin(0)), :dt => :t))
  @test collect(SDEModels.symbols(ex1)) == [:Y, :a, :t, :b, :dW1, :dW2]
  @test SDEModels.cat_expressions([:(a+b), :(c-d)]) == :(StaticArrays.SVector{2}(a+b,c-d))
  @test SDEModels.cat_expressions([:(a+b) :(c-d); :e 1]) == :(StaticArrays.SMatrix{2,2}(a+b, e, c-d, 1))
end

@sde_model TestModel begin
    dX = Y*a*dt + b*dW1 + d*dW2
    dY = Z*c*dt +   dW2 +   dW3
    dZ = X*e*dt + f*dW3 -   dW4
end

@sde_model Wiener dX = dW
@sde_model Deterministic dX = X*dt
@sde_model TimeDependent dX = t*a*X*dt + b*X*dW

m1 = BlackScholes(0.01, 0.3)
m2 = Heston(0.01, 10.0, 0.3, 0.1, 0.4)
m3 = TestModel(1, 2, 3, 4, 5, 6)
wiener = Wiener()
deterministic = Deterministic()
timedependend = TimeDependent(1, 2)

@testset "model creation" begin
  @test fieldnames(m1) == [:r, :σ]
  @test fieldnames(m2) == [:r, :κ, :θ, :σ, :ρ]
  @test fieldnames(m3) == [:a, :c, :e, :b, :d, :f]
  @test dim(m1) == (1,1)
  @test dim(m2) == (2,2)
  @test dim(m3) == (3,4)
  @test dim(wiener) == (1,1)
  @test dim(deterministic) == (1,0)
end

@testset "drift" begin
  t0 = 0.0
  t1 = 1.0
  s2 = state([100.0, 0.4])
  @test drift(m1, t0, state(100.0)) ≈ 100.0 * 0.01
  @test drift(m2, t0, s2) ≈ [0.01*100.0, 10.0*(0.3-0.4)]
  @test drift(m3, t0, state([0.1, 0.2, 0.3])) ≈ [0.2*1, 0.3*2, 0.1*3]
  @test drift(wiener, t0, state(100.0)) ≈ 0.0
  @test drift(deterministic, t0, state(100.0)) ≈ 100.0
  @test drift(timedependend, t0, state(3.0)) ≠ drift(timedependend, t1, state(3.0))
  @test drift(timedependend, 4.0, state(3.0)) ≈ 12
  @test corrected_drift(m2, t0, s2, 0.65) ≈ drift(m2, t0, s2) - [s2.x[2]*s2.x[1]; m2.σ^2/2] * 0.65
end

@testset "diffusion" begin
  t0 = 0.0
  @test diffusion(m1, t0, state(100.0)) ≈ 100.0 * 0.3
  @test diffusion(m2, t0, state([100.0, 0.4])) ≈ [100.0*sqrt(0.4) 0.0; 0.4*0.1*sqrt(0.4) sqrt(1-0.4^2)*0.1*sqrt(0.4)]
  @test diffusion(m3, t0, state([0.1, 0.2, 0.3])) ≈ [4 5 0 0; 0 1 1 0; 0 0 6 -1]
  @test diffusion(wiener, t0, state(100.0)) ≈ 1.0
  @test diffusion(deterministic, t0, state(100.0)) ≈ 0.0
end

@testset "simulation" begin
  t0 = 0.0
  @test size(simulate(m1, EulerMaruyama(0.01), t0, state(100.0), 1000)[1]) == (1001,)
  @test size(simulate(m1, Milstein(0.01), t0, state(100.0), 1000)[1]) == (1001,)
  @test size(simulate(m2, EulerMaruyama(0.01), t0, state([100.0, 0.4]), 500)[1]) == (501,)
  @test statevalue(sample(Deterministic(), EulerMaruyama(1), t0, state(1.0), 1)) ≈ 2
  @test statevalue(sample(Deterministic(), Milstein(1), t0, state(1.0), 1)) ≈ 2
  @test statevalue(simulate(deterministic, EulerMaruyama(1.0), t0, state(1.0), 5)[1]) ≈ [1, 2, 4, 8, 16, 32]
  @test statevalue(simulate(deterministic, Milstein(1.0), t0, state(1.0), 5)[1]) ≈ [1, 2, 4, 8, 16, 32]
end

@testset "multilevel" begin
  substeps = 4
  coarse = EulerMaruyama(0.1)
  fine = subdivide(coarse, substeps)
  x0 = state([100.0, 0.4])
  srand(0)
  xc, xf, tc, tf = SDEModels._multilevel_simulate(m2, coarse, fine, substeps, 0.0, x0, 10, 100)
  srand(0)
  x, t = simulate(m2, fine, 0.0, x0, 40, 100)
  @test t == tf
  @test isapprox(sum(norm.(statevalue(xf) - statevalue(x)).^2), 0, atol=1e-16)
  srand(0)
  yc, yf = SDEModels._multilevel_sample(m2, coarse, fine, substeps, 0.0, x0, 10, 100)
  @test isapprox(sum(norm.(statevalue(xf[end,:]) - statevalue(yf)).^2), 0, atol=1e-16)
  @test isapprox(sum(norm.(statevalue(xc[end,:]) - statevalue(yc)).^2), 0, atol=1e-16)

  ml_scheme = MultilevelScheme(coarse, 4, 0:4)
  srand(0)
  ml_x1, ml_t = simulate(m2, ml_scheme, 0.0, x0, 1, 1)
  @test length(unique(extrema.(ml_t))) == 1
  srand(0)
  ml_x2 = sample(m2, ml_scheme, 0.0, x0, 1, 1)
  @test isapprox(sum(norm.(statevalue(last.(ml_x1)) - statevalue(last.(ml_x2)))), 0, atol=1e-13)
end
