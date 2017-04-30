using SDEModels
using Base.Test

ex1 = :(dX = Y*a*dt + b*dW1 + dW2)

@testset "codegen" begin
  @test collect(SDEModels.symbols(ex1)) == [:dX, :Y, :a, :dt, :b, :dW1, :dW2]
  SDEModels.replace_symbols!(ex1, Dict(:dX => :(sin(0)), :dt => :t))
  @test collect(SDEModels.symbols(ex1)) == [:Y, :a, :t, :b, :dW1, :dW2]
  @test SDEModels.cat_expressions([:(a+b), :(c-d)]) == :([a+b,c-d])
  @test SDEModels.cat_expressions([:(a+b) :(c-d); :e 1]) == :([a+b c-d; e 1])
end

@sde_model TestModel begin
    dX = Y*a*dt + b*dW1 + d*dW2
    dY = Z*c*dt +   dW2 +   dW3
    dZ = X*e*dt + f*dW3 -   dW4
end

@sde_model Wiener dX = dW
@sde_model Deterministic dX = X*dt

m1 = BlackScholes(0.01, 0.3)
m2 = Heston(0.01, 10.0, 0.3, 0.1, 0.4)
m3 = TestModel(1, 2, 3, 4, 5, 6)
wiener = Wiener()
deterministic = Deterministic()

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
  @test drift(m1, 100.0) ≈ 100.0 * 0.01
  @test drift(m2, [100.0, 0.4]) ≈ [0.01*100.0, 10.0*(0.3-0.4)]
  @test drift(m3, [0.1, 0.2, 0.3]) ≈ [0.2*1, 0.3*2, 0.1*3]
  @test drift(wiener, 100.0) ≈ 0.0
  @test drift(deterministic, 100.0) ≈ 100.0
end

@testset "diffusion" begin
  @test diffusion(m1, 100.0) ≈ 100.0 * 0.3
  @test diffusion(m2, [100.0, 0.4]) ≈ [100.0*sqrt(0.4) 0.0; 0.4*0.1*sqrt(0.4) sqrt(1-0.4^2)*0.1*sqrt(0.4)]
  @test diffusion(m3, [0.1, 0.2, 0.3]) ≈ [4 5 0 0; 0 1 1 0; 0 0 6 -1]
  @test diffusion(wiener, 100.0) ≈ 1.0
  @test diffusion(deterministic, 100.0) ≈ 0.0
end

@testset "simulation" begin
  @test size(simulate(m1, EulerMaruyama(0.01), 100.0, 1000)) == (1,1000)
  @test size(simulate(m1, Milstein(0.01), 100.0, 1000)) == (1,1000)
  @test size(simulate(m2, EulerMaruyama(0.01), [100.0, 0.4], 500)) == (2,500)
  @test sample(Deterministic(), EulerMaruyama(1), 1) ≈ 2
  @test sample(Deterministic(), Milstein(1), 1) ≈ 2
  @test simulate(deterministic, EulerMaruyama(1.0), 1.0, 5) ≈ [2 4 8 16 32]
  @test simulate(deterministic, Milstein(1.0), 1.0, 5) ≈ [2 4 8 16 32]
  @test_approx_eq_eps subsample(Deterministic(), EulerMaruyama(1), 1, 10000000) e 1e-6
  # @test_approx_eq_eps simulate(Deterministic(), EulerMaruyama(1), 1, 1; substeps=10000000)[] e 1e-6
end
