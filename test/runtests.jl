using BayesBSpline, Random
using Test
import StatsAPI

@testset "CubicSplineDist: uniform initialization" begin
    x = LinRange(0.0, 1.0, 1001)

    for K = 4:20
        d = CubicSplineDist(K)
        @test isapprox(mean(d, 0.5), 1.0)
        @test isapprox(mean.(d, x), ones(length(x)))
        # @test isapprox(mean(d, x), ones(length(x)))
    end
end

@testset "CubicSplineDist: rand is density" begin
    t = LinRange(0.0, 1.0, 1001)

    for K = 4:20
        d = CubicSplineDist(K)
        s = rand(Xoshiro(1812), d)
        y = s.(t)
        midpoint = step(t) * ( 0.5*(y[1]+y[end]) + sum(@views(y[2:end-1])) )
        @test isapprox(midpoint, 1.0, atol=1e-4)
    end
end

@testset "CubicSplineDensity: uniform initialization" begin
    t = LinRange(0.0, 1.0, 1001)

    for K = 4:20
        f = CubicSplineDensity(K)
        y = f.(t)
        midpoint = step(t) * ( 0.5*(y[1]+y[end]) + sum(@views(y[2:end-1])) )
        @test isapprox(midpoint, 1.0, atol=1e-4)
    end
end

@testset "CubicSplineDensity: fit return type" begin
    x = rand(Xoshiro(1812), 200)
    K = 40

    f = StatsAPI.fit(CubicSplineDensity, x, K)
    @test typeof(f) == CubicSplineDensity
end

@testset "CubicSplineDensity: fit is density" begin
    # Test the unbinned version
    x = rand(Xoshiro(1812), 500)
    K = 40
    
    # Midpoint approximation
    t = LinRange(0.0, 1.0, 10001)
    f = StatsAPI.fit(CubicSplineDensity, x, K)
    y = f.(t)
    midpoint = step(t) * ( 0.5*(y[1]+y[end]) + sum(@views(y[2:end-1])) )
    @test isapprox(midpoint, 1.0, atol=1e-4)

    # Test the binned variant
    x = rand(Xoshiro(1812), 10_000)
    K = 40
    
    # Midpoint approximation
    t = LinRange(0.0, 1.0, 10001)
    f = StatsAPI.fit(CubicSplineDensity, x, K)
    y = f.(t)
    midpoint = step(t) * ( 0.5*(y[1]+y[end]) + sum(@views(y[2:end-1])) )
    @test isapprox(midpoint, 1.0, atol=1e-4)
end