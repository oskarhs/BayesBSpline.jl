include(joinpath(@__DIR__, "..", "src", "BayesBSpline.jl"))
using .BayesBSpline 
using BenchmarkTools, Distributions, Plots, Random
using RCall

function plot_density_estimate()
    rng = Xoshiro(2)
    d = Beta(2.0, 5.0)
    n = 10^4
    x = rand(rng, Beta(2.0, 5.0), n)

    kernel_est = convert(
        Array{Float64}, 
        R"""
        dens = density($x, bw="SJ")
        matrix(c(dens$x, dens$y), ncol=2)
        """
    )
    K = 40
    @btime f = fit(CubicSplineDensity, $x, $K)
    f = fit(CubicSplineDensity, x, K)

    t = LinRange(0.0, 1.0, 1001)
    p = plot(t, f.(t), ylims=[0.0, Inf], label="f", xlabel="x", ylabel="Density", color="red", lwd=2.5)
    plot!(p, t, pdf.(d, t), label="f₀", ls=:dash, color="blue", lwd=2.5)
    plot!(p, kernel_est[:,1], kernel_est[:,2], color="black", lwd=2.5)
    savefig(p, "example.pdf")

    p2 = plot(4:K, weights(f), xlabel="k")
    savefig(p2, "example2.pdf")
end

plot_density_estimate()