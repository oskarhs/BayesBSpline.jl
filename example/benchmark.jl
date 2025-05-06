include(joinpath(@__DIR__, "..", "src", "BayesBSpline.jl"))
using .BayesBSpline 
using BenchmarkTools, Distributions, Plots, Random
using RCall

function plot_density_estimate()
    rng = Xoshiro(2)
    d = Beta(3.0, 8.0)
    n = 5000
    x = rand(rng, d, n)

    kernel_est = convert(
        Array{Float64}, 
        R"""
        dens = density($x, bw="SJ")
        matrix(c(dens$x, dens$y), ncol=2)
        """
    )
    K = 40
    dum = CubicSplineDist(K)
    @btime fit!($dum, $x)
    @btime fit(CubicSplineDensity, $x, $K)

    f = fit(CubicSplineDensity, x, K)

    d1 = CubicSplineDist(K)
    fit!(d1, x)

    t = LinRange(0.0, 1.0, 1001)
    q = quantile(d1, t, [0.05, 0.95])

    p = plot(t, pdf.(d, t), ylims=[0.0, Inf], label="f₀", xlabel="x", ylabel="Density", color="black", lwd=2.5)
    #plot!(p, t, f.(t), label="f", ls=:dash, color="blue", lwd=2.5)
    plot!(p, t, mean.(d1, t), label="Posterior mean", color="red", lwd=2.5)
    plot!(p, t, q[:,1], color="red", label="90% credible interval", linestyle=:dash, linewidth=1.5, alpha=0.5,
            fillrange=q[:,2], fillalpha=0.1)
    plot!(p, t, q[:,2], color="red", label="", linestyle=:dash, linewidth=1.5, alpha=0.5)
    #plot!(p, kernel_est[:,1], kernel_est[:,2], color="black", lwd=2.5)
    savefig(p, "example.pdf")

    p2 = plot(4:K, weights(f), xlabel="k")
    plot!(p2, 4:K, d1.p_k, xlabel="k")
    savefig(p2, "example2.pdf")
end

plot_density_estimate()

