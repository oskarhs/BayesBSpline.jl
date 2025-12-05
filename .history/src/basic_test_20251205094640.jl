using Distributions, Plots, Random
include(joinpath(@__DIR__ ,"BayesBSpline.jl"))
using .BayesBSpline
using BSplineKit

means = [0.0, 5.0, 15.0, 30.0, 60.0]
sds = [0.5, 1.0, 2.0, 4.0, 8.0]


rng = Random.default_rng()
#d_true = Laplace()
#d_true = LogNormal()
#d_true = Uniform()
#d_true = SymTriangularDist()
#d_true = MixtureModel([Normal(0, 0.5), Normal(2, 0.1)], [0.4, 0.6])
d_true = MixtureModel([Normal(means[i], sds[i]) for i in eachindex(means)], fill(0.2, 5))

x = rand(rng, d_true, 1500)
#bsm = BayesBSpline.BSMModel(x, BSplineBasis(BSplineOrder(4), LinRange(minimum(x), maximum(x), 98)))
bsm = BayesBSpline.BSMModel(x)


R = maximum(x) - minimum(x)
x_est = (x .- minimum(x) .+ 0.05*R) / (1.1*R)
#x = LinRange(0, 1, 1000)

# Run the Gibbs sampler
n_samples = 5000
n_burnin = 1000
θ, β, τ2s, δ2s = BayesBSpline.sample_posterior(rng, bsm, n_samples, n_burnin)

# Plotting
bs = BSplineKit.basis(bsm)
K = length(bs)
θ_mean = [mean(θ[k,:]) for k in 1:K]
#basis = BayesBSpline.PSplineBasis(BSplineOrder(4), K)


#t = LinRange(0, 1, 10001)
#t_orig = minimum(x) .+ R*t
t = LinRange(minimum(x), maximum(x), 10001)
#kdest = kde(x; bandwidth=PosteriorStats.isj_bandwidth(x))
kdest = PosteriorStats.kde_reflected(x)

#med = pointwise_quantiles(θ, basis, t, 0.5) / (1.1*R)
#lower = pointwise_quantiles(θ, basis, t, 0.05) / (1.1*R)
#upper = pointwise_quantiles(θ, basis, t, 0.95) / (1.1*R)

S = Spline(bs, BayesBSpline.theta_to_coef(θ_mean, bs))
p = Plots.plot()
Plots.plot!(p, t, S.(t), color=:black, lw=1.2, label="Posterior mean")
#Plots.plot!(p, minimum(x) .- 0.05*R .+ 1.1*R*t, med, color=:blue, lw=1.2, label="Posterior median")
#Plots.plot!(p, minimum(x) .- 0.05*R .+ 1.1*R*t, lower, color=:green, ls=:dash, label="90% CI", alpha=0.5)
#Plots.plot!(p, minimum(x) .- 0.05*R .+ 1.1*R*t, upper, color=:green, label="", ls=:dash, alpha=0.5)
#Plots.plot!(p, minimum(x) .- 0.05*R .+ 1.1*R*t, upper, fillrange=lower, fillcolor=:green, fillalpha=0.2, label="", color=:transparent)

Plots.plot!(kdest.x, kdest.density, color=:grey, label="KDE", lw=1.2)
Plots.plot!(p, t, pdf(d_true, t), color=:red, label="True", lw=1.2, alpha=0.5)
#xlims!(p, -5, 5)
p