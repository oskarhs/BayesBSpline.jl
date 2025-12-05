using Distributions, Plots, Random, KernelDensity, PosteriorStats
include(joinpath(@__DIR__ ,"BayesBSpline.jl"))
using .BayesBSpline
using BSplineKit

harp_means = [0.0, 5.0, 15.0, 30.0, 60.0]
harp_sds = [0.5, 1.0, 2.0, 4.0, 8.0]


rng = Random.default_rng()
#d_true = Laplace()
#d_true = LogNormal()
#d_true = Normal()
#d_true = SymTriangularDist()
d_true = MixtureModel([Normal(0, 0.5), Normal(2, 0.1)], [0.4, 0.6])
#d_true = MixtureModel(vcat(Normal(0, 1) ,[Normal(0.5*j, 0.1) for j in -2:2]), [0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
#d_true = MixtureModel([Normal(harp_means[i], harp_sds[i]) for i in eachindex(means)], fill(0.2, 5))
#d_true = Beta(1.2, 1.2)

x = rand(rng, d_true, 250)

#bsm = BayesBSpline.BSMModel(x, BSplineBasis(BSplineOrder(4), LinRange(minimum(x), maximum(x), 98)))
bsm = BayesBSpline.BSMModel(x)

#bsm2 = BayesBSpline.BSMModel(x, (0, 1))
#bsm = BayesBSpline.BSMModel(x, 200, (0,1))
#bsm = BayesBSpline.BSMModel(x; n_bins=nothing)

R = maximum(x) - minimum(x)
#x = LinRange(0, 1, 1000)

# Run the Gibbs sampler
n_samples = 5000
n_burnin = 1000
bsmc = BayesBSpline.sample_posterior(rng, bsm, n_samples, n_burnin)

# Plotting
bs = BSplineKit.basis(bsm)
K = length(bs)

#t = LinRange(0, 1, 10001)
#t_orig = minimum(x) .+ R*t
t = LinRange(boundaries(basis(bsm))[1], boundaries(basis(bsm))[2], 2001)
kdest = kde(x; bandwidth=PosteriorStats.isj_bandwidth(x))
#kdest = PosteriorStats.kde_reflected(x, bounds=(0,1))

qs = [0.025, 0.5, 0.975]
quants = quantile(bsmc, t, qs)
low, med, up = (quants[:,i] for i in eachindex(qs))


f_samp = BayesBSpline.evaluate_posterior_splines(bsmc, t, [100, 200, 300, 400])
p = plot()
for i in axes(f_samp, 2)
    plot!(p, t, f_samp[:,i], color=:grey)
end
p

#S = Spline(bs, BayesBSpline.theta_to_coef(θ_mean, bs))
p = Plots.plot()
Plots.plot!(p, t, mean.(bsmc, t), color=:black, lw=1.2, label="Posterior mean")
Plots.plot!(p, t, med, color=:blue, lw=1.2, label="Posterior median")
Plots.plot!(p, t, low, color=:green, ls=:dash, label="95% CI", alpha=0.5)
Plots.plot!(p, t, up, color=:green, label="", ls=:dash, alpha=0.5)
Plots.plot!(p, t, up, fillrange=low, fillcolor=:green, fillalpha=0.2, label="", color=:transparent)

Plots.plot!(kdest.x, kdest.density, color=:grey, label="KDE", lw=1.2)
Plots.plot!(p, t, pdf(d_true, t), color=:red, label="True", lw=1.2, alpha=0.5)
#xlims!(p, -5, 5)
p

