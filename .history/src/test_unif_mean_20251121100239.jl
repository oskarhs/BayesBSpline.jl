using Random, Statistics
using ForwardDiff, BandedMatrices
using LinearAlgebra, Distributions, BSplineKit, Plots, Optim, Integrals

include(joinpath(@__DIR__, "BayesBSpline.jl"))
using .BayesBSpline

μ = BayesBSpline.find_uniform_prior_mean_β(50)


φ = 0.95

# Note to self: when we sample from these we automatically get samples of type T!
dist_σ2 = InverseGamma(a_σ, b_σ)
dist_τ2 = InverseGamma(a_τ, b_τ) # (1, 5e-3) yields quite symmetric prior
dist_δ2 = InverseGamma(a_δ, b_δ) # (1, 5e-1) yields quite symmetric prior
a_σ = 2.5
b_σ = 1.5 # b_σ = a_σ - 1
a_τ = 1.0
b_τ = 1e-2
a_δ = 1.1
b_δ = 0.1 # b_δ = a_δ - 1


B = 10^5

draws_new = Array{Float64}(undef, B, K)
for b in 1:B
    σ = sqrt(rand(rng, dist_σ2))
    τ = sqrt(rand(rng, dist_τ2))
    δ = sqrt.(rand(rng, dist_δ2, K-2))
    β = Vector{Float64}(undef, K-1)
    β[1] = μ[1] + σ * rand(rng, Normal())
    for k in 2:K-1
        β[k] = μ[k] + φ * (β[k-1] - μ[k-1]) + τ * δ[k-1] * rand(rng, Normal())
    end
    #β = [rand(rng, Normal(μ_new[k], sqrt(τ2 * δ[k]))) for k in 1:K-1]
    draws_new[b, :] = BayesBSpline.stickbreaking(β)
end
q05 = [quantile(draws_new[:, k], 0.05) for k in 1:K]
q95 = [quantile(draws_new[:, k], 0.95) for k in 1:K]
q50 = [quantile(draws_new[:, k], 0.5) for k in 1:K]
priormean = vec(mean(draws_new, dims=1))
t = LinRange(0, 1, 10001)
b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, K-2))
p = Plots.plot()
thetas = [priormean, q05, q50, q95]
labs = ["Mean", "Lower quantile", "Median", "Upper quantile"]
for i in eachindex(thetas)
    θ = thetas[i]
    S = Spline(b, theta_to_coef(θ, K))
    Plots.plot!(p, t, S.(t), label=labs[i])
    Plots.ylims!(p, -0.05*maximum(S.(t)), 1.05*maximum(S.(t)))
end
p