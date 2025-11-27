using Random, Statistics
using ForwardDiff, BandedMatrices
using LinearAlgebra, Distributions, BSplineKit, Plots, Optim, Integrals

# Numerically stable variant of log(1 + exp(x))
softplus(x::Real) = ifelse(x ≥ 0, x + log(1+exp(-x)), log(1+exp(x)))

# Numerically stable sigmoid
sigmoid(x::Real) = ifelse(x ≥ 0, 1/(1 + exp(-x)), exp(x)/(1 + exp(x)))

# Map an unconstrained K-1 dimensional vector to the K-simplex through stickbreaking
function stickbreaking(β::AbstractVector{T}) where {T<:Real}
    K = length(β) + 1
    log_π = Vector{T}(undef, K-1)
    softplus_sum = zero(T)
    for k in 1:K-1
        softplus_sum += softplus(β[k])
        log_π[k] = β[k] - softplus_sum
    end
    temp1 = exp.(log_π)
    p = vcat(temp1, 1.0 - sum(temp1))
    return p
end

# Map the simplex vector θ to the corresponding B-spline coefficient vector in terms of the (unnnormalized) B-spline basis.
function theta_to_coef(θ::AbstractVector{T}) where {T<:AbstractFloat}
    K = length(θ)
    coef = Vector{T}(undef, K)
    if K == 4
        coef = 4.0*θ
    elseif K == 5
        coef[1] = 8.0*θ[1]
        coef[2:K-1] = 4.0*θ[2:K-1]
        coef[K] = 8.0*θ[K]
    else
        for j = 1:3
            coef[j] = 4.0*(K-3.0)*θ[j] / j
        end
        for j = 4:K-3
            coef[j] = (K-3.0)*θ[j]
        end
        for j = K-2:K
            coef[j] = 4.0*(K-3.0)*θ[j] / (K-j + 1.0)
        end
    end
    return coef
end

# Maps the (unnormalized) B-spline coefficient vector to the corresponding simplex vector θ
function coef_to_theta(coef::AbstractVector{T}) where {T<:AbstractFloat}
    K = length(coef)
    θ = Vector{T}(undef, K)
    if K == 4
        θ = 0.25*coef
    elseif K == 5
        θ[1] = 0.125*coef[1]
        θ[2:K-1] = 0.25*coef[2:K-1]
        θ[k] = 0.125*coef[K]
    else
        for j = 1:3
            θ[j] = coef[j]*j/(4.0*(K-3.0))
        end
        for j = 4:K-3
            θ[j] = coef[j]*1.0/(K-3.0)
        end
        for j = K-2:K
            θ[j] = coef[j]*(K-j + 1.0)/(4.0*(K-3.0))
        end
    end
    return θ
end

# We adjust params of β[1] first, then β[2] and so on
K = 50
rng = Random.default_rng()
ind = 1

p0 = coef_to_theta(ones(K))

# This is zero of iff μ has the value that makes stickbreaking(β)[k] == p0[k].
# Extend this to cover the case where the variance has its own prior too.
# I think in this case we just get a T-distribution on μ instead of the normal, which does not complicate the optimization
function loss(μ_k, p0, p0_cum, k, τ2)
    prob = solve(IntegralProblem((x, p) -> pdf(Normal(μ_k, sqrt(τ2)), x) * sigmoid(x), (-Inf, Inf)), QuadGKJL()).u
    return (p0[k] - (1-p0_cum[k])*prob)^2
end

# Use for k ≥ 2
# Also: reuse the same random sample
function loss1(μ_k, p0, k, z, τ1)
    samp = Vector{Float64}(undef, size(z, 1))
    for m in axes(z, 1)
        ϵ_k = μ_k + τ1[m] * z[m,k]
        β = vcat(ϵ_k, zeros(length(p0)-1-k)) # Just augment with zeros, the values of the remaining β's do not matter
        samp[m] = stickbreaking(β)[k]
    end
    return (p0[k] - mean(samp))^2
end
function loss2(μ_k, μ, p0, k, z, τ1, τ, δ, φ)
    samp = Vector{Float64}(undef, size(z, 1))
    for m in axes(z, 1)
        ϵ = Vector{Float64}(undef, k)
        ϵ[1] = μ[1] + τ1[m] * z[m,1]
        for j in 2:k-1
            ϵ[j] = μ[j] + φ * (ϵ[j-1] - μ[j-1]) + τ[m] * sqrt(δ[m,j-1]) * z[m,j]
        end
        #ϵ = ifelse(k ≥ 3, μ[2:end] + τ[m] * sqrt.(δ[m,2:k-1]) .* z[m,2:k-1], Float64[])
        ϵ[k] = μ_k + φ * (ϵ[k-1] - μ[k-1]) + τ[m] * sqrt(δ[m,k-1]) * z[m,k]
        β = vcat(ϵ, zeros(length(p0)-1-k)) # Just augment with zeros, the values of the remaining β's do not matter
        samp[m] = stickbreaking(β)[k]
    end
    return (p0[k] - mean(samp))^2
end

τ2 = 1
μ = zeros(K-1)
p0_cum = vcat(0, cumsum(p0))

for k in 1:K-1
    lossfunc = μ_k -> loss(μ_k, p0, p0_cum, k, τ2)
    res = Optim.optimize(lossfunc, -30, 20, GoldenSection())
    μ[k] = Optim.minimizer(res)
end


φ = 0.95
M = 10_000
a_δ = 1.1
b_δ = a_δ - 1
dist_τ1 = InverseGamma(1, 1e-1) # (1, 1e-1) is reasonable, but would like a bit more variance in the first component
#dist_τ1 = InverseGamma(2.5, 2.5-1)
dist_τ2 = InverseGamma(1, 5e-3) # (1, 5e-3) yields quite symmetric prior
#dist_τ2 = InverseGamma(1, 1e-2) # (1, 5e-3) yields quite symmetric prior
dist_δ = InverseGamma(a_δ, b_δ) # (1, 5e-1) yields quite symmetric prior
z = rand(rng, Normal(), (M, K-1))
τ = sqrt.(rand(rng, dist_τ2, M))
τ1 = sqrt.(rand(rng, dist_τ1, M))
δ = rand(rng, dist_δ, (M, K-2))
μ_new = zeros(K-1)
lossfunc = μ_k -> loss1(μ_k, p0, 1, z, τ1)
res = Optim.optimize(lossfunc, -40, 20, GoldenSection())
μ_new[1] = Optim.minimizer(res)
for k in 2:K-1
    println(k)
    lossfunc = μ_k -> loss2(μ_k, μ_new[1:k-1], p0, k, z, τ1, τ, δ, φ)
    res = Optim.optimize(lossfunc, -40, 20, GoldenSection())
    μ_new[k] = Optim.minimizer(res)
end

B = 10^5

draws_new = Array{Float64}(undef, B, K)
for b in 1:B
    τ1 = rand(rng, dist_τ1)
    τ2 = rand(rng, dist_τ2)
    δ = rand(rng, dist_δ, K-2)
    β = Vector{Float64}(undef, K-1)
    β[1] = μ_new[1] + sqrt(τ1) * rand(rng, Normal())
    for k in 2:K-1
        β[k] = μ_new[k] + φ * (β[k-1] - μ_new[k-1]) + sqrt(τ2 * δ[k-1]) * rand(rng, Normal())
    end
    #β = [rand(rng, Normal(μ_new[k], sqrt(τ2 * δ[k]))) for k in 1:K-1]
    draws_new[b, :] = stickbreaking(β)
end
q05 = [quantile(draws_new[:, k], 0.05) for k in 1:K]
q95 = [quantile(draws_new[:, k], 0.95) for k in 1:K]
q50 = [quantile(draws_new[:, k], 0.5) for k in 1:K]
priormean = vec(mean(draws_new, dims=1))
t = LinRange(0, 1, 10001)
basis = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, K-2))
p = Plots.plot()
thetas = [priormean, q05, q50, q95]
labs = ["Mean", "Lower quantile", "Median", "Upper quantile"]
for i in eachindex(thetas)
    θ = thetas[i]
    S = Spline(basis, theta_to_coef(θ))
    Plots.plot!(p, t, S.(t), label=labs[i])
    Plots.ylims!(p, -0.05*maximum(S.(t)), 1.05*maximum(S.(t)))
end
p

# This is acceptable for now, can run a few experiments with this prior configuration
# Although ideally, I would like for the prior to be more diffuse
# We may have to use a proper 1st order random walk instead of an impropert one to make that idea work
# otherwise, we cannot simulate from the prior, which makes it difficult to determine an appropriate value for μ
# I.e. we do β_{k+1} - μ_{k+1} = φ * (β_k - μ_k) + u_k, with u_k normal and φ equal to some fixed positive value.
# Also: β_1 from a normal distribution, where we tune the scale of the Inverse Gamma prior variance to yield something reasonable

# This approach works, we recover something close to the true density with enough samples, but there is too little smoothing!
# I think it is 

τ1 = rand(rng, dist_τ1)
τ2 = rand(rng, dist_τ2)
δ = rand(rng, dist_δ, K-1)
β = Vector{Float64}(undef, K-1)
β[1] = μ_new[1] + sqrt(τ1) * rand(rng, Normal())
for k in 2:K-1
    β[k] = μ_new[k] + φ * (β[k-1] - μ_new[k-1]) + sqrt(τ2 * δ[k-1]) * rand(rng, Normal())
end
S = Spline(basis, theta_to_coef(stickbreaking(β), K))
Plots.plot(t, S.(t))
Plots.ylims!(-0.05*maximum(S.(t)), 1.05*maximum(S.(t)))


using Turing

function myloglik(x, θ, b)
    llik = zero(eltype(θ))
    for i in eachindex(x)
        k, vals = b(x[i])
        llik += log(θ[k] * vals[4] + θ[k-1] * vals[3] + θ[k-2] * vals[2] + θ[k-3] * vals[1])
    end
    return llik
end

# Evaluate spline basis functions prior to running the mdoel.
@model function Bsplinemix(x, μ_new, b, K, φ)
    τ1 ~ InverseGamma(1, 1e-1)
    #τ1 ~ InverseGamma(2.5, 1.5)
    #τ2 ~ InverseGamma(1, 1e-2)
    τ2 ~ InverseGamma(1, 5e-3)
    δ ~ filldist(InverseGamma(a_δ, b_δ), K-2)
    #β ~ arraydist([Normal(μ_new[k], sqrt(τ2 * δ[k])) for k in 1:K-1])
    β = Vector{Float64}(undef, K-1)
    β[1] ~ Normal(μ_new[1], sqrt(τ1))
    for k in 2:K-1
        β[k] ~ Normal(μ_new[k] + φ * (β[k-1] - μ_new[k-1]), sqrt(τ2 * δ[k-1]))
    end
    θ := stickbreaking(β)
    Turing.@addlogprob! myloglik(x, θ, b)
end

mix = BetaMixture(
    [0.4, 0.6],  # mixture weights
    [Beta(50, 120), Beta(3, 1.5)]
)
ground_truth = Beta(3,3)
#ground_truth = mix
x = rand(rng, ground_truth, 500)
model = Bsplinemix(x, μ_new, basis, K, 0.95)
chn = sample(rng, model, NUTS(), 1000)

t = LinRange(0, 1, 10001)
θ = mean(chn).nt.mean[100:end]
S = Spline(basis, theta_to_coef(θ))
p = Plots.plot()
Plots.plot!(p, t, S.(t), color=:black, lwd=2.0)
Plots.ylims!(p, -0.05*maximum(S.(t)), 1.05*maximum(S.(t)))

histogram!(p, x, normalize=:pdf, bins=50, alpha=0.5)

#kdest = PosteriorStats.kde_reflected(x, bounds=(0,1))
#plot!(kdest.x, kdest.density, color=:red, lwd=2.0)
plot!(p, t, pdf(ground_truth, t))
#ylims!(p, 0.0, 5.5)

#t0 = LinRange(0, 10, 10001)
#plot(t0, pdf(InverseGamma(a_δ, b_δ), t0), xticks=0:1:10)




using Distributions, StatsPlots

# Define a custom Beta mixture distribution type
struct BetaMixture <: ContinuousUnivariateDistribution
    weights::Vector{Float64}
    components::Vector{Beta}
end

# Constructor with normalization of weights
function BetaMixture(weights::Vector{Float64}, comps::Vector{Beta})
    w = weights ./ sum(weights)
    return BetaMixture(w, comps)
end

# PDF
function Distributions.pdf(d::BetaMixture, x::Real)
    sum(w * pdf(c, x) for (w, c) in zip(d.weights, d.components))
end

# Sampling
function Base.rand(rng::AbstractRNG, d::BetaMixture, n::Int=1)
    z = rand(Categorical(d.weights), n)
    xs = similar(rand(d.components[1], n))
    for i in 1:n
        xs[i] = rand(d.components[z[i]])
    end
    return xs
end

propertynames(chn[Symbol("τ1")])
chn[Symbol("τ1")].data

varmeans = Vector{Float64}(undef, K-2)
for k in 1:K-2
    varmeans[k] = mean(chn[Symbol("τ2")].data .* chn[Symbol("δ[$k]")].data)
end
plot(varmeans)

BandedMatrix(-1 => -φ / σ2, 0 => , 1 => -φ / σ2)

Q = Symmetric(BandedMatrix(-1 => [-0.5], 0 => [1.0, 1.0], 1 => [-0.5]))
MvNormalCanon([0.0, 0.0], Q)

# So we can actually do parameterize the normal this way.
# This is really nice.
