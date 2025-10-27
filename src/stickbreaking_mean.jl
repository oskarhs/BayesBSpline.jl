using Random, Statistics, BandedMatrices
using ForwardDiff
using LinearAlgebra, Distributions, BSplineKit, Plots, Optim, Integrals

# Numerically stable variant of log(1 + exp(x))
softplus(x::Real) = ifelse(x ≥ 0, x + log(1+exp(-x)), log(1+exp(x)))

# Numerically stable sigmoid
sigmoid(x::Real) = ifelse(x ≥ 0, 1/(1 + exp(-x)), exp(x)/(1 + exp(x)))

# Map an unconstrained K-1 dimensional vector to the K-simplex through stickbreaking
function stickbreaking(β::AbstractVector{T}) where {T<:AbstractFloat}
    K = length(β) + 1
    log_π = Vector{T}(undef, K-1)
    softplus_sum = zero(eltype(β))
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
    prob = solve(IntegralProblem((x, p) -> pdf(Normal(μ_k, sqrt(τ2/(1-φ^2))), x) * sigmoid(x), (-Inf, Inf)), QuadGKJL()).u
    return (p0[k] - (1-p0_cum[k])*prob)^2
end


τ2 = 1
μ = zeros(K-1)
p0_cum = vcat(0, cumsum(p0))
for k in 1:K-1
    lossfunc = μ_k -> loss(μ_k, p0, p0_cum, k, τ2)
    res = Optim.optimize(lossfunc, -20, 20, GoldenSection())
    μ[k] = Optim.minimizer(res)
end

# For now, fix variances to be equal.

M = 10^5
means = zeros(K)
m1 = 0
m_β = 0
for _ in 1:M
    #a = rand(rng, InverseGamma(0.5, 1))
    #ζ = rand(rng, InverseGamma(0.5, 1/a))
    ζ = τ2/(1-φ^2)
    #β = rand(rng, MvNormal(μ, Diagonal(fill(ζ, K-1))))
    β = rand(rng, MvNormal(μ, Diagonal(fill(ζ, K-1))))
    #β = rand(rng, dist)
    means += stickbreaking(β)
end
means = means / M

B = 10^5

draws_new = Array{Float64}(undef, B, K)
for b in 1:B
    #a = rand(InverseGamma(0.5, 1))
    #ζ = rand(rng, InverseGamma(0.5, 1/a))
    ζ = τ2/(1-φ^2)
    β = rand(rng, MvNormal(μ, Diagonal(fill(ζ, K-1))))
    #β = rand(rng, dist)
    draws_new[b, :] = stickbreaking(β)
end
q05 = [quantile(draws_new[:, k], 0.05) for k in 1:K]
q95 = [quantile(draws_new[:, k], 0.95) for k in 1:K]
q50 = [quantile(draws_new[:, k], 0.5) for k in 1:K]
priormean = vec(mean(draws_new, dims=1))
t = LinRange(0, 1, 10001)
b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, K-2))
p = Plots.plot()
for θ in [priormean, q05, q50, q95]
    S = Spline(b, theta_to_coef(θ, K))
    Plots.plot!(p, t, S.(t), label="")
    Plots.ylims!(p, -0.05*maximum(S.(t)), 1.05*maximum(S.(t)))
end
p