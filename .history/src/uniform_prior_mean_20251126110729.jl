"""
    find_uniform_prior_mean_β([rng::Random.AbstractRNG], K::Int=50; a_σ::T = 2.5, b_σ::T = 1.5, a_τ::T = 1, b_τ::T = 1e-2, a_δ::T = 1, b_δ::T = 0.1) where {T<:AbstractFloat}

Numerically determine the vector `μ` such that the BSplineMix prior has an approximately uniform prior mean.

# Arguments
* `K`: Dimension of the uniformly spaced spline basis.

# Keyword arguments
* `a_σ, b_σ`: Hyperparameters of the InverseGamma prior on σ²
* `a_τ, b_τ`: Hyperparameters of the InverseGamma prior on τ²
* `a_δ, b_δ`: Hyperparameters of the InverseGamma prior on each δₖ²

# Returns
* `μ`: The prior mean of β.
"""
function find_uniform_prior_mean_β(rng::Random.AbstractRNG, K::Int; a_σ::T = 2.5, b_σ::T = 1.5, a_τ::T = 1.0, b_τ::T = 1e-2, a_δ::T = 1.1, b_δ::T = 0.1, φ::T=1.0) where {T<:AbstractFloat}
    M = 50_000

    # Note to self: when we sample from these we automatically get samples of type T!
    dist_σ2 = InverseGamma(a_σ, b_σ)
    dist_τ2 = InverseGamma(a_τ, b_τ) # (1, 5e-3) yields quite symmetric prior
    dist_δ2 = InverseGamma(a_δ, b_δ) # (1, 5e-1) yields quite symmetric prior

    # Compute the prior mean of θ that yields a uniform-centered spline:
    p0 = coef_to_theta(ones(T, K))

    # Generate random samples used for optimization:
    z = rand(rng, Normal(T(0.0), T(1.0)), (M, K-1)) # Actually only have to fill the lower triangle here, but this is not a computational bottleneck anyway.
    σ = sqrt.(rand(rng, dist_σ2, M))
    τ = sqrt.(rand(rng, dist_τ2, M))
    δ = sqrt.(rand(rng, dist_δ2, (M, K-2)))

    # Store intermediate quantities
    θ_cum = Matrix{T}(undef, (M, K))
    β = Matrix{T}(undef, (M, K-1))

    # Find the values of μ s.t. the prior mean is approximately uniform using optimization:
    μ = zeros(K-1)
    lossfunc = μ_1 -> loss1(μ_1, p0, z, σ)
    res = Optim.optimize(lossfunc, -40, 20, GoldenSection())
    μ[1] = Optim.minimizer(res)
    for m in 1:M
        β[m,1] = μ[1] + σ[m] * z[m,1]
        θ_cum[m,1] = sigmoid(β[m,1])
    end
    for k in 2:K-1
        lossfunc = μ_k -> loss2(μ_k, μ[1:k-1], p0, k, z, τ, δ, φ, θ_cum, β)
        res = Optim.optimize(lossfunc, -40, 20, GoldenSection())
        μ[k] = Optim.minimizer(res)
        for m in 1:M
            β[m,k] = μ[k] + φ * (β[m,k-1] - μ[k-1]) + τ[m] * δ[m,k-1] * z[m,k]
            θ_cum[m, k] = θ_cum[m, k-1] + (1.0-θ_cum[m, k-1]) * sigmoid(β[m,k])
        end
    end
    return μ
end

function find_uniform_prior_mean_β(K::Int; kwargs...)
    return find_uniform_prior_mean_β(Random.default_rng(), K; kwargs...)
end

function loss1(μ_k::T, p0::AbstractVector{T}, z::AbstractMatrix{T}, σ::AbstractVector{T}) where {T<:AbstractFloat}
    θ_1 = Vector{T}(undef, size(z, 1))
    for m in axes(z, 1)
        β_1 = μ_k + σ[m] * z[m,1]
        #β = vcat(β, zeros(T, length(p0)-2)) # Just augment with zeros, the values of the remaining β's do not matter
        θ_1[m] = sigmoid(β_1)
    end
    return (p0[1] - mean(θ_1))^2
end
# After computing previous μ, the values of β[j] for j < k do not change from iteration to iteration... Not of critical importance for now, but potentially large speedups to be had here
# Best way would be to store a vector with βs from 1:k-1
function loss2(μ_k::T, μ::AbstractVector{T}, p0::AbstractVector{T}, k::Int, z::AbstractMatrix{T}, τ::AbstractVector{T}, δ::AbstractMatrix{T}, φ::T, θ_cum::AbstractMatrix{T}, β::AbstractMatrix{T}) where {T<:AbstractFloat}
    θ_k = Vector{T}(undef, size(z, 1))
    for m in axes(z, 1)
        #β = Vector{T}(undef, k)
        #β[1] = μ[1] + σ[m] * z[m,1]
        #for j in 2:k-1
        #    β[j] = μ[j] + φ * (β[j-1] - μ[j-1]) + τ[m] * δ[m,j-1] * z[m,j]
        #end
        β_k = μ_k + φ * (β[m,k-1] - μ[k-1]) + τ[m] * δ[m,k-1] * z[m,k]
        θ_k[m] = (1.0 - θ_cum[m, k-1]) * sigmoid(β_k)
    end
    return (p0[k] - mean(θ_k))^2
end