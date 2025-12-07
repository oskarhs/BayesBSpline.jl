#=     a_σ = 2.5
    b_σ = 1.5 # b_σ = a_σ - 1
    a_τ = 1
    b_τ = 1e-2
    a_δ = 1.1
    b_δ = 0.1 # b_δ = a_δ - 1 =#

"""
    find_uniform_prior_mean_β([rng::Random.AbstractRNG], K::Int=50; a_σ::T = 2.5, b_σ::T = 1.5, a_τ::T = 1, b_τ::T = 1e-2, a_δ::T = 1, b_δ::T = 0.1) where {T<:AbstractFloat}

Numerically determine the vector `μ` such that the BSplineMix prior has an approximately uniform prior mean.

# Arguments
* `K`: Dimension of spline basis.

# Keyword arguments
* ``:

# Returns
* `μ`: The prior mean of β.
"""
function find_uniform_prior_mean_β(rng::Random.AbstractRNG, K::Int=50; a_σ::T = 2.5, b_σ::T = 1.5, a_τ::T = 1.0, b_τ::T = 1e-2, a_δ::T = 1.1, b_δ::T = 0.1) where {T<:AbstractFloat}
    φ = 0.95
    M = 10_000

    # Note to self: when we sample from these we automatically get samples of type T!
    dist_σ2 = InverseGamma(a_σ, b_σ)
    dist_τ2 = InverseGamma(a_τ, b_τ) # (1, 5e-3) yields quite symmetric prior
    dist_δ2 = InverseGamma(a_δ, b_δ) # (1, 5e-1) yields quite symmetric prior

    # Compute the prior mean of θ that yields a uniform-centered spline:
    p0 = coef_to_theta(ones(T, K))

    # Generate random samples used for optimization:
    z = rand(rng, Normal(zero(T), one(T)), (M, K-1)) # Actually only have to fill the lower triangle here, but this is not a computational bottleneck anyway.
    σ = sqrt.(rand(rng, dist_σ2, M))
    τ = sqrt.(rand(rng, dist_τ2, M))
    δ = sqrt.(rand(rng, dist_δ2, (M, K-2)))

    # Find the values of μ s.t. the prior mean is approximately uniform using optimization:
    μ = zeros(K-1)
    lossfunc = μ_1 -> loss1(μ_1, p0, z, σ)
    res = Optim.optimize(lossfunc, -40, 20, GoldenSection())
    μ[1] = Optim.minimizer(res)
    for k in 2:K-1
        println(k)
        lossfunc = μ_k -> loss2(μ_k, μ[1:k-1], p0, k, z, σ, τ, δ, φ)
        res = Optim.optimize(lossfunc, -40, 20, GoldenSection())
        μ[k] = Optim.minimizer(res)
    end
    return μ
end

function find_uniform_prior_mean_β(K::Int=50; kwargs...)
    return find_uniform_prior_mean_β(Random.default_rng(), K; kwargs...)
end

function loss1(μ_k::T, p0::AbstractVector{T}, z::AbstractMatrix{T}, σ::AbstractVector{T}) where {T<:AbstractFloat}
    θ_1 = Vector{T}(undef, size(z, 1))
    for m in axes(z, 1)
        β = μ_k + σ[m] * z[m,1]
        β = vcat(ϵ_k, zeros(T, length(p0)-2)) # Just augment with zeros, the values of the remaining β's do not matter
        θ_1 = BayesBSpline.stickbreaking(β)[1]
    end
    return (p0[1] - mean(θ_1))^2
end
# After computing previous μ, the values of β[j] for j < k do not change from iteration to iteration... Not of critical importance for now, but potentially large speedups to be had here
# Best way would be to store a vector with βs from 1:k-1
function loss2(μ_k::T, μ::AbstractVector{T}, p0::AbstractVector{T}, k::Int, z::AbstractMatrix{T}, σ::AbstractVector{T}, τ::AbstractVector{T}, δ::AbstractMatrix{T}, φ::T) where {T<:AbstractFloat}
    θ_k = Vector{T}(undef, size(z, 1))
    for m in axes(z, 1)
        β = Vector{T}(undef, k)
        β[1] = μ[1] + σ[m] * z[m,1]
        for j in 2:k-1
            β[j] = μ[j] + φ * (β[j-1] - μ[j-1]) + τ[m] * δ[m,j-1] * z[m,j]
        end
        β[k] = μ_k + φ * (β[k-1] - μ[k-1]) + τ[m] * δ[m,k-1] * z[m,k]
        β = vcat(β, zeros(T, length(p0)-1-k)) # Just augment with zeros, the values of the remaining β's do not matter
        θ_k[m] = stickbreaking(β)[k]
    end
    return (p0[k] - mean(θ_k))^2
end


#= K = 50
rng = Random.default_rng()
ind = 1

p0 = coef_to_theta(ones(K))

# Use for k ≥ 2
# Also: reuse the same random sample


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
#dist_τ1 = InverseGamma(1, 1e-1) # (1, 1e-1) is reasonable, but would like a bit more variance in the first component
dist_τ1 = InverseGamma(2.5, 2.5-1)
#dist_τ2 = InverseGamma(1, 5e-3) # (1, 5e-3) yields quite symmetric prior
dist_τ2 = InverseGamma(1, 1e-2) # (1, 5e-3) yields quite symmetric prior
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
end =#