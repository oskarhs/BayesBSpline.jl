const μ_50 = [-5.7435005016854594, -4.961289565352378, -4.467471755810539, -4.098204485962134, -4.036330910975099, -3.9783841533892934, -3.920499410995347, -3.86694465395476, -3.806677996816535, -3.7644242932958956, -3.7156326173760097, -3.6800648665416267, -3.632148206098766, -3.592682192740396, -3.552454811711799, -3.512963089506606, -3.471855663765868, -3.4278656823141564, -3.384220810383284, -3.342237358589844, -3.3058469350056243, -3.2602241029060637, -3.215263930934274, -3.1744765777241963, -3.1171048181263474, -3.074572501452939, -3.023885722221279, -2.9716619392414656, -2.91943114413079, -2.8604866753666047, -2.7976790541511827, -2.735342217017036, -2.668376702691905, -2.596671170371428, -2.5200074015612453, -2.4373894676697736, -2.3493788668388866, -2.2575056852589324, -2.1449533683591158, -2.027008145955328, -1.8901681280226776, -1.7351653643739535, -1.5620607886955458, -1.3394857877193818, -1.0610942268371077, -0.6819485256545953, -0.10262940721266187, 0.43039551540783294, 1.4439482920828717]


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
* `K`: Dimension of the uniformly spaced spline basis.

# Keyword arguments
* `a_σ, b_σ`: Hyperparameters of the InverseGamma prior on σ²
* `a_τ, b_τ`: Hyperparameters of the InverseGamma prior on τ²
* `a_δ, b_δ`: Hyperparameters of the InverseGamma prior on each δₖ²

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
    z = rand(rng, Normal(T(0.0), T(1.0)), (M, K-1)) # Actually only have to fill the lower triangle here, but this is not a computational bottleneck anyway.
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
        β = vcat(β, zeros(T, length(p0)-2)) # Just augment with zeros, the values of the remaining β's do not matter
        θ_1[m] = BayesBSpline.stickbreaking(β)[1]
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