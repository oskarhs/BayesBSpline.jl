using Random, Statistics
using ForwardDiff, BandedMatrices
using LinearAlgebra, Distributions, BSplineKit, Plots, Optim
using PolyaGammaHybridSamplers

include(joinpath(@__DIR__, "BayesBSpline.jl"))
using .BayesBSpline

# Evaluate basis functions for all observations:
function create_spline_basis_matrix(x::AbstractVector{T}, K::Int) where {T<:Real}
    basis = BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2))
    n = length(x)
    b_ind = Vector{Int}(undef, n)
    B = Matrix{T}(undef, (n, 4))
    # Note: BSplineKit returns the evaluated spline functions in "reverse" order
    for i in eachindex(x)
        j, basis_eval = basis(x[i])
        b_ind[i] = j-3 # So we compute b_{j-3}, b_{j-2}, b_{j-1} and b_j for x_i
        B[i,:] .= reverse(basis_eval)
    end
    return B, b_ind
end

# Generate M samples from the posterior of β using Gibbs sampling
function sample_posterior(rng::Random.AbstractRNG, x::AbstractVector{T}, M::Int) where {T<:Real}
    K = 50
    n = length(x)

    # Prior Hyperparameters
    φ::T = 0.95
    a_σ::T = 2.5
    b_σ::T = 1.5 # b_σ = a_σ - 1
    a_τ::T = 1.0
    b_τ::T = 1e-2
    a_δ = fill(T(1.1), K-2)
    b_δ = fill(T(0.1), K-2)
    #a_δ::T = 1.1
    #b_δ::T = 0.1 # b_δ = a_δ - 1 # for online updating we will need a vector of these instead.

    β = Matrix{T}(undef, (K-1, M))
    β[:,1] = T.(BayesBSpline.μ_50)
    B, b_ind = create_spline_basis_matrix(x, K)
    μ = T.(BayesBSpline.μ_50)
    τ2 = one(T)                # Global smoothing parameter
    δ2 = Vector{T}(undef, K-2) # Local smoothing parameters
    z = Vector{Int}(undef, n)  # Class labels
    ω = Vector{T}(undef, K-1)  # PolyaGamma variables

    for m in 2:M
        # Update σ2:
        a_σ_new = a_σ + T(0.5)
        b_σ_new = b_σ + T(0.5) * abs2(β[1,m-1]- μ[1])
        σ2 = rand(rng, InverseGamma(a_σ_new, b_σ_new))

        # Update δ2: (some inefficiencies here, but okay for now)
        for k in 1:K-2
            a_δ_k_new = a_δ[k] + T(0.5)
            b_δ_k_new = b_δ[k] + T(0.5) * abs2( β[k+1,m-1] -  μ[k+1] - φ * (β[k,m-1] - μ[k])) / τ2
            δ2[k] = rand(rng, InverseGamma(a_δ_k_new, b_δ_k_new))
        end

        # Update τ2
        a_τ_new = a_τ + T(0.5) * (K - 2)
        b_τ_new = b_τ
        for k in 1:K-2
            b_τ_new += T(0.5) * abs2( β[k+1,m-1] -  μ[k+1] - φ * (β[k,m-1] - μ[k])) / δ2[k]
        end
        τ2 = rand(rng, InverseGamma(a_τ_new, b_τ_new))

        # Update z
        for i in 1:n
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            logprobs = Vector{T}(undef, 4)
            for l in 1:4
                k = k0 + l -1
                sumterm = sum(@. cosh(T(0.5)*β[1:k, m-1]) - T(0.5) * β[1:k, m-1] - log(2))
                logprobs[l] = log(B[i, l]) + β[k, m-1] + sumterm
            end
            probs = softmax(logprobs)
            z[i] = rand(rng, DiscreteNonParametric(k0:k0+3, probs))
        end
        # Update ω
        # Compute N and S
        N = countint(z)
        S = n .- cumsum(vcat(0, N[1:K-1]))
        for k in 1:K-1
            c_k_new = S[k]
            d_k_new = abs(β[k, m-1])
            ω[k] = rand(rng, PolyaGammaHybridSampler(c_k_new, d_k_new))
        end

        # Update β
        # Compute the Q matrix
        diag_Q = Vector{T}(undef, K-1)
        offdiag_Q = Vector{T}(undef, K-2)
        diag_Q[1] = 1/σ2 + φ^2/(τ2 * δ2[1])
        for k in 2:K-2
            diag_Q[k] = 1/(τ2 * δ2[k-1]) + φ^2/(τ2 * δ2[k])
            offdiag_Q[k-1] = -φ/(τ2*δ[k-1])
        end
        offdiag_Q[K-2] = -φ/(τ2*δ[K-2])
        diag_Q[K-1] = 1/(τ2 * δ2[K-2])
        Q = Symmetric(BandedMatrix(-1=>offdiag_Q, 0=>diag_Q, 1=>offdiag_Q))
        # Compute the Ω matrix (Note: Q + D retains the banded structure!)
        Ω = Diagonal(ω)
        inv_Σ_new = Ω + Q
        # Compute inv(Σ_new) * μ_new
        canon_mean_new = Q * μ + (N - S/2)
        # Sample β
        β[:, m] = rand(MvNormalCanon(canon_mean_new, inv_Σ_new))
    end
    return β
end

rng = Random.default_rng()
x = rand(rng, Beta(2,2), 1000)

M = 1000
β = sample_posterior(rng, x, M)