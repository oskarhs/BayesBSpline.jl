using Random, Statistics
using ForwardDiff, BandedMatrices
using LinearAlgebra, Distributions, BSplineKit, Plots, Optim
using PolyaGammaHybridSamplers, Memoization, PosteriorStats

include(joinpath(@__DIR__, "BayesBSpline.jl"))
using .BayesBSpline

# Evaluate basis functions for all observations:
# Remember to normalize:
function create_spline_basis_matrix(x::AbstractVector{T}, K::Int) where {T<:Real}
    basis = BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2))
    n = length(x)
    b_ind = Vector{Int}(undef, n)
    B = Matrix{T}(undef, (n, 4))
    norm_fac = BayesBSpline.compute_norm_fac(K)[1:K, end]
    # Note: BSplineKit returns the evaluated spline functions in "reverse" order
    for i in eachindex(x)
        j, basis_eval = basis(x[i])
        b_ind[i] = j-3 # So we compute b_{j-3}, b_{j-2}, b_{j-1} and b_j for x_i
        B[i,:] .= reverse(basis_eval) .* norm_fac[b_ind[i]:b_ind[i]+3]
    end
    return B, b_ind
end


function sample_posterior(rng::Random.AbstractRNG=Random.default_rng(), bsm::BSMModel{T, A, NT}, n_samples::Integer, n_adaps::Integer) where {T, A, NT}
    basis = BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2))
    #basis = PSplineBasis(BSplineOrder(4), K)
    B, b_ind, bincounts = create_spline_basis_matrix_binned(x, basis)
    bincounts = bincounts
    n_bins = length(bincounts)
    n = length(x)

    # Prior Hyperparameters
    # Can choose a_σ, b_σ differently based on whether or not we 
    a_τ::T = 1.0
    b_τ::T = 1e-3
    a_δ::T = 0.5
    b_δ::T = 0.5
    kwargs = Dict(
        :a_τ => a_τ,
        :b_τ => b_τ,
        :a_δ => a_δ,
        :b_δ => b_δ,
    )

    # Here: determine μ via the medians (e.g. we penalize differences away from the values that yield a uniform prior mean)
    μ = BayesBSpline.compute_μ(basis, T)

    # Set up penalty matrix:
    P = BandedMatrix((0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3)), (K-3, K-1))

    # Prior for β in this case is improper. Need difference matrix.
    
    β = Matrix{T}(undef, (K-1, M))
    β[:,1] = T.(μ)
    log_B = log.(B)
    τ2 = one(T)                # Global smoothing parameter
    δ2 = Vector{T}(undef, K-3) # Local smoothing parameters
    ω = Vector{T}(undef, K-1)  # PolyaGamma variables

    logprobs = Vector{T}(undef, 4)  # class label probabilities

    θ = Matrix{T}(undef, (K, M)) # Mixture probabilities
    θ[:, 1] = max.(eps(), BayesBSpline.stickbreaking(β[:, 1]))
    θ[:, 1] = θ[:, 1] / sum(θ[:, 1])
    log_θ = similar(θ)
    log_θ[:, 1] = log.(θ[:,1])
    τ2s = Vector{T}(undef, M)
    τ2s[1] = τ2
    δ2s = Matrix{T}(undef, (K-3, M))

    for m in 2:M

        # Update δ2: (some inefficiencies here, but okay for now)
        for k in 1:K-3
            a_δ_k_new = a_δ + T(0.5)
            b_δ_k_new = b_δ + T(0.5) * abs2( β[k+2,m-1] -  μ[k+2] - ( 2*(β[k+1,m-1] - μ[k+1]) - (β[k,m-1] - μ[k]) )) / τ2
            δ2[k] = rand(rng, InverseGamma(a_δ_k_new, b_δ_k_new))
            #δ2[k] = 1.0
        end

        # Update τ2
        a_τ_new = a_τ + T(0.5) * (K - 3)
        b_τ_new = b_τ
        for k in 1:K-3
            b_τ_new += T(0.5) * abs2( β[k+2,m-1] -  μ[k+2] - ( 2*(β[k+1,m-1] - μ[k+1]) - (β[k,m-1] - μ[k]) )) / δ2[k]
        end
        τ2 = rand(rng, InverseGamma(a_τ_new, b_τ_new))
        #τ2 = 0.01

        # Update z
        N = zeros(Int, K)               # class label counts (of z[i]'s)
        for i in 1:n_bins
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            #logprobs = Vector{T}(undef, 4)
            for l in 1:4
                k = k0 + l - 1
                #= if k != K
                    #sumterm = sum(@. -log(cosh(T(0.5)*β[1:k, m-1])) - T(0.5) * β[1:k, m-1] - log(T(2)))
                    sumterm = sum(@. -log(cosh(T(0.5)*β[k0+1:k, m-1])) - T(0.5) * β[k0+1:k, m-1] - log(T(2)))
                    logprobs[l] = log_B[i, l] + β[k, m-1] + sumterm
                else
                    sumterm = sum(@. -log(cosh(T(0.5)*β[k0+1:K-1, m-1])) - T(0.5) * β[k0+1:K-1, m-1] - log(T(2)))
                    logprobs[l] = log_B[i, l] + sumterm
                end =#
                logprobs[l] = log_B[i,l] + log_θ[k,m-1] 
            end
            probs = BayesBSpline.softmax(logprobs)
            counts = rand(rng, Multinomial(bincounts[i], probs))
            N[k0:k0+3] .+= counts
        end
        # Update ω
        # Compute N and S
        S = n .- cumsum(vcat(0, N[1:K-1]))
        for k in 1:K-1
            c_k_new = S[k]
            d_k_new = β[k, m-1]
            ω[k] = rand(rng, PolyaGammaHybridSampler(c_k_new, d_k_new))
        end

        # Update β
        # Compute the Q matrix
        D = Diagonal(1 ./(τ2*δ2))
        Q = transpose(P) * D * P
        # Compute the Ω matrix (Note: Q + D retains the banded structure!)
        Ω = Diagonal(ω)
        inv_Σ_new = Ω + Q
        # Compute inv(Σ_new) * μ_new
        canon_mean_new = Q * μ + (N[1:K-1] - S[1:K-1]/2)
        # Sample β
        β[:, m] = rand(rng, MvNormalCanon(canon_mean_new, inv_Σ_new))

        # Record θ
        θ[:, m] = max.(eps(), BayesBSpline.stickbreaking(β[:, m]))
        θ[:, m] = θ[:, m] / sum(θ[:, m])
        log_θ[:, m] = log.(θ[:,m])
        τ2s[m] = τ2
        δ2s[:,m] = δ2
    end

    return θ, β, τ2s, δ2s
end


# Generate M samples from the posterior of β using Gibbs sampling
function sample_posterior(rng::Random.AbstractRNG, x::AbstractVector{T}, M::Int) where {T<:Real}
    K = 50
    n = length(x)

    # Prior Hyperparameters
    φ::T = 0.95
    #= a_σ::T = 2.5
    b_σ::T = 1.5 # b_σ = a_σ - 1
    a_τ::T = 1.0
    b_τ::T = 1e-2
    a_δ = fill(T(1.1), K-2)
    b_δ = fill(T(0.1), K-2) =#
    #a_σ::T = 1.0
    #b_σ::T = 1e-1
    a_σ::T = 3.5
    b_σ::T = 2.5
    a_τ::T = 1.0
    b_τ::T = 5e-3
    a_δ::T = 1.0
    b_δ::T = 5e-4
    kwargs = Dict(
        :a_σ => a_σ,
        :b_σ => b_σ,
        :a_τ => a_τ,
        :b_τ => b_τ,
        :a_δ => a_δ,
        :b_δ => b_δ,
    )

    μ = @memoize BayesBSpline.find_uniform_prior_mean_β(rng, K; kwargs...)
    
    β = Matrix{T}(undef, (K-1, M))
    β[:,1] = T.(BayesBSpline.μ_50)
    B, b_ind = create_spline_basis_matrix(x, K)
    log_B = log.(B)
    τ2 = one(T)                # Global smoothing parameter
    δ2 = Vector{T}(undef, K-2) # Local smoothing parameters
    z = Vector{Int}(undef, n)  # Class labels
    ω = Vector{T}(undef, K-1)  # PolyaGamma variables

    θ = Matrix{T}(undef, (K, M)) # Mixture probabilities
    log_θ = similar(θ)
    θ[:, 1] = BayesBSpline.stickbreaking(β[:, 1])
    #log_θ[:,1] = log.(θ[:,1])

    for m in 2:M
        # Update σ2:
        a_σ_new = a_σ + T(0.5)
        b_σ_new = b_σ + T(0.5) * abs2(β[1,m-1]- μ[1])
        σ2 = rand(rng, InverseGamma(a_σ_new, b_σ_new))

        # Update δ2: (some inefficiencies here, but okay for now)
        for k in 1:K-2
            a_δ_k_new = a_δ + T(0.5)
            b_δ_k_new = b_δ + T(0.5) * abs2( β[k+1,m-1] -  μ[k+1] - φ * (β[k,m-1] - μ[k])) / τ2
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
                k = k0 + l - 1
                #= if k != K
                    sumterm = sum(@. -log(cosh(T(0.5)*β[1:k, m-1])) - T(0.5) * β[1:k, m-1] - log(T(2)))
                    logprobs[l] = log(B[i, l]) + β[k, m-1] + sumterm
                else
                    sumterm = sum(@. -log(cosh(T(0.5)*β[1:K-1, m-1])) - T(0.5) * β[1:K-1, m-1] - log(T(2)))
                    logprobs[l] = log(B[i, l]) + sumterm
                end =#
                logprobs[l] = log_B[i,l] + log(θ[k,m-1]) 
            end
            probs = BayesBSpline.softmax(logprobs)
            z[i] = rand(rng, DiscreteNonParametric(k0:k0+3, probs))
        end
        # Update ω
        # Compute N and S
        N = BayesBSpline.countint(z, K)
        S = n .- cumsum(vcat(0, N[1:K-1]))
        for k in 1:K-1
            c_k_new = S[k]
            d_k_new = β[k, m-1]
            ω[k] = rand(rng, PolyaGammaHybridSampler(c_k_new, d_k_new))
        end

        # Update β
        # Compute the Q matrix
        diag_Q = Vector{T}(undef, K-1)
        offdiag_Q = Vector{T}(undef, K-2)
        diag_Q[1] = 1/σ2 + φ^2/(τ2 * δ2[1])
        for k in 2:K-2
            diag_Q[k] = 1/(τ2 * δ2[k-1]) + φ^2/(τ2 * δ2[k])
            offdiag_Q[k-1] = -φ/(τ2*δ2[k-1])
        end
        offdiag_Q[K-2] = -φ/(τ2*δ2[K-2])
        diag_Q[K-1] = 1/(τ2 * δ2[K-2])
        Q = Symmetric(BandedMatrix(-1=>offdiag_Q, 0=>diag_Q, 1=>offdiag_Q))
        # Compute the Ω matrix (Note: Q + D retains the banded structure!)
        Ω = Diagonal(ω)
        inv_Σ_new = Ω + Q
        # Compute inv(Σ_new) * μ_new
        canon_mean_new = Q * μ + (N[1:K-1] - S[1:K-1]/2)
        # Sample β
        β[:, m] = rand(rng, MvNormalCanon(canon_mean_new, inv_Σ_new))

        # Record θ
        θ[:, m] = BayesBSpline.stickbreaking(β[:, m])
        #log_θ[:,m] = log.(θ[:,m])
    end
    return θ
end


#= 
K = 50
rng = Random.default_rng()
#d_true = Beta(3,3)
#d_true = mixture_dist
d_true = Claw()
#d_true = Normal()
x = rand(rng, d_true, 1000)
R = maximum(x) - minimum(x)
x_est = (x .- minimum(x) .+ 0.05*R) / (1.1*R)
#x = LinRange(0, 1, 1000)

# Run the Gibbs sampler
M = 5000
θ = sample_posterior(rng, x_est, M)
θ_mean = [mean(θ[k,1001:end]) for k in 1:K]
basis = BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2))

t = LinRange(0, 1, 10001)
t_orig = minimum(x) .+ R*t
kdest = kde(x; bandwidth=PosteriorStats.isj_bandwidth(x))
#kdest = PosteriorStats.kde_reflected(x)

S = Spline(basis, BayesBSpline.theta_to_coef(θ_mean))
p = Plots.plot()
Plots.plot!(p, minimum(x) .- 0.05*R .+ 1.1*R*t, S.(t) / (1.1*R), color=:black, lw=1.2, label="Gibbs")
#Plots.plot!(p, t, Spline(basis, BayesBSpline.theta_to_coef(θ_mean_NUTS)).(t), color=:blue, label="NUTS")
Plots.plot!(kdest.x, kdest.density, color=:blue, label="KDE", lw=1.2)
Plots.plot!(p, t_orig, pdf(d_true, t_orig), color=:red, label="True", lw=1.2)
#histogram!(p, x, bins=50, normalize=:pdf, alpha=0.5)
p
 =#