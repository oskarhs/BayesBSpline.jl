using Random, Statistics
using ForwardDiff, BandedMatrices
using LinearAlgebra, Distributions, BSplineKit, Plots, Optim
using PolyaGammaHybridSamplers, Memoization, PosteriorStats

include(joinpath(@__DIR__, "BayesBSpline.jl"))
using .BayesBSpline

# Evaluate basis functions for all observations:
# Remember to normalize:
function create_spline_basis_matrix_binned(x::AbstractVector{T}, K::Int) where {T<:Real}
    n_bins = 50*(K-2)-1 # Make the number of bins a multiple of K-2 so that at most 4 basis functions are nonzero at a time
    bincounts = BayesBSpline.bin_regular(x, 0.0, 1.0, n_bins, true)
    binedges = LinRange(0, 1, n_bins+1)
    basis = BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2))
    n = length(x)
    b_ind = Vector{Int}(undef, n_bins)
    B = Matrix{T}(undef, (n_bins, 4))
    norm_fac = BayesBSpline.compute_norm_fac(K)[1:K, end]
    
    # Compute ∫ bⱼ(x) dx over each bin for the nonzero coefficients
    #integral(Spline(basis, unitvector(K, 1)))

    # Note: BSplineKit returns the evaluated spline functions in "reverse" order
    for i in 1:n_bins
        x0 = binedges[i]
        x1 = binedges[i+1]
        j = find_knot_interval(LinRange(0, 1, K-2), x0)[1] # So we compute b_{j-3}, b_{j-2}, b_{j-1} and b_j for x_i
        b_ind[i] = j
        for l in 1:4
            k = j + l - 1
            S = integral(Spline(basis, BayesBSpline.unitvector(K, k, T)))
            B[i,l] = (S(x1) - S(x0)) * norm_fac[k]
        end
    end
    return B, b_ind, bincounts, binedges
end


# Generate M samples from the posterior of β using Gibbs sampling
function sample_posterior_binned(rng::Random.AbstractRNG, x::AbstractVector{T}, M::Int, K::Int) where {T<:Real}
    B, b_ind, bincounts, binedges = create_spline_basis_matrix_binned(x, K)
    bincounts = Int64.(bincounts)
    n_bins = length(bincounts)
    n = length(x)

    # Prior Hyperparameters
    # Can choose a_σ, b_σ differently based on whether or not we 
    #φ::T = 0.95
    φ::T = 1.0
    a_σ::T = 1.0
    b_σ::T = 5e-1
    #a_σ::T = 3.5
    #b_σ::T = 2.5
    a_τ::T = 1.1
    b_τ::T = 1e-4
    a_δ::T = 1.5
    b_δ::T = 0.5
    #a_δ::T = 1.0
    #b_δ::T = 5e-4
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
    β[:,1] = T.(μ)
    log_B = log.(B)
    #μ = T.()
    #= μ = T.(
        [-5.55086254405145, -4.68879389619881, -4.213207497333472, -3.8740867418904514, -3.832643292779035, -3.7957373408317245, -3.7639319282126364, -3.7323973153796524, -3.699945741531716, -3.6684282139121667, -3.6373666977443104, -3.6059871920960167, -3.575024083007192, -3.542285133963017, -3.5102030055447724, -3.4785546211072504, -3.444080643272235, -3.412542374142108, -3.3767515230192537, -3.340180969872507, -3.304118570595692, -3.2661934794904393, -3.226566805296623, -3.1854399727276728, -3.14326772948271, -3.0990484076227847, -3.053720949782737, -3.0073251036834554, -2.9536841812387657, -2.9008581564328546, -2.846240541140525, -2.7854346915870027, -2.7223021779456795, -2.654889614521274, -2.5836344634905246, -2.507210455921083, -2.42293202692572, -2.3315303305083668, -2.2293625540179747, -2.116982629825232, -1.9909192748514934, -1.8474115620750406, -1.6793831752036616, -1.476015806203944, -1.223691804969293, -0.8821477373029734, -0.36288656233481276, 0.054657988126208874, 0.7803886040437757]
    ) =#
    #μ = T.(BayesBSpline.μ_50)
    τ2 = one(T)                # Global smoothing parameter
    δ2 = Vector{T}(undef, K-2) # Local smoothing parameters
    ω = Vector{T}(undef, K-1)  # PolyaGamma variables

    θ = Matrix{T}(undef, (K, M)) # Mixture probabilities
    θ[:, 1] = BayesBSpline.stickbreaking(β[:, 1])
    log_θ = similar(θ)
    log_θ[:, 1] = log.(θ[:,1])

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
            #δ2[k] = 1.0
        end

        # Update τ2
        a_τ_new = a_τ + T(0.5) * (K - 2)
        b_τ_new = b_τ
        for k in 1:K-2
            b_τ_new += T(0.5) * abs2( β[k+1,m-1] -  μ[k+1] - φ * (β[k,m-1] - μ[k])) / δ2[k]
        end
        τ2 = rand(rng, InverseGamma(a_τ_new, b_τ_new))

        # Update z
        N = zeros(Int, K)
        logprobs = Vector{T}(undef, 4)
        for i in 1:n_bins
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            #logprobs = Vector{T}(undef, 4)
            for l in 1:4
                k = k0 + l - 1
                #= if k != K
                    sumterm = sum(@. -log(cosh(T(0.5)*β[1:k, m-1])) - T(0.5) * β[1:k, m-1] - log(T(2)))
                    logprobs[l] = log(B[i, l]) + β[k, m-1] + sumterm
                else
                    sumterm = sum(@. -log(cosh(T(0.5)*β[1:K-1, m-1])) - T(0.5) * β[1:K-1, m-1] - log(T(2)))
                    logprobs[l] = log(B[i, l]) + sumterm
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
        θ[:, m] = max.(eps(), BayesBSpline.stickbreaking(β[:, m]))
        θ[:, m] = θ[:, m] / sum(θ[:, m])
        log_θ[:, m] = log.(θ[:,m])
    end
    return θ
end

K = 100
rng = Random.default_rng()
#d_true = Beta(3,3)
#d_true = mixture_dist
d_true = Normal()
#d_true = Claw()
#d_true = MixtureModel([Normal(-2, 1), Normal(2, 0.1)], [0.4, 0.6])
x = rand(rng, d_true, 250)
R = maximum(x) - minimum(x)
x_est = (x .- minimum(x) .+ 0.05*R) / (1.1*R)
#x = LinRange(0, 1, 1000)

# Run the Gibbs sampler
M = 5000
θ = sample_posterior_binned(rng, x_est, M, K)
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
Plots.plot!(p, t_orig, pdf(d_true, t_orig), color=:red, label="True", lw=1.2, alpha=0.5, ls=:dash)
#histogram!(p, x, bins=50, normalize=:pdf, alpha=0.5)
p
