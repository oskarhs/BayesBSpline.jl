using Random, Statistics
using ForwardDiff, BandedMatrices
using LinearAlgebra, Distributions, BSplineKit, Plots, Optim
using PolyaGammaHybridSamplers, Memoization, PosteriorStats, KernelDensity, BridgeSampling

include(joinpath(@__DIR__, "BayesBSpline.jl"))
using .BayesBSpline

# Evaluate basis functions for all observations:
# Remember to normalize:
function create_spline_basis_matrix_binned(x::AbstractVector{T}, basis::A) where {T<:Real, A<:AbstractBSplineBasis}
    # A total of 1000 bins should suffice
    K = length(basis)

    n_bins = (fld(1000, K-2)+1)*(K-2)-1 # Make the number of bins a multiple of K-2 so that at most 4 basis functions are nonzero at a time
    bincounts = BayesBSpline.bin_regular(x, 0.0, 1.0, n_bins, true)
    binedges = LinRange(0, 1, n_bins+1)
    # basis = BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2))
    n = length(x)
    b_ind = Vector{Int}(undef, n_bins)
    B = Matrix{T}(undef, (n_bins, 4))
    norm_fac = BayesBSpline.compute_norm_fac(basis, T)
    
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
    basis = BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2))
    B, b_ind, bincounts, binedges = create_spline_basis_matrix_binned(x, basis)
    bincounts = Int64.(bincounts)
    n_bins = length(bincounts)
    n = length(x)

    # Prior Hyperparameters
    # Can choose a_σ, b_σ differently based on whether or not we 
    φ::T = 1.0
    #φ::T = 1.0
    a_τ::T = 1.0
    b_τ::T = 1e-3
    #a_δ::T = 1.1
    #b_δ::T = 0.1
    a_δ::T = 0.5
    b_δ::T = 0.5
    #a_δ::T = 1.0
    #b_δ::T = 5e-4
    kwargs = Dict(
        :a_τ => a_τ,
        :b_τ => b_τ,
        :a_δ => a_δ,
        :b_δ => b_δ,
        :φ   => φ
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
            b_δ_k_new = b_δ + T(0.5) * abs2( β[k+2,m-1] -  μ[k+2] - φ *( 2*(β[k+1,m-1] - μ[k+1]) - (β[k,m-1] - μ[k]) )) / τ2
            δ2[k] = rand(rng, InverseGamma(a_δ_k_new, b_δ_k_new))
            #δ2[k] = 1.0
        end

        # Update τ2
        a_τ_new = a_τ + T(0.5) * (K - 3)
        b_τ_new = b_τ
        for k in 1:K-3
            b_τ_new += T(0.5) * abs2( β[k+2,m-1] -  μ[k+2] - φ *( 2*(β[k+1,m-1] - μ[k+1]) - (β[k,m-1] - μ[k]) )) / δ2[k]
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
    return θ, β, τ2s, δ2s, kwargs
end

function samples_as_matrix(β::AbstractMatrix{T}, σ2s::AbstractVector{T}, τ2s::AbstractVector{T}, δ2s::AbstractMatrix{T}) where {T<:Real}
    M = size(β, 2)
    K = size(β, 1) + 1
    samples = Matrix{T}(undef, (2*K-1, M))
    samples[1:K-1, 1:M] = β
    samples[K, 1:M] = σ2s
    samples[K+1, 1:M] = τ2s
    samples[K+2:end, 1:M] = δ2s
    return samples
end

function logposterior(η::AbstractVector{T}, b_ind::AbstractVector{Int}, B::AbstractMatrix{T}, K::Int; kwargs...) where {T<:Real}
    # Unpack parameters
    β = η[1:K-1]
    σ2 = η[K]
    τ2 = η[K+1]
    δ2 = η[K+2:end]

    # Unpack Hyperparameters
    a_σ = kwargs[:a_σ]
    b_σ = kwargs[:b_σ]
    a_τ = kwargs[:a_τ]
    b_τ = kwargs[:b_τ]
    a_δ = kwargs[:a_δ]
    b_δ = kwargs[:b_δ]
    φ   = kwargs[:φ]

    μ = BayesBSpline.find_uniform_prior_mean_β(rng, K; kwargs...)
    log_B = log.(B)

    logpost = zero(T)

    # Likelihood contribution: p(x|β)
    θ = max.(eps(), BayesBSpline.stickbreaking(β))
    θ = θ / sum(θ)
    log_θ = log.(θ)
    for i in eachindex(x)
        j = b_ind[i]
        logpost += sum(log_θ[j:j+3] .+ log_B[i,1:4])
    end

    # Prior on β, p(β|σ2, τ2, δ2)
    diag_Q = Vector{T}(undef, K-1)      # Diagonal of prior precision Q
    offdiag_Q = Vector{T}(undef, K-2)   # Off-diagonal of prior precision Q
    diag_Q[1] = 1/σ2 + φ^2/(τ2 * δ2[1])
    for k in 2:K-2
        diag_Q[k] = 1/(τ2 * δ2[k-1]) + φ^2/(τ2 * δ2[k])
        offdiag_Q[k-1] = -φ/(τ2*δ2[k-1])
    end
    offdiag_Q[K-2] = -φ/(τ2*δ2[K-2])
    diag_Q[K-1] = 1/(τ2 * δ2[K-2])
    Q = Symmetric(BandedMatrix(-1=>offdiag_Q, 0=>diag_Q, 1=>offdiag_Q))
    logpost += logpdf(MvNormalCanon(Q*μ, Q), β)

    # Prior on σ2
    logpost += logpdf(InverseGamma(a_σ, b_σ), σ2)

    # Prior on τ2
    logpost += logpdf(InverseGamma(a_τ, b_τ), τ2)

    # Prior on δ2
    logpost += sum(logpdf(InverseGamma(a_δ, b_δ), δ2))

    return logpost
end

function pointwise_quantiles(θ::AbstractMatrix{T}, basis::A, t, q) where {T<:Real, A<:AbstractBSplineBasis}
    S_samp = Matrix{T}(undef, (length(t), size(θ, 2)))
    for i in axes(θ, 2)
        S_samp[:,i] = Spline(basis, BayesBSpline.theta_to_coef(θ[:,i], basis)).(t)
    end
    return [quantile(S_samp[j,:], q) for j in eachindex(t)]
end


K = 200
rng = Random.default_rng()
#d_true = Claw()
#d_true = StronglySkewed()
#d_true = MixtureModel([Normal(0, 0.5), Normal(2, 0.1)], [0.4, 0.6])
#d_true = Normal()
d_true = Harp()
#d_true = Kurtotic()
x = rand(rng, d_true, 1500)
R = maximum(x) - minimum(x)
x_est = (x .- minimum(x) .+ 0.05*R) / (1.1*R)
#x = LinRange(0, 1, 1000)

# Run the Gibbs sampler
M = 5_000
θ, β, τ2s, δ2s, kwargs = sample_posterior_binned(rng, x_est, M, K)
θ_mean = [mean(θ[k,1001:end]) for k in 1:K]
basis = BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2))


t = LinRange(0, 1, 10001)
t_orig = minimum(x) .+ R*t
kdest = kde(x; bandwidth=PosteriorStats.isj_bandwidth(x))
#kdest = PosteriorStats.kde_reflected(x)

lower = pointwise_quantiles(θ, basis, t, 0.05) / (1.1*R)
upper = pointwise_quantiles(θ, basis, t, 0.95) / (1.1*R)

S = Spline(basis, BayesBSpline.theta_to_coef(θ_mean, basis))
p = Plots.plot()
Plots.plot!(p, minimum(x) .- 0.05*R .+ 1.1*R*t, S.(t) / (1.1*R), color=:black, lw=1.2, label="Gibbs")
Plots.plot!(p, minimum(x) .- 0.05*R .+ 1.1*R*t, lower, color=:green, ls=:dash, label="90% CI", alpha=0.5)
Plots.plot!(p, minimum(x) .- 0.05*R .+ 1.1*R*t, upper, color=:green, label="", ls=:dash, alpha=0.5)
Plots.plot!(p, minimum(x) .- 0.05*R .+ 1.1*R*t, upper, fillrange=lower, fillcolor=:green, fillalpha=0.2, label="", color=:transparent)

Plots.plot!(kdest.x, kdest.density, color=:blue, label="KDE", lw=1.2)
Plots.plot!(p, t_orig, pdf(d_true, t_orig), color=:red, label="True", lw=1.2, alpha=0.5)
if typeof(d_true) <: MixtureModel
    xlims!(p, -1.8, 2.35)
elseif typeof(d_true) <: Claw
    xlims!(p, -3, 3)
elseif typeof(d_true) <: StronglySkewed
    xlims!(p, -3.5, 3)
elseif typeof(d_true) <: Harp
    xlims!(p, -2, 80)
elseif typeof(d_true) <: Kurtotic
    xlims!(p, -3.2, 3.2)
end
p

plot(τ2s[1001:end])
plot([mean(δ2s[k, 1001:M]) for k in 1:K-3])
#plot(δ2s[1001:M, 1])

basis = BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2))
gm = galerkin_matrix(BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2)), (Derivative(2), Derivative(2)))