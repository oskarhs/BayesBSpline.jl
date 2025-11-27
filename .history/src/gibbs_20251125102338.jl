using Random, Statistics
using ForwardDiff, BandedMatrices
using LinearAlgebra, Distributions, BSplineKit, Plots, Optim
using PolyaGammaHybridSamplers

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
    a_σ::T = 1.0
    b_σ::T = 5e-3
    a_τ::T = 1.0
    b_τ::T = 5e-3
    a_δ::T = 1.0
    b_δ::T = 5e-3
    
    β = Matrix{T}(undef, (K-1, M))
    β[:,1] = T.(BayesBSpline.μ_50)
    B, b_ind = create_spline_basis_matrix(x, K)

    μ = T.(
        [-5.2643299112716075, -4.543074825930411, -4.118027364419802, -3.806069376460615, -3.7824714555959535, -3.757805060929923, -3.7341873708747544, -3.711756707865128, -3.684392792488028, -3.6589192752942785, -3.6302946623193577, -3.6019725122460233, -3.5733318865322152, -3.545161311144167, -3.5158040474296075, -3.483897405344335, -3.451351991135713, -3.419871093967889, -3.3850741975717957, -3.3515858366812368, -3.314952967879204, -3.2807498280858987, -3.238216138474518, -3.1974020894852044, -3.1559861227764836, -3.1119567030982327, -3.0663060644892863, -3.018285914864334, -2.968848764184586, -2.9159974165374116, -2.8601212868519585, -2.801385579038754, -2.73667209496, -2.668370786678853, -2.5959473053185222, -2.520274830901291, -2.435481722065169, -2.343382632421931, -2.2422700392266974, -2.1309656774773966, -2.0049618922829993, -1.8611600015193197, -1.6924739983676258, -1.4895583926265306, -1.2358380935920572, -0.8980345355435663, -0.38167749170694665, 0.03269763815343868, 0.7452442438378776]
    )
    #μ = T.(BayesBSpline.μ_50)
    τ2 = one(T)                # Global smoothing parameter
    δ2 = Vector{T}(undef, K-2) # Local smoothing parameters
    z = Vector{Int}(undef, n)  # Class labels
    ω = Vector{T}(undef, K-1)  # PolyaGamma variables

    θ = Matrix{T}(undef, (K, M)) # Mixture probabilities
    θ[:, 1] = BayesBSpline.stickbreaking(β[:, 1])

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
                logprobs[l] = log(B[i,l]) + log(θ[k,m-1]) 
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
    end
    return θ
end

K = 50
rng = Random.default_rng()
#d_true = Beta(10,8)
#d_true = Harp()
#d_true = Uniform()
d_true = mixture_dist
x = rand(rng, d_true, 20000)
R = maximum(x) - minimum(x)
x_est = (x .- minimum(x)) / (maximum(x) - minimum(x))
#x = LinRange(0, 1, 1000)

# Run the Gibbs sampler
M = 5000
θ = sample_posterior(rng, x_est, M)
θ_mean = [mean(θ[k,1001:end]) for k in 1:K]
basis = BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2))


# Compare with NUTS
#= function myloglik(x, θ, b)
    llik = zero(eltype(θ))
    for i in eachindex(x)
        k, vals = b(x[i])
        llik += log(θ[k] * vals[4] + θ[k-1] * vals[3] + θ[k-2] * vals[2] + θ[k-3] * vals[1])
    end
    return llik
end

@model function Bsplinemix(x, μ_new, b, K, φ)
    #τ1 ~ InverseGamma(1, 1e-1)
    τ1 ~ InverseGamma(2.5, 1.5)
    τ2 ~ InverseGamma(1, 1e-2)
    δ ~ filldist(InverseGamma(1.1, 0.1), K-2)
    #β ~ arraydist([Normal(μ_new[k], sqrt(τ2 * δ[k])) for k in 1:K-1])
    β = Vector{Float64}(undef, K-1)
    β[1] ~ Normal(μ_new[1], sqrt(τ1))
    for k in 2:K-1
        β[k] ~ Normal(μ_new[k] + φ * (β[k-1] - μ_new[k-1]), sqrt(τ2 * δ[k-1]))
    end
    θ := BayesBSpline.stickbreaking(β)
    Turing.@addlogprob! myloglik(x, θ, b)
end =#



#model = Bsplinemix(x, BayesBSpline.μ_50, basis, K, 0.95)
#chn = sample(rng, model, NUTS(), 1000)
#θ_mean_NUTS = mean(chn).nt.mean[100:end]


t = LinRange(0, 1, 10001)
t_orig = minimum(x) .+ R*t
kdest = kde(x; bandwidth=PosteriorStats.isj_bandwidth(x))
#kdest = PosteriorStats.kde_reflected(x)

S = Spline(basis, BayesBSpline.theta_to_coef(θ_mean))
p = Plots.plot()
Plots.plot!(p, minimum(x) .+ R*t, S.(t) / R, color=:black, label="Gibbs", lw=2.0)
#Plots.plot!(p, t, Spline(basis, BayesBSpline.theta_to_coef(θ_mean_NUTS)).(t), color=:blue, label="NUTS")
#Plots.plot!(kdest.x, kdest.density, color=:blue, label="KDE", lw=1.5)
Plots.plot!(p, t_orig, pdf(d_true, t_orig), color=:red, label="True", lw=1.5)
#histogram!(p, x, bins=50, normalize=:pdf, alpha=0.5)
p


# Sample from prior:
#chn = sample(rng, model, Prior(), 1000)
#θ_mean_NUTS = mean(chn).nt.mean[100:end]
#plot(t, Spline(basis, BayesBSpline.theta_to_coef(θ_mean_NUTS)).(t), color=:blue, label="NUTS")
