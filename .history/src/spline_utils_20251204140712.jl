# Find the mixture weights corresponding to given coefficients in the unnormalized B-spline basis
function coef_to_theta(coef::AbstractVector{T}, basis::A) where {T<:Real, A<:AbstractBSplineBasis}
    θ = coef ./ compute_norm_fac(basis, T)
    return θ
end

# Find the B-spline coefficients corresponding to given mixture weights in the normalized B-spline basis
function theta_to_coef(θ::AbstractVector{T}, basis::A) where {T<:Real, A<:AbstractBSplineBasis}
    coef = θ .* compute_norm_fac(basis, T)
    return coef
end

# Compute the vector Z of normalizing constants for a given B-spline basis on [0,1].
# The resulting normalized B-spline basis is given by bₖ(x) = Bₖ(x) / Zₖ
function compute_norm_fac(basis::A, T::Type{<:Real}=Float64) where {A<:AbstractBSplineBasis}
    K = length(basis)
    norm_fac = Vector{T}(undef, K)
    bmin::T, bmax::T = boundaries(basis)
    for k in 1:K
        S = integral(Spline(basis, BayesBSpline.unitvector(K, k, T)))
        norm_fac[k] = 1/(S(bmax) - S(bmin))
    end
    return norm_fac
end

# Compute the vector μ such that ∑ₖ θₖ bₖ(x) = 1 for all x, where θ = stickbreaking(μ)
function compute_μ(basis::A, T::Type{<:Real}=Float64) where {A<:AbstractBSplineBasis}
    K = length(basis)
    p0 = coef_to_theta(ones(T, K), basis)

    μ = Vector{T}(undef, K-1)
    θ_cum = Vector{T}(undef, K)
    θ_cum[1] = 0

    for k in 1:K-1
        μ[k] = logit(p0[k] / (1-θ_cum[k]))
        θ_cum[k+1] = θ_cum[k] + p0[k]
    end
    return μ
end

"""
    pointwise_quantiles(θ::AbstractMatrix{T}, basis::A, t::AbstractVector{T}, q::T) where {T<:Real, A<:AbstractBSplineBasis}
    pointwise_quantiles(θ::AbstractMatrix{T}, basis::A, t::AbstractVector{T}, q::AbstractVector{T}) where {T<:Real, A<:AbstractBSplineBasis}

Compute the posterior quantiles of f(t) over a grid of t-values.

The latter function returns a Matrix of dimension (length(t), length(q))
"""
function pointwise_quantiles end

function pointwise_quantiles(θ::AbstractMatrix{T}, basis::A, t::AbstractVector{T}, q::T) where {T<:Real, A<:AbstractBSplineBasis}
    if !(0 ≤ q ≤ 1)
        throws(DomainError("Requested quantile level is not in [0,1]."))
    end
    S_samp = Matrix{T}(undef, (length(t), size(θ, 2)))
    for i in axes(θ, 2)
        S_samp[:,i] = Spline(basis, BayesBSpline.theta_to_coef(θ[:,i], basis)).(t)
    end
    #return [quantile(S_samp[j,:], q) for j in eachindex(t)]
    return mapslices(x -> quantile(x, q), S_samp; dims=2)[:]
end

function pointwise_quantiles(θ::AbstractMatrix{T}, basis::A, t::AbstractVector{T}, q::AbstractVector{T}) where {T<:Real, A<:AbstractBSplineBasis}
    if !all(0 .≤ q .≤ 1)
        throw(DomainError("All requested quantile levels must lie in the interval [0,1]."))
    end
    
    n_t = length(t)
    n_samples = size(θ, 2)
    S_samp = Matrix{T}(undef, n_t, n_samples)
    
    for i in axes(θ, 2)
        S = Spline(basis, BayesBSpline.theta_to_coef(θ[:,i], basis))
        S_samp[:,i] .= S.(t)  # evaluate spline at all t points
    end
    
    # Compute quantiles for each row (t point) across θ samples
    result = mapslices(x -> quantile(x, q), S_samp; dims=2)
    return result  # shape: (length(t), length(q))
end