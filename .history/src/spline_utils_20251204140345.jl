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