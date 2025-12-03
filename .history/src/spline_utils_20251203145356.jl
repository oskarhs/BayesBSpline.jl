# Numerically stable variant of log(1 + exp(x))
softplus(x::Real) = ifelse(x ≥ 0, x + log(1+exp(-x)), log(1+exp(x)))

# Numerically stable sigmoid
sigmoid(x::Real) = ifelse(x ≥ 0, 1/(1 + exp(-x)), exp(x)/(1 + exp(x)))

# Logit map
logit(x::Real) = log(x / (1-x))

# Numerically stable softmax
function softmax(x::AbstractVector{T}) where {T<:Real}
    xmax = maximum(x)
    num = @. exp(x - xmax)
    return num /sum(num)
end

# Count the number of times each integer from 1:K occurs in the array `z`
function countint(z::AbstractVector{<:Integer}, K::Int)
    counts = zeros(Int, K)
    for i in eachindex(z)
        counts[z[i]] += 1
    end
    return counts
end

# Map an unconstrained K-1 dimensional vector to the K-simplex through stickbreaking
# To ensure numerical stability, the calculation is performed in log-space.
function stickbreaking(β::AbstractVector{T}) where {T<:Real}
    K = length(β) + 1
    log_π = Vector{T}(undef, K)
    softplus_sum = zero(T)
    for k in 1:K-1
        softplus_sum += softplus(β[k])
        log_π[k] = β[k] - softplus_sum
    end
    log_π[K] = -softplus_sum
    return softmax(log_π)
end

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

# Compute bin counts on a regular grid consisting of `M` bins over the interval [xmin, xmax]
function bin_regular(x::AbstractVector{T}, xmin::T, xmax::T, M::Int, right::Bool) where {T<:Real}
    R = xmax - xmin
    bincounts = zeros(Int, M)
    edges_inc = M/R
    if right
        for val in x
            idval = min(M-1, floor(Int, (val-xmin)*edges_inc+eps())) + 1
            bincounts[idval] += 1.0
        end
    else
        for val in x
            idval = max(0, floor(Int, (val-xmin)*edges_inc-eps())) + 1
            bincounts[idval] += 1.0
        end
    end
    return bincounts
end

# Create the k'th unit vector in the canonical basis for R^K.
function unitvector(K::Int, k::Int, T::Type{<:Real}=Float64)
    if !(1 ≤ k ≤ K)
        throw(ArgumentError("Index out of range."))
    end
    unitvec = zeros(T, K)
    unitvec[k] = 1
    return unitvec
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
    NormalizedBSplineBasis{A, T}

Compute the 
"""
struct NormalizedBSplineBasis{A<:AbstractBSplineBasis, T::Type{<:Real}} <: AbstractBSplineBasis
    basis::A
    norm_fac::Vector{T}

    function NormalizedBSplineBasis(basis::A, T::Type{<:Real}) where {A}
        norm_fac = compute_norm_fac(basis, T)
        return new{A, T}(basis, norm_fac)
    end
end

NormalizedBSplineBasis(basis::A) where {A<:AbstractBSplineBasis} = NormalizedBSplineBasis(basis, Float64)

Base.parent(norm_basis::NormalizedBSplineBasis) = norm_basis.basis

"""
    PSplineBasis(m::Integer, K::Integer, T::Type{<:AbstractFloat}=Float64)

Helper function to construct a K-dimensional P-spline basis of a given order spanning the interval [0, 1].

# Arguments
* `ord`: The BSplineOrder of the basis
* `K`: The basis dimension
* `T`: Element type of spline basis knots

# Returns
* `basis`: A BSplineBasis
"""
function PSplineBasis(ord::BSplineOrder{m}, K::Integer, T::Type{<:AbstractFloat}=Float64) where {m}
    deg = m - 1
    knots = range(T(-deg/(K-deg)), T(K/(K-deg)), K+m)
    return BSplineBasis(ord, knots, augment=Val(false))
end