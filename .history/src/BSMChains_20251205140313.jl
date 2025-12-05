"""
    BSMChains{T<:Real}

Struct holding posterior samples from a `BSMModel`.

# Fields
* `samples`: NamedTuple holding posterior samples of model parameters.
* `basis`: The B-spline basis of the fitted BSMModel.
* `n_samples`: Total number of Monte Carlo samples. 
* `n_burnin`: Number of burn-in samples.
"""
struct BSMChains{T<:Real, A<:AbstractBSplineBasis, NT<:NamedTuple}
    samples::NT
    basis::A
    n_samples::Int
    n_burnin::Int
    function BSMChains(samples::NT, basis::A, n_samples::Int, n_burnin::Int) where {A, NT}
        return new{eltype(eltype(NT)), A, NT}(samples, basis, n_samples, n_burnin)
    end
end

# 
# Base.Broadcast.broadcastable(bsmc::BSMChains) = Ref(bsmc)
#= function Base.Broadcast.broadcasted(::typeof(pdf), h::AutomaticHistogram, x::AbstractVector)
    vals = Vector{Float64}(undef, length(x))
    @inbounds for i in eachindex(x)
        vals[i] = pdf(h, x[i])
    end
    return vals
end
 =#
BSplineKit.basis(bsmc::B) where {B<:BSMChains} = bsmc.basis

"""
    Distributions.mean(bsmc::BSMChains, t::Real) -> Real
    Distributions.mean(bsmc::BSMChains, t::AbstractVector{<:Real}) -> Vector{<:Real}

Compute the approximate posterior mean of f(t) via Monte Carlo samples.
"""
function Distributions.mean(bsmc::BSMChains, t::Real)
    bs = basis(bsmc)
    mean_coef = mapslices(mean, bsmc.samples.coef[:, bsmc.n_burnin+1:end]; dims=2)[:]
    f = Spline(bs, mean_coef)
    return f(t)
end
function Base.Broadcast.broadcasted(::typeof(mean), bsmc::BSMChains, t::AbstractVector{<:Real}) # Specialized broadcasting routine so that we don't recompute the mean of the coefficients.
    bs = basis(bsmc)
    mean_coef = mapslices(mean, bsmc.samples.coef[:, bsmc.n_burnin+1:end]; dims=2)[:]
    f = Spline(bs, mean_coef)
    return f.(t)
end


"""
    quantile(
        bsmc::BSMChains, t::AbstractVector{<:Real}, q::Real
    ) -> Vector{<:Real}
    quantile(
        bsmc::BSMChains, t::AbstractVector{<:Real}, q::AbstractVector{<:Real}
    ) -> Matrix{<:Real}

Compute the posterior quantiles of f(t) on the grid of `t` via Monte Carlo samples.

The latter function returns a Matrix of dimension (length(t), length(q)), where each column corresponds to each given quantile.
"""
function Distributions.quantile(bsmc::BSMChains, t::AbstractVector{T}, q::T) where {T<:Real}
    if !(0 ≤ q ≤ 1)
        throws(DomainError("Requested quantile level is not in [0,1]."))
    end
    bs = basis(bsmc)
    coefs = bsmc.samples.coef[:, bsmc.n_burnin+1:end]
    S_samp = Matrix{T}(undef, (length(t), size(coefs, 2)))
    for i in axes(coefs, 2)
        S_samp[:,i] = Spline(bs, coefs[:,i]).(t)
    end
    return mapslices(x -> quantile(x, q), S_samp; dims=2)[:]
end

function Distributions.quantile(bsmc::BSMChains, t::AbstractVector{T}, q::AbstractVector{T}) where {T<:Real}
    if !all(0 .≤ q .≤ 1)
        throw(DomainError("All requested quantile levels must lie in the interval [0,1]."))
    end
    bs = basis(bsmc)
    coefs = bsmc.samples.coef[:, bsmc.n_burnin+1:end]
    S_samp = Matrix{T}(undef, (length(t), size(coefs, 2)))
    for i in axes(coefs, 2)
        S_samp[:,i] = Spline(bs, coefs[:,i]).(t)
    end
    
    # Compute quantiles for each row (t point) across coef samples
    result = mapslices(x -> quantile(x, q), S_samp; dims=2)
    return result  # shape: (length(t), length(q))
end

function eval_posterior_splines1(bsmc::BSMChains, t::AbstractVector{T}) where {T<:Real}
    bs = basis(bsmc)
    coefs = bsmc.samples.coef[:, bsmc.n_burnin+1:end]
    S_samp = Matrix{T}(undef, (length(t), size(coefs, 2)))
    for i in axes(coefs, 2)
        S_samp[:,i] = Spline(bs, coefs[:,i]).(t)
    end
    return S_samp
end

function eval_posterior_splines2(bsmc::BSMChains, t::AbstractVector{T}) where {T<:Real}
    bs = basis(bsmc)
    B, b_ind = create_spline_basis_matrix(t, bs)
    coefs = bsmc.samples.coef[:, bsmc.n_burnin+1:end]
    S_samp = Matrix{T}(undef, (length(t), size(coefs, 2)))
    for m in axes(coefs, 2)
        for i in eachindex(t)
            k0 = b_ind[i]
            val = zero(T)
            for l in 1:4
                k = k0 + l - 1
                val += B[i,l] * coefs[k, m]
            end
            S_samp[i,m] = val
        end
    end
    return S_samp
end

function eval_posterior_splines3(bsmc::BSMChains, t::AbstractVector{T}) where {T<:Real}
    bs = basis(bsmc)
    B_sparse = create_unnormalized_sparse_spline_basis_matrix(t, bs)
    coefs = bsmc.samples.coef[:, bsmc.n_burnin+1:end]
    #S_samp = Matrix{T}(undef, (length(t), size(coefs, 2)))
    S_samp = B_sparse * coefs
    return S_samp
end