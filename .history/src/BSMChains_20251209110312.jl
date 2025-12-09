"""
    BSMChains{T<:Real}

Struct holding posterior samples from a `BSMModel`.

# Fields
* `samples`: Vector of NamedTuples holding posterior samples of model parameters.
* `basis`: The B-spline basis of the fitted BSMModel.
* `n_samples`: Total number of Monte Carlo samples. 
* `n_burnin`: Number of burn-in samples.
"""
struct BSMChains{T<:Real, A<:AbstractBSplineBasis, V<:AbstractVector{<:NamedTuple}}
    samples::V
    basis::A
    n_samples::Int
    n_burnin::Int
    function BSMChains(samples::V, basis::A, n_samples::Int, n_burnin::Int) where {A, V}
        return new{eltype(eltype(V)), A, V}(samples, basis, n_samples, n_burnin)
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
function Distributions.mean(bsmc::BSMChains, t::AbstractVector{<:Real})
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

The latter function returns a Matrix of dimension `(length(t), length(q))`, where each column corresponds to a given quantile.
"""
function Distributions.quantile(bsmc::BSMChains, t::AbstractVector{T}, q::T) where {T<:Real}
    if !(0 ≤ q ≤ 1)
        throws(DomainError("Requested quantile level is not in [0,1]."))
    end
    f_samp = evaluate_posterior_density(bsmc, t)

    return mapslices(x -> quantile(x, q), f_samp; dims=2)[:]
end

function Distributions.quantile(bsmc::BSMChains, t::AbstractVector{T}, q::AbstractVector{T}) where {T<:Real}
    if !all(0 .≤ q .≤ 1)
        throw(DomainError("All requested quantile levels must lie in the interval [0,1]."))
    end
    f_samp = evaluate_posterior_density(bsmc, t)
    
    # Compute quantiles for each row (t point) across coef samples
    result = mapslices(x -> quantile(x, q), f_samp; dims=2)
    return result  # shape: (length(t), length(q))
end

"""
    evaluate_posterior_density(
        bsmc::BSMChains, t::AbstractVector{<:Real}
    ) -> Matrix{<:Real}

    evaluate_posterior_density(
        bsmc::BSMChains, t::AbstractVector{<:Real}, samples_ind::AbstractVector{<:Integer}
    ) -> Matrix{<:Real}

Evaluate f(t | θ) on the grid `t` for (a subset of) the draws from the posterior.

# Arguments
* `bsmc`: Posterior samples object
* `t`: Grid at which we evaluate the posterior density draws.
* `samples_ind`: Indeces of the non-burn in samples for which to evaluate the density. If not provided, the density is evaluated for every non-burn in posterior draw.

# Returns
* `f_samp`: Matrix of dimension `(length(t), length(samples_ind))`, where each column stores f(t) evaluated at a single posterior sample of θ.
"""
function evaluate_posterior_density(bsmc::BSMChains, t::AbstractVector{T}) where {T<:Real}
    bs = basis(bsmc)
    B_sparse = create_unnormalized_sparse_spline_basis_matrix(t, bs)
    coefs = bsmc.samples.coef[:, bsmc.n_burnin+1:end]
    f_samp = B_sparse * coefs
    return f_samp
end

function evaluate_posterior_density(bsmc::BSMChains, t::AbstractVector{T}, samples_ind::AbstractVector{<:Integer}) where {T<:Real}
    bs = basis(bsmc)
    B_sparse = create_unnormalized_sparse_spline_basis_matrix(t, bs)
    coefs = bsmc.samples.coef[:, bsmc.n_burnin+1:end][:, samples_ind]
    f_samp = B_sparse * coefs
    return f_samp
end