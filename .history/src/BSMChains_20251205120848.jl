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

Compute the approximate posterior mean of f(t) via monte carlo samples.
"""
function Distributions.mean(bsmc::BSMChains, t::Real)
    bs = basis(bsmc)
    mean_coef = mapslices(mean, bsmc.samples.coef[:, bsmc.n_burnin+1:end]; dims=2)[:]
    f = Spline(bs, mean_coef)
    return f(t)
end
function Base.Broadcast.broadcasted(::typeof(mean), bsmc::BSMChains, t::AbstractVector{<:Real}) # Specialized broadcasting 
    bs = basis(bsmc)
    mean_coef = mapslices(mean, bsmc.samples.coef[:, bsmc.n_burnin+1:end]; dims=2)[:]
    f = Spline(bs, mean_coef)
    return f.(t)
end