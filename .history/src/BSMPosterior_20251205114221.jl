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
    function BSMChains(samples::NT, basis::A) where {A, NT}
        return new{eltype(NT), A, NT}(samples, basis, n_samples, n_burnin)
    end
end

# 
# Base.Broadcast.broadcastable(bsmc::BSMChains) = Ref(bsmc)
function Base.Broadcast.broadcasted(::typeof(pdf), h::AutomaticHistogram, x::AbstractVector)
    vals = Vector{Float64}(undef, length(x))
    @inbounds for i in eachindex(x)
        vals[i] = pdf(h, x[i])
    end
    return vals
end

BSplineKit.basis(bsm::B) where {B<:BSMChains} = bsm.basis

"""
    Distributions.mean(bsmc::BSMChains, t::Real) -> Real
"""
function Distributions.mean(bsmc::BSMChains{T, A, NT}, t::Real) where {T, A, NT}
    bs = basis(BSMChains)
    K = length(bs)
    mean_coef = [mean(bsmc.samples.coef[k, n_samples:1]) for k in 1:K]
    f = Spline(bs, mean_coef)
    return f(t)
end
function Base.Broadcast.broadcasted(::typeof(mean), h::AutomaticHistogram, x::AbstractVector)
end