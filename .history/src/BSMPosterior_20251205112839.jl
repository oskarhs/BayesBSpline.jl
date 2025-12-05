"""
    BSMChains{T<:Real}

Struct holding posterior samples from a `BSMModel`

# Fields
* `samples`: NamedTuple holding posterior samples of model parameters.
* `basis`: The B-spline basis of the fitted BSMModel
* `n_samples`: Total number of Monte Carlo samples. 
* `n_burnin`: Number of burn-in samples
"""
struct BSMChains{T<:Real, A<:AbstractBSplineBasis, NT<:NamedTuple}
    samples::NT
    basis::A
    function BSMChains(samples::NT, basis::A) where {A, NT}
        return new{eltype(NT), A, NT}(samples, basis, n_samples, n_burnin)
    end
end