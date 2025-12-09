"""
    BSMChains{T<:Real}

Struct holding posterior samples from a `BSMModel`.

# Fields
* `samples`: Vector of NamedTuple holding posterior samples of model parameters.
* `model`: The `BSMModel` object to which samples were fit.
* `n_samples`: Total number of Monte Carlo samples. 
* `n_burnin`: Number of burn-in samples.
"""
struct BSMChains{T<:Real, M<:BSMModel, V<:AbstractVector{<:NamedTuple}}
    samples::V
    model::M
    n_samples::Int
    n_burnin::Int
    function BSMChains{T}(samples::V, model::M, n_samples::Int, n_burnin::Int) where {T, M, V}
        return new{T, M, V}(samples, model, n_samples, n_burnin)
    end
end

function Base.show(io::IO, ::MIME"text/plain", bsmc::BSMChains)
    println(io, "BSMChains object holding ", bsmc.n_samples, " posterior samples, including ", bsmc.n_burnin, " burn-in samples.")
    println(io, bsmc.model)
    nothing
end

Base.show(io::IO, bsm::BSMModel) = show(io, MIME("text/plain"), bsm)

Base.eltype(::BSMChains{T, M, V}) where {T, M, V} = T

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
#BSplineKit.basis(bsmc::B) where {B<:BSMChains} = bsmc.basis

"""
    Distributions.mean(bsmc::BSMChains, t::Real) -> Real
    Distributions.mean(bsmc::BSMChains, t::AbstractVector{<:Real}) -> Vector{<:Real}

Compute the approximate posterior mean of f(t) via Monte Carlo samples.
"""
function Distributions.mean(bsmc::BSMChains, t::Real)
    bs = basis(bsmc.model)
    nonburn_params = bsmc.samples[bsmc.n_burnin+1:end]
    spline_coefs = Matrix{Float64}(undef, (length(bsmc.model), length(nonburn_params)))
    for i in eachindex(nonburn_params)
        spline_coefs[:, i] = nonburn_params[i].coef
    end
    mean_coef = mapslices(mean, spline_coefs; dims=2)[:]
    f = Spline(bs, mean_coef)
    return f(t)
end
function Base.Broadcast.broadcasted(::typeof(mean), bsmc::BSMChains, t::AbstractVector{<:Real}) # Specialized broadcasting routine so that we don't recompute the mean of the coefficients.
    bs = basis(bsmc.model)
    nonburn_params = bsmc.samples[bsmc.n_burnin+1:end]
    spline_coefs = Matrix{Float64}(undef, (length(bsmc.model), length(nonburn_params)))
    for i in eachindex(nonburn_params)
        spline_coefs[:, i] = nonburn_params[i].coef
    end
    mean_coef = mapslices(mean, spline_coefs; dims=2)[:]
    f = Spline(bs, mean_coef)
    return f.(t)
end
function Distributions.mean(bsmc::BSMChains, t::AbstractVector{<:Real})
    bs = basis(bsmc.model)
    nonburn_params = bsmc.samples[bsmc.n_burnin+1:end]
    spline_coefs = Matrix{Float64}(undef, (length(bsmc.model), length(nonburn_params)))
    for i in eachindex(nonburn_params)
        spline_coefs[:, i] = nonburn_params[i].coef
    end
    mean_coef = mapslices(mean, spline_coefs; dims=2)[:]
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
    f_samp = pdf(bsmc.model, bsmc.samples[bsmc.n_burnin+1:end], t)

    return mapslices(x -> quantile(x, q), f_samp; dims=2)[:]
end

function Distributions.quantile(bsmc::BSMChains, t::AbstractVector{T}, q::AbstractVector{T}) where {T<:Real}
    if !all(0 .≤ q .≤ 1)
        throw(DomainError("All requested quantile levels must lie in the interval [0,1]."))
    end
    #f_samp = evaluate_posterior_density(bsmc, t)
    f_samp = pdf(bsmc.model, bsmc.samples[bsmc.n_burnin+1:end], t)
    
    # Compute quantiles for each row (t point) across coef samples
    result = mapslices(x -> quantile(x, q), f_samp; dims=2)
    return result  # shape: (length(t), length(q))
end

Distributions.median(bsmc::BSMChains, t) = quantile(bsmc, t, 0.5)

"""
    var(bsmc::BSMChains, t)

Compute the posterior variance of f(t) for every element in the collection `t`.
"""
function Distributions.var(bsmc::BSMChains, t::AbstractVector{<:Real})
    f_samp = pdf(bsmc.model, bsmc.samples[bsmc.n_burnin+1:end], t)
    
    result = mapslices(x -> var(x), f_samp; dims=2)[:]
    return result
end
function Distributions.var(bsmc::BSMChains, t::Real)
    f_samp = pdf(bsmc.model, bsmc.samples[bsmc.n_burnin+1:end], t)
    
    result = mapslices(x -> var(x), f_samp; dims=2)[1]
    return result
end

"""
    std(bsmc::BSMChains, t)

Compute the posterior standard deviation of f(t) for every element in the collection `t`.
"""
Distributions.std(bsmc::BSMChains, t) = sqrt(var(bsmc, t))