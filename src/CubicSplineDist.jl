"""
    struct CubicSplineDist(K::Int)

Random probability density used to fit Bayesian B-spline models of variable dimension.

`K` controls the maximum number of splines used for the computation, offering a tradeoff between greater model flexibility and computational efficiency.
Main usage is through the `fit!` function, which updates the parameters of a `CubicSplineDist` to those of the posterior given `x` using mean-field variational inference.
To evaluate the density at a given point, we can compute the posterior mean via `mean(d, x)`. Other quantities are most easily determined via Monte Carlo simulations from the posterior distributions, such as quantiles.

# Examples
```julia-repl
julia> x = [0.037, 0.208, 0.189, 0.656, 0.45, 0.846, 0.986, 0.751, 0.249, 0.447];
julia> d = CubicSplineDist(20) # use a basis of at most 20 B-splines
julia> fit!(d, x);
julia> mean(d, 0.5);
```
"""
struct CubicSplineDist <: ContinuousUnivariateDistribution
    K::Int
    a_mat::AbstractMatrix{<:Real}
    p_k::AbstractVector{<:Real}
    _norm_fac::AbstractMatrix{<:Real}

    function CubicSplineDist(K::Int, a_mat::AbstractMatrix{<:Real}, p_k::AbstractArray{<:Real})
        if !isapprox(sum(p_k), 1.0)
            error("Prior weights must sum to one.")
        end
        return new(K, a_mat, p_k, compute_norm_fac(K))
    end
end

CubicSplineDist(K::Int, a_mat::AbstractMatrix{<:Real}) = CubicSplineDist(K, a_mat, fill(1.0/(K-3.0), K-3))

# Simple constructor, initializes a_mat corresponding to a uniform prior mean
function CubicSplineDist(K::Int; a::Real=1.0)
    a_mat = Matrix{Float64}(undef, K, K-3)
    for k = 4:K
        a_mat[1:k,k-3] = coef_to_theta(fill(a, k), k)
    end
    return CubicSplineDist(K, a_mat, fill(1.0/(K-3.0), K-3))
end

"""
    rand([rng::AbstractRNG,], d::CubicSplineDist)

Generate a random probability density from the distribution d. Returns a cubic spline density.

# Examples
```julia-repl
julia> x = [0.037, 0.208, 0.189, 0.656, 0.45, 0.846, 0.986, 0.751, 0.249, 0.447];
julia> d = CubicSplineDist(20) # use a basis of at most 20 B-splines
julia> s = rand(d);
julia> s(0.5);
```
"""
function rand(rng::AbstractRNG, d::CubicSplineDist)
    k = rand(rng, DiscreteNonParametric(4:d.K, d.p_k))
    θ = rand(rng, Dirichlet(@views(d.a_mat[1:k, k-3])))
    b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, k-2))
    return Spline(b, theta_to_coef(θ, k))
end
function rand(d::CubicSplineDist)
    rng = Xoshiro()
    return rand(rng, d)
end

"""
    mean(d::CubicSplineDist, x::Real)

Compute the mean of the random density `d` at a given point `x`.
```
"""
function mean(d::CubicSplineDist, x::Real) # add range chekcs here later
    val = 0.0
    for k = 4:d.K # k is the number of basis functions
        val_k = 0.0
        b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, k-2)) # K-dimensional thingy means we use k-2 knots for cubic splines
        i, bs = b(x)
        for j in eachindex(bs)
            l = i-j+1
            val_k += bs[j] * d.a_mat[l,k-3]*d._norm_fac[l,k-3] # Recall that Julia stores in column-major format (add inbounds later)
        end
        val += val_k*d.p_k[k-3] / sum(@views(d.a_mat[1:k, k-3]))
    end
    return val
end
function Base.broadcast(mean, d::CubicSplineDist, x::AbstractVector{<:Real})
    val = zeros(Float64, length(x))
    for k = 4:d.K # k is the number of basis functions
        b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, k-2)) # K-dimensional thingy means we use k-2 knots for cubic splines
        norm = sum(@views(d.a_mat[1:k, k-3]))
        s = Spline(b, @views(d.a_mat[1:k,k-3]/norm .* d._norm_fac[1:k,k-3]))
        @. val += s(x) * d.p_k[k-3]
    end
    return val
end

"""
    quantile([rng::AbstractRNG,], d::CubicSplineDist, x::AbstractVector{<:Real}, q::AbstractVector{<:Real}; n_sim::Int=500)

Compute the `q`-quantiles of `d` evaluated at the points `x` via Monte Carlo.

The keyword argument `n_sim` controls the number of simulations used to compute the estimate. Defaults to `n_sim = 500`.
```
"""
function quantile(rng::AbstractRNG, d::CubicSplineDist, x::AbstractVector{<:Real}, q::AbstractVector{<:Real}; n_sim::Int=500)
    n_eval = length(x)
    q_val = Array{Float64}(undef, n_eval, length(q))
    fs = Array{Float64}(undef, n_eval, n_sim)
    for i in 1:n_sim
        s = rand(rng, d)
        fs[1:n_eval, i] = s.(x)
    end
    quantiles = Matrix{Float64}(undef, n_eval, length(q))
    for m in eachindex(q)
        quantiles[1:n_eval, m] = mapslices(x -> quantile(x, q[m]), fs; dims=2)
    end
    return quantiles
end
function quantile(d::CubicSplineDist, x::AbstractVector{<:Real}, q::AbstractVector{<:Real}; n_sim::Int=500)
    rng = Xoshiro()
    return quantile(rng, d, x, q; n_sim=n_sim)
end