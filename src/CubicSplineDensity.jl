"""
    struct CubicSplineDensity(K::Int)

Struct used to fit a variable-dimension b-spline mixture via maximum likelihood. 

`K` controls the maximum number of splines used for the computation, offering a tradeoff between greater model flexibility and computational efficiency.
Main usage is through the `fit` function.    
A call method is also provided, evaluating a fitted CubicSplineDensity at a given point `x`.

# Examples
```julia-repl
julia> x = [0.037, 0.208, 0.189, 0.656, 0.45, 0.846, 0.986, 0.751, 0.249, 0.447];
julia> fit(CubicSplineDensity, x, 20);
julia> f(0.5)
0.8789407981471621
```
"""
struct CubicSplineDensity
    K::Int
    _θ::AbstractMatrix{<:Real}
    _weights::AbstractVector{<:Real}
    _norm_fac::AbstractMatrix{<:Real}

    function CubicSplineDensity(K::Int)
        _θ = Matrix{Float64}(undef, K, K-3)
        @inbounds for k = 4:K
            _θ[1:k,k-3] = coef_to_theta(fill(1.0, k), k)
        end
        return new(K, _θ, fill(1.0/(K-3.0), K-3), compute_norm_fac(K))
    end
end


"""
    weights(f::CubicSplineDensity)

Returns the vector of model weights post-fitting, e.g. w(k) ∝ exp(IC(k)) for each value of k∈{4,5,…,K}.
"""
function weights(f::CubicSplineDensity)
    return f._weights
end

"""
    max_dimension(f::CubicSplineDensity)

Returns K, the largest dimension of the spline bases in the mixture. 
"""
function max_dimension(f::CubicSplineDensity)
    return f.K
end

"""
    eval_density(f::CubicSplineDensity, x::Real)

Evaluate a fitted CubicSplineDensity at a given point `x`.
"""
function eval_density(f::CubicSplineDensity, x::Real)
    val = 0.0
    for k = 4:f.K # k is the number of basis functions
        val_k = 0.0
        b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, k-2)) # K-dimensional thingy means we use k-2 knots for cubic splines
        i, bs = b(x)
        @inbounds for j in eachindex(bs)
            l = i-j+1
            val_k += bs[j] * f._θ[l,k-3]*f._norm_fac[l,k-3] # Recall that Julia stores in column-major format
        end
        @inbounds val += val_k*f._weights[k-3]
    end
    return val
end

function Base.broadcast(eval_density::typeof(eval_density), f::CubicSplineDensity, x::AbstractVector{<:Real})
    val = zeros(Float64, length(x))
    for k = 4:f.K # k is the number of basis functions
        b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, k-2)) # K-dimensional thingy means we use k-2 knots for cubic splines
        @inbounds s = Spline(b, @views(f._θ[1:k,k-3] .* f._norm_fac[1:k,k-3]))
        @. val += s(x) * f._weights[k-3]
    end
    return val
end


"""
    (f::CubicSplineDensity)(x::Real)

Evaluates a fitted `CubicSplineDensity` object at a given point `x`.
Alias for `eval_density(f::CubicSplineDensity, x::Real)`.
"""
function (f::CubicSplineDensity)(x::Real)
    return eval_density(f, x)
end

function Base.broadcast((f::CubicSplineDensity), x::AbstractVector{<:Real})
    return eval_density.(f, x)
end