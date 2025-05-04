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

# Implement rand method (should probably return a spline)
function rand(rng::AbstractRNG, d::CubicSplineDist)
    k = rand(rng, DiscreteNonParametric(4:d.K, d.p_k))
    θ = rand(rng, Dirichlet(@views(d.a_mat[1:k, k-3])))
    b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, k-2))
    return Spline(b, theta_to_coef(θ, k))
end

# Mean of the random cubic spline, evaluated at a single x
function mean(d::CubicSplineDist, x::Real)
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
    val = zeros(T, length(x))
    for k = 4:d.K # k is the number of basis functions
        b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, k-2)) # K-dimensional thingy means we use k-2 knots for cubic splines
        norm = sum(@views(d.a_mat[1:k, k-3]))
        s = Spline(b, @views(d.a_mat[1:k,k-3]/norm .* d._norm_fac[1:k,k-3]))
        @. val += s(x) * d.p_k[k-3]
    end
    return val
end

# Evaluate the q-quantile of f(x)
function quantile(d::CubicSplineDist, x::AbstractVector{<:Real}, q::AbstractVector{<:Real}; rng::AbstractRNG=Xoshiro(), sim::Int=500)
    n_eval = length(x)
    q_val = Array{Float64}(undef, n_eval, length(q))
    fs = Array{Float64}(undef, n_eval, sim)
    for i in 1:sim
        s = rand(rng, d)
        fs[1:n_eval, i] = s.(x)
    end
    quantiles = [quantile(row, q) for row in eachrow(fs)]
    return val
end

#= function test_evaluate()
    K = 50

    d = CubicSplineDist(K)

    ip = IntegralProblem((t,p) -> mean(d, t), 0.0, 1.0)
    I = solve(ip, GaussLegendre()).u
    println("Integral: ", I)

    x = LinRange(0, 1, 1000)
    plot(x, mean(d, x), ylims=[0.0, Inf])
    #x = LinRange(0, 1, 1000)
    #plot(x, mean.(d, x), ylims=[0.0, Inf])
    #println(mean(d, x))
end
test_evaluate() =#