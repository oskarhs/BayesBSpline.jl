struct CubicSplineDist <: ContinuousUnivariateDistribution
    K::Int # number of splines in basis, need at least K = 4
    a_mat::AbstractMatrix{<:Real} # K×(K-3) matrix of coefficients for the normalized b-splines
    p_k::AbstractVector{<:Real} # Array specifying the prior probability p(k)
    _norm_fac::AbstractMatrix{<:Real} # K×(K-3) matrix of normalization factors

    function CubicSplineDist(K::Int, a_mat::AbstractMatrix{<:Real}, p_k::AbstractArray{<:Real})
        if !isapprox(sum(p_k), 1.0)
            error("Prior weights must sum to one.")
        end
        return new(K, a_mat, p_k, compute_norm_fac(K))
    end
end

CubicSplineDist(K::Int, a_mat::AbstractMatrix{<:Real}) = CubicSplineDist(K, a_mat, fill(1.0/(K-3.0), K-3))

# Simple constructor, initializes a_mat corresponding to a uniform prior mean
function CubicSplineDist(K::Int)
    a_mat = Matrix{Float64}(undef, K, K-3)
    for k = 4:K
        a_mat[1:k,k-3] = coef_to_theta(fill(1.0, k), k)
    end
    return CubicSplineDist(K, a_mat, fill(1.0/(K-3.0), K-3))
end

# Can create a constructor later that takes in a given density and that centers the prior on this, but not a high priority rn

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

# This is considerably faster, worth keeping
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
# Implement rand method (should probably return a spline)
#function rand(rng::AbstractRNG, d::CubicSplineDist)

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