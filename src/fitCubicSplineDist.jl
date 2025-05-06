"""
    fit!(d::CubicSplineDist, x::AbstractVector{<:Real}, K::Int; binning::Bool=true, nbins::Int=500, maxiter::Int=50, tol::Real=1e-4)

Updates the parameters of a `CubicSplineDist` object to those of the posterior distribution given x using CAVI.

Provides a binned implementation suitable for larger data sets.

# Arguments
`d`: A CubicSplineDist object representing the prior distribution.
`x`: Observed sample for which the posterior distribution is computed.
`K`: Maximum spline basis dimension in the mixture.

# Keyword arguments
`binning`: Boolean indicating whether or not binning should be used when computing the estimate for sample sizes greater than `1000`. Defaults to `true`.
`nbins`: Number of bins used if the binned estimator is computed. Defaults to `500`. Ignored if `binning==false` or `n ≤ 1000`.
`maxiter`: Maximum number of iterations used in the EM algorithm. Defaults to `50`.
`tol`: Tolerance parameter for convergence checks. The EM algorithm terminates when `norm(θ_new - θ_old, 1) < tol`. Defaults to `1e-4`.

# Examples
```julia-repl
julia> x = [0.037, 0.208, 0.189, 0.656, 0.45, 0.846, 0.986, 0.751, 0.249, 0.447];
julia> d = CubicSplineDist(20) # use a basis of at most 20 B-splines
julia> fit!(d, x);
```
"""
function fit!(f::CubicSplineDist, x::AbstractVector{<:Real}, maxiter::Int=50, tol::Real=1e-3)
    n = length(x)
    p_k = Vector{Float64}(undef, f.K-3)
    a = 1.0
    Threads.@threads for k = 4:f.K
        @inbounds r_new, elbo = _fit_cavi(x, k, f.a_mat[1:k, k-3], f._norm_fac[1:k, k-3], tol, maxiter)
        @inbounds p_k[k-3] = elbo
        @inbounds f.a_mat[1:k, k-3] = @views(f.a_mat[1:k, k-3]) .+ r_new
    end
    p_k[isnan.(p_k)] .= -Inf 
    p_k = exp.(p_k .- maximum(p_k))
    @inbounds f.p_k[1:f.K-3] .= p_k / sum(p_k)
end

# Fit VB posterior using CAVI for given k
function _fit_cavi(x::AbstractVector{<:Real}, k::Int, a_vec::AbstractVector{<:Real}, _norm_fac::AbstractVector{<:Real}, tol::Real=1e-3, maxiter::Int=50)
    sum_a = sum(a_vec)
    b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, k-2)) # K-dimensional thingy means we use k-2 knots for cubic splines
    n = length(x)
    q = Vector{Float64}(undef, 4)
    r_new = Vector{Float64}(undef, k)
    elbo_new = 0.0
    r_old = fill(n/k, k)
    for _ in 1:maxiter
        elbo_new = 0.0
        fill!(r_new, 0.0)
        for i = 1:n
            @inbounds j, bs = evaluate_all(b, x[i])
            @. q = @views digamma(a_vec[j:-1:j-3] + r_old[j:-1:j-3]) + log(bs * _norm_fac[j:-1:j-3])
            q = exp.(q .- maximum(q))
            q = q / sum(q)
            for m in eachindex(bs)
                l = j-m+1
                @inbounds r_new[l] += q[m]
                @inbounds elbo_new += q[m]*(log(bs[m]*_norm_fac[l]) - log(q[m]))
            end
        end
        for j in 1:k
            elbo_new += loggamma(a_vec[j] + r_new[j]) - loggamma(a_vec[j])
        end
        elbo_new += loggamma(sum_a) - loggamma(sum_a + n)
        if norm(r_new - r_old, 1) < tol * (sum_a+n)
            break
        else 
            r_old = copy(r_new)
        end
    end
    return r_new, elbo_new
end