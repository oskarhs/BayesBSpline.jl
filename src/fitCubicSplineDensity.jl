"""
    fit(CubicSplineDensity, x::AbstractVector{<:Real}, K::Int; binning::Bool=true, nbins::Int=500, maxiter::Int=50, tol::Real=1e-4)

Fits a `CubicSplineDensity` with a given maximal basis dimension to data using the EM algorithm.

Provides a binned implementation suitable for larger data sets.

# Arguments
`CubicSplineDensity`: Type of model fitted to the data
`x`: Sample used to construct the density estimate.
`K`: Maximum spline basis dimension in the mixture.

# Keyword arguments
`binning`: Boolean indicating whether or not binning should be used when computing the estimate for sample sizes greater than `1000`. Defaults to `true`.
`nbins`: Number of bins used if the binned estimator is computed. Defaults to `500`. Ignored if `binning==false` or `n ≤ 1000`.
`maxiter`: Maximum number of iterations used in the EM algorithm. Defaults to `50`.
`tol`: Tolerance parameter for convergence checks. The EM algorithm terminates when `norm(θ_new - θ_old, 1) < tol`. Defaults to `1e-4`.

# Examples
```julia-repl
julia> x = [0.037, 0.208, 0.189, 0.656, 0.45, 0.846, 0.986, 0.751, 0.249, 0.447];
julia> fit(CubicSplineDensity, x, 20); # maximum basis dimension of 20.
julia> f(0.5)
0.8789407981471621
```
"""
function StatsAPI.fit(::Type{CubicSplineDensity}, x::AbstractVector{<:Real}, K::Int;
                        binning::Bool=true, nbins::Int=500, maxiter::Int=100, tol::Real=1e-3)
    f = CubicSplineDensity(K)
    if length(x) ≤ 1000 || binning==false
        _fit!(f, x, maxiter, tol)
    else
        _fit_binned!(f, x, nbins, maxiter, tol)
    end
    return f
end

function _fit!(f::CubicSplineDensity, x::AbstractVector{<:Real}, maxiter::Int=50, tol::Real=1e-3)
    n = length(x)
    _weights = Vector{Float64}(undef, f.K-3)
    Threads.@threads for k = 4:f.K
        θ_new, llik = _fit_em(x, k, f._norm_fac[1:k, k-3], tol, maxiter)
        f._θ[1:k, k-3] = θ_new
        _weights[k-3] = llik - 0.5*k*log(n)
    end
    _weights[isnan.(_weights)] .= -Inf 
    _weights = exp.(_weights .- maximum(_weights))
    @inbounds f._weights[1:f.K-3] .= _weights / sum(_weights)
end

function _fit_em(x::AbstractVector{<:Real}, k::Int, _norm_fac::AbstractVector{<:Real}, tol::Real=1e-3, maxiter=50)
    b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, k-2)) # K-dimensional thingy means we use k-2 knots for cubic splines
    n = length(x)
    q = Vector{Float64}(undef, 4)
    θ_new = Vector{Float64}(undef, k)
    θ_old = fill(1.0/k, k)
    for _ = 1:maxiter
        fill!(θ_new, 0.0)
        for i = 1:n
            @inbounds j, bs = evaluate_all(b, x[i])
            @. q = @views log(θ_old[j:-1:j-3]) + log(bs * _norm_fac[j:-1:j-3])
            q = exp.(q .- maximum(q))
            q = q / sum(q)
            @simd for m in 1:4
                r = j-m+1
                @inbounds θ_new[r] += q[m] / n
            end
        end
        if norm(θ_new - θ_old, 1) < tol
            break
        else 
            θ_old = copy(θ_new)
        end
    end
    @inbounds s = Spline(b, θ_new .* @views(_norm_fac[1:k]))
    llik = sum(x->log(s(x)), x)
    return θ_new, llik
end


function _fit_binned!(f::CubicSplineDensity, x::AbstractVector{<:Real}, nbins::Int, maxiter::Int=50, tol::Real=1e-3)
    N = bin_regular(x, 0.0, 1.0, nbins, true)
    mids = LinRange(0.5/nbins, 1.0-0.5/nbins, nbins)
    n = length(x)
    _weights = Vector{Float64}(undef, f.K-3)
    Threads.@threads for k = 4:f.K
        b = BSplineBasis(BSplineOrder(4), LinRange(0.0, 1.0, k-2)) # K-dimensional thingy means we use k-2 knots for cubic splines
        θ_new = Vector{Float64}(undef, k)
        @inbounds θ_old = f._θ[1:k, k-3]
        for l = 1:maxiter
            θ_new = fill!(θ_new, 0.0)
            for i in eachindex(mids)
                @inbounds j, bs = b(mids[i])
                @inbounds denominator = n*dot(@views(θ_old[j:-1:j-3]) .* @views(f._norm_fac[j:-1:j-3,k-3]), bs)
                @simd for m in eachindex(bs)
                    r = j-m+1
                    @inbounds θ_new[r] += ifelse(N[i] > 0.0, N[i] * θ_old[r] * f._norm_fac[r,k-3] * bs[m] / denominator, 0.0)
                end
            end
            if norm(θ_new - θ_old, 1) < tol
                break
            else 
                θ_old = copy(θ_new)
            end
        end
        @inbounds f._θ[1:k,k-3] = θ_new
        @inbounds s = Spline(b, θ_new .* @views(f._norm_fac[1:k, k-3]))
        @inbounds _weights[k-3] = sum( @.(N * log( s(mids) )) ) - 0.5*log(n)*k
    end
    # set NaN-weights to zero:
    _weights[isnan.(_weights)] .= -Inf 
    _weights = exp.(_weights .- maximum(_weights))
    @inbounds f._weights[1:f.K-3] .= _weights / sum(_weights)
end