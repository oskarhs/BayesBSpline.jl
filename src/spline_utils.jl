# Find the mixture weights corresponding to given coefficients in the unnormalized b-spline basis
function coef_to_theta(coef::AbstractVector, k::Int)
    θ = Vector{Float64}(undef, k)
    if k == 4
        θ = 0.25*coef
    elseif k == 5
        θ[1] = 0.125*coef[1]
        θ[2:k-1] = 0.25*coef[2:k-1]
        θ[k] = 0.125*coef[k]
    else
        for j = 1:3
            θ[j] = coef[j]*j/(4.0*(k-3.0))
        end
        for j = 4:k-3
            θ[j] = coef[j]*1.0/(k-3.0)
        end
        for j = k-2:k
            θ[j] = coef[j]*(k-j + 1.0)/(4.0*(k-3.0))
        end
    end
    return θ
end

# Find the b-spline coefficients corresponding to given mixture weights in the normalized b-spline basis
function theta_to_coef(θ::AbstractVector, k::Int)
    coef = Vector{Float64}(undef, k)
    if k == 4
        coef = 4.0*θ
    elseif k == 5
        coef[1] = 8.0*θ[1]
        coef[2:k-1] = 4.0*θ[2:k-1]
        coef[k] = 8.0*θ[k]
    else
        for j = 1:3
            coef[j] = 4.0*(k-3.0)*θ[j] / j
        end
        for j = 4:k-3
            coef[j] = (k-3.0)*θ[j]
        end
        for j = k-2:k
            coef[j] = 4.0*(k-3.0)*θ[j] / (k-j + 1.0)
        end
    end
    return coef
end

# Compute factors to multiply θ with when evaluating the cubic spline
function compute_norm_fac(K::Int)
    _norm_fac = Matrix{Float64}(undef, K, K-3)
    # k = 4
    _norm_fac[1:4,1] .= 4.0
    if K == 4
        return _norm_fac
    end

    _norm_fac[1,2] = 8.0 
    _norm_fac[2:4,2] .= 4.0
    _norm_fac[5,2] = 8.0
    if K == 5
        return _norm_fac
    end

    for k = 6:K
        if k == 4
            _norm_fac[1:k,k-3] .= 4.0
        elseif k == 5
            _norm_fac[1,k-3] = 8.0
            _norm_fac[2:k-1,k-3] .= 4.0
            _norm_fac[k,k-3] = 8.0
        else
            for i = 1:3
                _norm_fac[i,k-3] = (4.0*(k-3.0))/i
            end
            for i = 4:k-3
                _norm_fac[i,k-3] = (k-3.0)
            end
            for i = k-2:k
                _norm_fac[i,k-3] = (4.0*(k-3.0))/(k-i + 1.0)
            end
        end
    end
    return _norm_fac
end

# Compute bin couns on a regular grid over the interval [xmin, xmax]
function bin_regular(x::AbstractVector{<:Real}, xmin::Real, xmax::Real, k::Int, right::Bool)
    R = xmax - xmin
    bincounts = zeros(Float64, k)
    edges_inc = k/R
    if right
        for val in x
            idval = min(k-1, floor(Int, (val-xmin)*edges_inc+eps())) + 1
            @inbounds bincounts[idval] += 1.0
        end
    else
        for val in x
            idval = max(0, floor(Int, (val-xmin)*edges_inc-eps())) + 1
            @inbounds bincounts[idval] += 1.0
        end
    end
    return bincounts
end