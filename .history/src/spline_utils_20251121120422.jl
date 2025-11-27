# Numerically stable variant of log(1 + exp(x))
softplus(x::Real) = ifelse(x ≥ 0, x + log(1+exp(-x)), log(1+exp(x)))

# Numerically stable sigmoid
sigmoid(x::Real) = ifelse(x ≥ 0, 1/(1 + exp(-x)), exp(x)/(1 + exp(x)))

# Numerically stable softmax
function softmax(x::AbstractVector{T}) where {T<:Real}
    xmax = maximum(x)
    num = @. exp(x - xmax)
    return num /sum(num)
end

# Count the number of times each integer from 1:K occurs in the array `z`
function countint(z::AbstractVector{<:Integer}, K::Int)
    counts = zeros(Int, K)
    for i in eachindex(z)
        counts[z[i]] += 1
    end
    return counts
end

# Map an unconstrained K-1 dimensional vector to the K-simplex through stickbreaking
function stickbreaking(β::AbstractVector{T}) where {T<:Real}
    K = length(β) + 1
    log_π = Vector{T}(undef, K-1)
    softplus_sum = zero(T)
    for k in 1:K-1
        softplus_sum += softplus(β[k])
        log_π[k] = β[k] - softplus_sum
    end
    temp1 = exp.(log_π)
    p = vcat(temp1, 1.0 - sum(temp1))
    return p
end

# Find the mixture weights corresponding to given coefficients in the unnormalized b-spline basis
function coef_to_theta(coef::AbstractVector{T}) where {T<:Real}
    K = length(coef)
    θ = Vector{T}(undef, K)
    if K == 4
        θ = 0.25*coef
    elseif K == 5
        θ[1] = 0.125*coef[1]
        θ[2:K-1] = 0.25*coef[2:K-1]
        θ[K] = 0.125*coef[K]
    else
        for j = 1:3
            θ[j] = coef[j]*j/(4.0*(K-3.0))
        end
        for j = 4:K-3
            θ[j] = coef[j]*1.0/(K-3.0)
        end
        for j = K-2:K
            θ[j] = coef[j]*(K-j + 1.0)/(4.0*(K-3.0))
        end
    end
    return θ
end

# Find the b-spline coefficients corresponding to given mixture weights in the normalized b-spline basis
function theta_to_coef(θ::AbstractVector{T}) where {T<:Real}
    K = length(θ)
    coef = Vector{T}(undef, K)
    if K == 4
        coef = 4.0*θ
    elseif K == 5
        coef[1] = 8.0*θ[1]
        coef[2:K-1] = 4.0*θ[2:K-1]
        coef[K] = 8.0*θ[K]
    else
        for j = 1:3
            coef[j] = 4.0*(K-3.0)*θ[j] / j
        end
        for j = 4:K-3
            coef[j] = (K-3.0)*θ[j]
        end
        for j = K-2:K
            coef[j] = 4.0*(K-3.0)*θ[j] / (K-j + 1.0)
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

# Compute bin counts on a regular grid consisting of `M` bins over the interval [xmin, xmax]
function bin_regular(x::AbstractVector{T}, xmin::T, xmax::T, M::Int, right::Bool) where {T<:Real}
    R = xmax - xmin
    bincounts = zeros(T, M)
    edges_inc = M/R
    if right
        for val in x
            idval = min(M-1, floor(Int, (val-xmin)*edges_inc+eps())) + 1
            bincounts[idval] += 1.0
        end
    else
        for val in x
            idval = max(0, floor(Int, (val-xmin)*edges_inc-eps())) + 1
            bincounts[idval] += 1.0
        end
    end
    return bincounts
end