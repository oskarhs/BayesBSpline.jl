abstract type BayesianDensityModel end

"""
    BSMModel{A<:AbstractSplineBasis, T<:Real}
    
Initialize a B-spline mixture model.

The prior distributions of the local and global smoothing parameters are given by

    τ² ∼ InverseGamma(a_τ, b_τ)
    δₖ² ∼ InverseGamma(a_δ, b_δ),   1 ≤ k ≤ K-3.

# Arguments
* `x`: The data vector.
* `basis`: The B-spline basis in the model.

# Keyword arguments
* `a_τ`: Shape hyperparameter for the global smoothing parameter τ².
* `b_τ`: Rate hyperparameter for the global smoothing parameter τ².
* `a_δ`: Shape hyperparameter for the local smoothing parameters δₖ².
* `b_δ`: Rate hyperparameter for the local smoothing parameters δₖ².

# Returns
* `bsm`: A B-Spline mixture model object.
"""
struct BSMModel{T<:Real, A<:AbstractBSplineBasis, NT<:NamedTuple}
    data::NT
    basis::A
    bounds::Tuple{T, T}
    a_τ::T
    b_τ::T
    a_δ::T
    b_δ::T
    function BSMModel{T}(x::AbstractVector{<:Real}, basis::A; n_bins::Union{Nothing,<:Integer}=1000, bounds::Tuple{<:Real,<:Real}=get_default_bounds(x), a_τ::Real=1.0, b_τ::Real=1e-3, a_δ::Real=0.5, b_δ::Real=0.5) where {T<:Real, A<:AbstractBSplineBasis}
        check_bsmkwargs(x, n_bins, bounds, a_τ, b_τ, a_δ, b_δ) # verify that supplied parameters make sense

        T_a_τ = T(a_τ)
        T_b_τ = T(b_τ)
        T_a_δ = T(a_δ)
        T_b_δ = T(b_δ)
        T_x = T.(x)
        T_bounds = (T(bounds[1]), T(bounds[2]))

        # Normalize the data to the interval [0, 1]
        z = @. (T_x - T_bounds[1]) / (T_bounds[2] - T_bounds[1])
        
        n = length(x)
        if !isnothing(n_bins)
            B, b_ind, bincounts = create_spline_basis_matrix_binned(z, basis, n_bins)
            data = (B = B, b_ind = b_ind, bincounts = bincounts, n = n)
        else
            B, b_ind = create_spline_basis_matrix(z, basis)
            data = (B = B, b_ind = b_ind, n = n)
        end
        return new{T,A,typeof(data)}(data, basis, T_bounds, T_a_τ, T_b_τ, T_a_δ, T_b_δ)
    end
end

BSMModel{T}(x::AbstractVector{<:Real}, K::Integer; kwargs...) where {T<:Real} = BSMModel{T}(x, BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2)); kwargs...)
BSMModel{T}(x::AbstractVector{<:Real}; kwargs...) where {T<:Real} = BSMModel{T}(x, get_default_splinedim(x); kwargs...)
BSMModel(args...; kwargs...) = BSMModel{Float64}(args...; kwargs...)

BSplineKit.basis(bsm::B) where {B<:BSMModel} = bsm.basis
BSplineKit.order(bsm::B) where {B<:BSMModel} = order(bsm.basis)
BSplineKit.length(bsm::B) where {B<:BSMModel} = length(bsm.basis)
BSplineKit.knots(bsm::B) where {B<:BSMModel} = knots(bsm.basis)

"""
    params(bsm::BSMModel)

Returns the hyperparameters of the B-Spline mixture model `bsm` as a tuple (a_τ, b_τ, a_δ, b_δ)
"""
Distributions.params(bsm::B) where {B<:BSMModel} = (bsm.a_τ, bsm.b_τ, bsm.a_δ, bsm.b_δ)

Base.eltype(::BSMModel{T,<:AbstractBSplineBasis, NT}) where {T<:Real, NT} = T

function Base.show(io::IO, ::MIME"text/plain", bsm::BSMModel{T, A, NamedTuple{(:B, :b_ind, :bincounts, :n), D}}) where {T, A, D}
    n_bins = length(bsm.data.b_ind)
    println(io, length(bsm), "-dimensional ", nameof(typeof(bsm)), '{', eltype(bsm), "}:")
    println(io, "Using ", bsm.data.n, " binned observations, on a regular grid consisting of ", n_bins, " bins:")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, " bounds: ", bsm.bounds)
    end
#=     let io = IOContext(io, :compact => true, :limit => true)
        println(io, length(bsm), "-dimensional ", nameof(typeof(bsm)), '{', eltype(bsm), "}:", " on [", bsm.bounds[1], ", ", bsm.bounds[2], "] with ", bsm.data.n, " binned observations, using a regular grid consisting of ", n_bins, " bins:")
    end
 =#    print(io, " basis:  ")
    summary(io, basis(bsm))
    println(io, "\n order:  ", order(bsm))
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, " knots:  ", knots(bsm))
    end
    nothing
end

Base.show(io::IO, bsm::BSMModel{T, A, NamedTuple{(:B, :b_ind, :bincounts, :n), D}}) where {T, A, D} = show(io, MIME("text/plain"), bsm)


function get_default_splinedim(x::AbstractVector{<:Real})
    n = length(x)
    return max(min(200, ceil(Int, n/10)), 100)
end

function get_default_bounds(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    R = xmax - xmin
    return xmin - 0.05*R, xmax + 0.05*R
end

function check_bsmkwargs(x::AbstractVector{<:Real}, n_bins::Union{Nothing,<:Integer}, bounds::Tuple{<:Real, <:Real}, a_τ::Real, b_τ::Real, a_δ::Real, b_δ::Real)
    if !isnothing(n_bins) && n_bins ≤ 1
        throw(ArgumentError("Number of bins must be a positive integer or 'nothing'."))
    end
    xmin, xmax = extrema(x)
    if bounds[1] ≥ bounds[2]
        throw(ArgumentError("Supplied upper bound must be strictly greater than the lower bound."))
    elseif bounds[1] > xmin || bounds[2] < xmax
        throw(ArgumentError("Data is not contained within supplied bounds."))
    end
    hyperpar = [a_τ, b_τ, a_δ, b_δ]
    hyperpar_symb = [:a_τ, :b_τ, :a_δ, :b_δ]
    for i in eachindex(hyperpar)
        if hyperpar[i] ≤ 0.0
            throw(ArgumentError("Hyperparameter $(hyperpar_symb[i]) must be strictly positive."))
        end
    end
end

# Basically, we want this to return an object that we feed into sample/mfvb

#= function StatsBase.sample(rng::AbstractRNG, bsm::BayesianDensityModel{M, NamedTuple{(:B, :b_ind, :bincounts), D}}, n_samp::Int) where {M<:BSMModel, D}
    println("Hello!")
end
function StatsBase.sample(bsm::BayesianDensityModel{M, NamedTuple{(:B, :b_ind, :bincounts), D}}, n_samp::Int) where {M<:BSMModel, D}
    StatsBase.sample(Random.default_rng(), bsm, n_samp)
end =#