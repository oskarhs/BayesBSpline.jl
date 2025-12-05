abstract type BayesianDensityModel end

"""
    BSMModel{T<:Real, A<:AbstractSplineBasis}
    
Struct representing a B-spline mixture model.

The BSMModel struct is used to generate quantities that are needed for the model fitting procedure.

# Constructors
    
    BSMModel(x::AbstractVector{<:Real}, [K::Integer=get_default_splinedim(x)], [bounds::Tuple{<:Real,<:Real}=get_default_bounds(x)]; kwargs...) 
    BSMModel(x::AbstractVector{<:Real}, basis::AbstractBSplineBasis; kwargs...)

# Arguments
* `x`: The data vector.
* `basis`: The B-spline basis in the model. Defaults to a regular (augmented) spline basis covering [minimum(x) - 0.05*R, maximum(x) + 0.05*R] where `R` is the sample range. 
* `K`: B-spline basis dimension of a regular augmented spline basis. Defaults to max(100, min(200, ⌈n/5⌉))
* `bounds`: A tuple specifying the range of the `K`-dimensional B-spline basis. Defaults to [minimum(x) - 0.05*R, maximum(x) + 0.05*R] where `R` is the sample range. 

# Keyword arguments
* `n_bins`: Number of bins used when fitting the `BSMModel` to data. Binned fitting can be disabled by setting this equal to `nothing`. Defaults to `1000`.
* `a_τ`: Shape hyperparameter for the global smoothing parameter τ².
* `b_τ`: Rate hyperparameter for the global smoothing parameter τ².
* `a_δ`: Shape hyperparameter for the local smoothing parameters δₖ².
* `b_δ`: Rate hyperparameter for the local smoothing parameters δₖ².

# Returns
* `bsm`: A B-Spline mixture model object.

# Extended help

### Binned fitting
To disable binned fitting, one can set the `n_bins=nothing`.
Note that the binning is only used as part of the model fitting procedure, and the structure of the resulting model object is the same regardless of whether the binning step is performed or not.
Empirically, the results obtained from running the binned and unbinned model fitting procedures tend to be very similar.
We therefore recommend using the binned fitting procedure, due to the large improvements in model fitting speed, particularly for larger samples.

### Hyperparameter selection
The prior distributions of the local and global smoothing parameters are given by

    τ² ∼ InverseGamma(a_τ, b_τ)
    δₖ² ∼ InverseGamma(a_δ, b_δ),   1 ≤ k ≤ K-3.

As noninformative defaults, we reccomend using `a_τ = 1`, `b_τ = 1e-3`, `a_δ = 0.5`, `b_δ = 0.5`.
To control the smoothness in the resulting density estimates, we recommend adjusting the value of `b_τ`, with smaller values yielding smoother curves.
Similar models for regression suggest that values in the range [5e-5, 5e-3] are reasonable.
"""
struct BSMModel{T<:Real, A<:AbstractBSplineBasis, NT<:NamedTuple}
    data::NT
    basis::A
    a_τ::T
    b_τ::T
    a_δ::T
    b_δ::T
    function BSMModel{T}(x::AbstractVector{<:Real}, basis::A; n_bins::Union{Nothing,<:Integer}=1000, a_τ::Real=1.0, b_τ::Real=1e-3, a_δ::Real=0.5, b_δ::Real=0.5) where {T<:Real, A<:AbstractBSplineBasis}
        bounds = boundaries(basis)
        check_bsmkwargs(x, n_bins, bounds, a_τ, b_τ, a_δ, b_δ) # verify that supplied parameters make sense

        K = length(basis)

        # Here: determine μ via the medians (e.g. we penalize differences away from the values that yield a uniform prior mean)
        μ = compute_μ(basis, T)

        # Set up difference matrix:
        P = BandedMatrix((0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3)), (K-3, K-1))

        T_a_τ = T(a_τ)
        T_b_τ = T(b_τ)
        T_a_δ = T(a_δ)
        T_b_δ = T(b_δ)
        T_x = T.(x)

        # Normalize the data to the interval [0, 1]
        #z = @. (T_x - T_bounds[1]) / (T_bounds[2] - T_bounds[1])
        
        n = length(x)
        if !isnothing(n_bins)
            # Create binned B-Spline basis matrix
            B, b_ind, bincounts = create_spline_basis_matrix_binned(T_x, basis, n_bins)
            log_B = log.(B)

            data = (log_B = log_B, b_ind = b_ind, bincounts = bincounts, μ = μ, P = P, n = n)
        else
            B, b_ind = create_spline_basis_matrix(T_x, basis)
            log_B = log.(B)

            # Here: determine μ via the medians (e.g. we penalize differences away from the values that yield a uniform prior mean)
            μ = compute_μ(basis, T)

            # Set up difference matrix:
            P = BandedMatrix((0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3)), (K-3, K-1))

            data = (log_B = log_B, b_ind = b_ind, μ = μ, P = P, n = n)
        end
        return new{T,A,typeof(data)}(data, basis, T_a_τ, T_b_τ, T_a_δ, T_b_δ)
    end
end
BSMModel{T}(x::AbstractVector{<:Real}, K::Integer=get_default_splinedim(x), bounds::Tuple{<:Real,<:Real}=get_default_bounds(x); kwargs...) where {T<:Real} = BSMModel{T}(x, BSplineBasis(BSplineOrder(4), LinRange(bounds[1], bounds[2], K-2)); kwargs...)
BSMModel{T}(x::AbstractVector{<:Real}, bounds::Tuple{<:Real,<:Real}=get_default_bounds(x); kwargs...) where {T<:Real} = BSMModel{T}(x, get_default_splinedim(x), bounds; kwargs...)
BSMModel(args...; kwargs...) = BSMModel{Float64}(args...; kwargs...)

BSplineKit.basis(bsm::B) where {B<:BSMModel} = bsm.basis
BSplineKit.order(bsm::B) where {B<:BSMModel} = order(bsm.basis)
BSplineKit.length(bsm::B) where {B<:BSMModel} = length(bsm.basis)
BSplineKit.knots(bsm::B) where {B<:BSMModel} = knots(bsm.basis)

"""
    params(bsm::BSMModel) -> NTuple{4, <:Real}

Returns the hyperparameters of the B-Spline mixture model `bsm` as a tuple (a_τ, b_τ, a_δ, b_δ)
"""
Distributions.params(bsm::B) where {B<:BSMModel} = (bsm.a_τ, bsm.b_τ, bsm.a_δ, bsm.b_δ)

Base.eltype(::BSMModel{T,<:AbstractBSplineBasis,<:NamedTuple}) where {T<:Real} = T

# Print method for binned data
function Base.show(io::IO, ::MIME"text/plain", bsm::BSMModel{T, A, NamedTuple{(:log_B, :b_ind, :bincounts, :μ, :P, :n), Vals}}) where {T, A, Vals}
    n_bins = length(bsm.data.b_ind)
    println(io, length(bsm), "-dimensional ", nameof(typeof(bsm)), '{', eltype(bsm), "}:")
    println(io, "Using ", bsm.data.n, " binned observations on a regular grid consisting of ", n_bins, " bins.")
    print(io, " basis:  ")
    let io = IOContext(io, :compact => true, :limit => true)
        summary(io, basis(bsm))
    end
    println(io, "\n order:  ", order(bsm))
    let io = IOContext(io, :compact => true, :limit => true)
        print(io, " knots:  ", knots(bsm))
    end
    nothing
end

# Print method for unbinned data
function Base.show(io::IO, ::MIME"text/plain", bsm::BSMModel{T, A, NamedTuple{(:log_B, :b_ind, :μ, :P, :n), Vals}}) where {T, A, Vals}
    println(io, length(bsm), "-dimensional ", nameof(typeof(bsm)), '{', eltype(bsm), "}:")
    println(io, "Using ", bsm.data.n, " unbinned observations.")
    print(io, " basis:  ")
    let io = IOContext(io, :compact => true, :limit => true)
        summary(io, basis(bsm))
    end
    println(io, "\n order:  ", order(bsm))
    let io = IOContext(io, :compact => true, :limit => true)
        print(io, " knots:  ", knots(bsm))
    end
    nothing
end

Base.show(io::IO, bsm::BSMModel) = show(io, MIME("text/plain"), bsm)


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