abstract type BayesianDensityModel end

#= struct BayesianDensityModel{M, DataType}
    model::M
    data::DataType
end

function Base.show(io::IO, bdm::BayesianDensityModel)
    println(io, "BayesianDensityModel of type ", nameof(typeof(bdm.model)))
    println(io, bdm.model)
    println(io, bdm.data)
end
 =#
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
* `a_τ`: Prior shape hyperparameter for the global smoothing parameter τ².
* `b_τ`: Prior rate hyperparameter for the global smoothing parameter τ².
* `a_δ`: Prior shape hyperparameter for the local smoothing parameters.
* `b_δ`: Prior rate hyperparameter for the local smoothing parameters.

# Returns
* `bsm`: A B-Spline mixture model object.
"""
struct BSMModel{T<:Real, A<:AbstractBSplineBasis, NT<:NamedTuple}
    data::NT
    basis::A
    a_τ::T
    b_τ::T
    a_δ::T
    b_δ::T
    function BSMModel{T,A}(x::AbstractVector{<:Real}, basis::A; n_bins::Union{Nothing,<:Integer}=1000, a_τ::Real=1, b_τ::Real=1e-3, a_δ::Real=0.5, b_δ::Real=0.5) where {T<:Real, A<:AbstractBSplineBasis}
        check_bsmkwargs(n_bins, a_τ, b_τ, a_δ, b_δ)
        new_x = T.(x)
        n = length(x)
        if !isnothing(n_bins)
            B, b_ind, bincounts = create_spline_basis_matrix_binned(new_x, basis, n_bins)
            data = (B = B, b_ind = b_ind, bincounts = bincounts, n = n)
        else
            B, b_ind = create_spline_basis_matrix(new_x, basis)
            data = (B = B, b_ind = b_ind, n = n)
        end
        return new{T,A,typeof(data)}(data, basis, T(a_τ), T(b_τ), T(a_δ), T(b_δ))
    end
end

BSMModel{T}(x::AbstractVector{<:Real}, basis::A; kwargs...) where {T<:Real, A<:AbstractBSplineBasis} = BSMModel{T, A}(x, basis; kwargs...)
BSMModel{T}(x::AbstractVector{<:Real}, K::Integer; kwargs...) where {T<:Real} = BSMModel{T}(x, BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2)); kwargs...)
BSMModel(args...; kwargs...) = BSMModel{Float64}(args...; kwargs...)

BSplineKit.basis(bsm::B) where {B<:BSMModel} = bsm.basis
BSplineKit.order(bsm::B) where {B<:BSMModel} = order(bsm.basis)
BSplineKit.length(bsm::B) where {B<:BSMModel} = length(bsm.basis)
BSplineKit.knots(bsm::B) where {B<:BSMModel} = knots(bsm.basis)

"""
    params(bsm::BSMModel)

Returns the hyperparameters of the B-Spline mixture model bsm as a tuple (a_τ, b_τ, a_δ, b_δ)
"""
Distributions.params(bsm::B) where {B<:BSMModel} = (bsm.a_τ, bsm.b_τ, bsm.a_δ, bsm.b_δ)

Base.eltype(::BSMModel{T,<:AbstractBSplineBasis}) where {T<:Real} = T

function Base.show(io::IO, ::MIME"text/plain", bsm::BSMModel)
    println(io, length(bsm), "-element ", nameof(typeof(bsm)), '{', eltype(bsm), '}', " with spline basis:")
    print(io, " basis: ")
    summary(io, basis(bsm))
    println(io, "\n order: ", order(bsm))
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, " knots: ", knots(bsm))
    end
    println(io, "")
    nothing
end

Base.show(io::IO, bsm::BSMModel) = show(io, MIME("text/plain"), bsm)


function get_default_splinedim(x::AbstractVector{<:Real})
    n = length(x)
    return max(min(200, ceil(Int, n/10)), 100)
end

function check_bsmkwargs(n_bins::Union{Nothing,<:Integer}, a_τ::Real, b_τ::Real, a_δ::Real, b_δ::Real)
    if !isnothing(n_bins) && n_bins ≤ 1
        throw(ArgumentError("Number of bins must be a positive integer or 'nothing'."))
    end
    hyperpar = [a_τ, b_τ, a_τ, b_τ]
    hyperpar_symb = [:a_τ, :b_τ, :a_δ, :b_δ]
    for i in eachindex(hyperpar)
        if hyperpar[i] ≤ 0.0
            throw(ArgumentError("Hyperparameter $(hyperpar_symb[i]) must be strictly positive."))
        end
    end
end
# Just set up basis matrices and similar
# Also set up the support of the estimate...
# Remember: rescale data to [0,1]!!!
function (bsm::BSMModel)(x::AbstractVector{T}; nbins::Union{Nothing, <:Integer}=1000) where {T<:Real}
    n = length(x)
    if !isnothing(n_bins)
        B, b_ind, bincounts = create_spline_basis_matrix_binned(x, basis)
        data = (B = B, b_ind = b_ind, bincounts = bincounts, n = n)
    else
        B, b_ind = create_spline_basis_matrix(x, basis)
        data = (B = B, b_ind = b_ind, n = n)
    end

    return BayesianDensityModel(bsm, data)
end
# Basically, we want this to return an object that we feed into sample/mfvb

#= function StatsBase.sample(rng::AbstractRNG, bsm::BayesianDensityModel{M, NamedTuple{(:B, :b_ind, :bincounts), D}}, n_samp::Int) where {M<:BSMModel, D}
    println("Hello!")
end
function StatsBase.sample(bsm::BayesianDensityModel{M, NamedTuple{(:B, :b_ind, :bincounts), D}}, n_samp::Int) where {M<:BSMModel, D}
    StatsBase.sample(Random.default_rng(), bsm, n_samp)
end =#