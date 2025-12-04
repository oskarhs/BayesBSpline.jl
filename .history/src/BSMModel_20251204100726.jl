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
struct BSMModel{T<:Real, A<:AbstractBSplineBasis, V<:AbstractVector{T}}
    x::V
    basis::A
    a_τ::T
    b_τ::T
    a_δ::T
    b_δ::T
    function BSMModel{T,A}(x::V{S}, basis::A; a_τ=1, b_τ=1e-3, a_δ=0.5, b_δ=0.5) where {T<:Real, A<:AbstractBSplineBasis, V<:AbstractVector}
        return new{T,A,V}(T.(x), basis, T(a_τ), T(b_τ), T(a_δ), T(b_δ))
    end
end

BSMModel{T}(basis::A; kwargs...) where {T<:Real, A<:AbstractBSplineBasis} = BSMModel{T, A}(basis; kwargs...)
BSMModel{T}(K::Integer; kwargs...) where {T<:Real} = BSMModel{T}(BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2)); kwargs...)
BSMModel(args...; kwargs...) = BSMModel{Float64}(args...; kwargs...)

BSplineKit.basis(bsm::B) where {B<:BSMModel} = bsm.basis
BSplineKit.order(bsm::B) where {B<:BSMModel} = order(bsm.basis)
BSplineKit.length(bsm::B) where {B<:BSMModel} = length(bsm.basis)
BSplineKit.knots(bsm::B) where {B<:BSMModel} = knots(bsm.basis)


Base.eltype(::BSMModel{T,<:AbstractBSplineBasis}) where {T<:Real} = T

function Base.show(io::IO, ::MIME"text/plain", bsm::BSMModel)
    println(io, length(bsm), "-element ", nameof(typeof(bsm)), '{', eltype(bsm), '}', ':')
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

# Just set up basis matrices and similar
# Also set up the support of the estimate...
# Remember: rescale data to [0,1]!!!
function (bsm::BSMModel)(x::AbstractVector{T}; nbins::Union{Nothing, <:Integer}=1000) where {T<:Real}
    n = length(x)
    if !isnothing(nbins)
        B, b_ind, bincounts = create_spline_basis_matrix_binned(x, basis)
        data = (B = B, b_ind = b_ind, bincounts = bincounts, n = n)
    else
        B, b_ind = create_spline_basis_matrix(x, basis)
        data = (B = B, b_ind = b_ind, n = n)
    end

    return BayesianDensityModel(bsm, data)
end
# Basically, we want this to return an object that we feed into sample/mfvb

function StatsBase.sample(rng::AbstractRNG, bsm::BayesianDensityModel{M, NamedTuple{(:B, :b_ind, :bincounts), D}}, n_samp::Int) where {M<:BSMModel, D}
    println("Hello!")
end
function StatsBase.sample(bsm::BayesianDensityModel{M, NamedTuple{(:B, :b_ind, :bincounts), D}}, n_samp::Int) where {M<:BSMModel, D}
    StatsBase.sample(Random.default_rng(), bsm, n_samp)
end