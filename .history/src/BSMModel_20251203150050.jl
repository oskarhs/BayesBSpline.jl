struct BayesianDensityModel{M, DataType}
    model::M
    data::DataType
end

"""
    BSMModel{A<:AbstractSplineBasis, T<:Real, I<:Integer}
    
Initialize a B-spline mixture model.

The prior distributions of the local and global smoothing parameters are given by

    τ² ∼ InverseGamma(a_τ, b_τ)
    δₖ² ∼ InverseGamma(a_δ, b_δ),   1 ≤ k ≤ K-3.

# Arguments
* `basis`: The B-spline basis in the model.

# Keyword arguments
* `a_τ, b_τ`: Prior hyperparameters for the global smoothing parameter.
* `a_δ, b_δ`: Prior hyperparameters for the local smoothing parameters.

# Returns
* `bsm`: A B-Spline mixture model object.
"""
struct BSMModel{A<:AbstractSplineBasis, T<:Real}
    basis::A
    a_τ::T
    b_τ::T
    a_δ::T
    b_δ::T
    function BSMModel(basis::A; a_τ::T=1, b_τ::T=1e-3, a_δ::T=0.5, b_δ::T=0.5) where {A<:AbstractSplineBasis, T<:Real}
        return new{A,T}(basis, a_τ::T, b_τ::T, a_δ::T, b_δ::T)
    end
end

BSMModel{T}(basis::A; kwargs...) where {A<:AbstractSplineBasis} = BSMModel{T, A}(basis; kwargs...)
BSMModel{T}(K::Integer; kwargs...) = BSMModel{T}(BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2)); kwargs...)
BSMModel(args...) = BSMModel{Float64}(args...; kwargs...)

BSplineBasis.basis(bsm::BSMModel) = bsm.basis

# Just set up basis matrices and similar
function (bsm::BSMModel)(x::AbstractVector{T}; nbins::Union{Nothing, <:Integer}=1000) where {T<:Real}
    #if !isnothing(nbins)
    B, b_ind, bincounts = create_spline_basis_matrix_binned(x, basis)
    data = (B = B, b_ind = b_ind, bincounts = bincounts)
    #end

    return BayesianDensityModel(bsm, data)
end
# Basically, we want this to return an object that we can call with 

function StatsBase.sample(rng::AbstractRNG, bsm::BSMModel{M, NamedTuple{B, b_ind, bincounts}}, M::Int) where {M<:BSMModel}
    println("Hello!")
end