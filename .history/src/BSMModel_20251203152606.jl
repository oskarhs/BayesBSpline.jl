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
struct BSMModel{A<:AbstractBSplineBasis, T<:Real}
    basis::A
    a_τ::T
    b_τ::T
    a_δ::T
    b_δ::T
    function BSMModel{A,T}(basis::A; a_τ=1, b_τ=1e-3, a_δ=0.5, b_δ=0.5) where {A<:AbstractBSplineBasis, T<:Real}
        return new{A,T}(basis, T(a_τ), T(b_τ), T(a_δ), T(b_δ))
    end
end

BSMModel{T}(basis::A; kwargs...) where {A<:AbstractBSplineBasis, T<:Real} = BSMModel{A, T}(basis; kwargs...)
BSMModel{T}(K::Integer; kwargs...) where {T<:Real} = BSMModel{T}(BSplineBasis(BSplineOrder(4), LinRange(0, 1, K-2)); kwargs...)
BSMModel(args...; kwargs...) = BSMModel{Float64}(args...; kwargs...)

function BSMModel(basis::A; kwargs...) where {A<:AbstractBSplineBasis}
    T = promote_type(typeof(a_τ), typeof(b_τ), typeof(a_δ), typeof(b_δ))
    return BSMModel{A,T}(basis; a_τ  T(a_τ), T(b_τ), T(a_δ), T(b_δ))
end

BSplineKit.basis(bsm::B) where {B<:BSMModel} = bsm.basis

# Just set up basis matrices and similar
# Also set up the support of the estimate...
# Remember: rescale data to [0,1]!!!
function (bsm::BSMModel)(x::AbstractVector{T}; nbins::Union{Nothing, <:Integer}=1000) where {T<:Real}
    #if !isnothing(nbins)
    B, b_ind, bincounts = create_spline_basis_matrix_binned(x, basis(bsm))
    data = (B = B, b_ind = b_ind, bincounts = bincounts)
    #end

    return BayesianDensityModel(bsm, data)
end
# Basically, we want this to return an object that we feed into sample/mfvb

#= function StatsBase.sample(rng::AbstractRNG, bsm::BSMModel{M, NamedTuple{B, b_ind, bincounts}}, n_samp::Int) where {M<:BSMModel}
    println("Hello!")
end =#