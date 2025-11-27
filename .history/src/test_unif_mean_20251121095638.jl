using Random, Statistics
using ForwardDiff, BandedMatrices
using LinearAlgebra, Distributions, BSplineKit, Plots, Optim, Integrals

include(joinpath(@__DIR__, "BayesBSpline.jl"))
using .BayesBSpline

