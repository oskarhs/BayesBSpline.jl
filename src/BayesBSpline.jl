module BayesBSpline

using BSplineKit
export BSplineKit

using Random, Distributions, Memoization
#using Plots

import Distributions: mean, ContinuousUnivariateDistribution
import Base: rand, broadcast
import LinearAlgebra: norm, dot
import StatsAPI

export CubicSplineDist, mean
export CubicSplineDensity, eval_density, fit, weights

include("spline_utils.jl")
include("CubicSplineDist.jl")
include("CubicSplineDensity.jl")
include("fitCubicSplineDensity.jl")

end
