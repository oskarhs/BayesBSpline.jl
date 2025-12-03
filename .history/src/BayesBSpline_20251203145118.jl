module BayesBSpline

using BSplineKit
export BSplineKit

using Random, Distributions, Base.Threads, Optim
#using Plots

import Distributions: mean, quantile, ContinuousUnivariateDistribution
import Base: rand, broadcast
import LinearAlgebra: norm, dot
import StatsAPI
import SpecialFunctions: loggamma, digamma

export CubicSplineDist, mean, fit!, quantile, rand
export CubicSplineDensity, eval_density, fit, weights, fit_turbo

export PSplineBasis

include("CubicSplineDist.jl")
include("CubicSplineDensity.jl")
include("fitCubicSplineDensity.jl")
include("fitCubicSplineDist.jl")
include("spline_utils.jl")
include("uniform_prior_mean.jl")
include("gibbs_binned.jl")
include("BSMModel.jl")


end
