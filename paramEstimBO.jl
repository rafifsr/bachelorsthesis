using Pkg
Pkg.activate(@__DIR__)
using DiffEqBayes, OrdinaryDiffEq, RecursiveArrayTools, Distributions, Plots, StatsPlots,
      BenchmarkTools, TransformVariables, DynamicHMC, Turing

# ------------------------------
# 1. Model Definition
# ------------------------------
function pendulum(du, u, p, t)
    ω, L = p
    x, y = u
    du[1] = y
    du[2] = -ω * y - (9.8 / L) * sin(x)
end

# ------------------------------
# 2. Define ODE Problem
# ------------------------------
u0 = [1.0, 0.1]
tspan = (0.0, 10.0)
prob1 = ODEProblem(pendulum, u0, tspan, [1.0, 2.5])
sol = solve(prob1, Tsit5())
plot(sol)

# ------------------------------
# 3. Dummy Data
# ------------------------------
t = collect(range(1, stop = 10, length = 10))
randomized = VectorOfArray([(sol(t[i]) + 0.01randn(2)) for i in 1:length(t)])
data = convert(Array, randomized)
scatter!(data')

# ------------------------------
# Bayesian Parameter Estimation
# ------------------------------
priors = [
    truncated(Normal(0.1, 1.0), lower = 0.0),
    truncated(Normal(3.0, 1.0), lower = 0.0),
]
bayesian_result = turing_inference(prob1, Tsit5(), t, data, priors; 
                                   syms = [:omega, :L])