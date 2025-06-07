using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, RecursiveArrayTools, Plots, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO

function f(du, u, p, t)
    du[1] = dx = p[1] * u[1] - u[1] * u[2]
    du[2] = dy = -3 * u[2] + u[1] * u[2]
end

u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
p = [1.5]
prob = ODEProblem(f, u0, tspan, p)

sol = solve(prob, Tsit5())
t = collect(range(0, stop = 10, length = 200))
randomized = VectorOfArray([(sol(t[i]) + 0.01randn(2)) for i in 1:length(t)])
data = convert(Array, randomized)

newprob = remake(prob, p = [1.42])
newsol = solve(newprob, Tsit5())
plot(sol, label="Original", xlabel="Time", lw=2)
plot!(newsol, label="New", lw=2, linestyle=:dash)