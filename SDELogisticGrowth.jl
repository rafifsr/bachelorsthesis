using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations

# Drift: logistic growth
function f(du, u, p, t)
    r = p[1]
    du[1] = r * u[1] * (1 - u[1])
end

# Diffusion: multiplicative noise
function g(du, u, p, t)
    σ = p[2]
    du[1] = σ * u[1]
end

u0 = [0.01]              # Initial biomass (normalized)
tspan = (0.0, 100.0)      
p = (1.0, 0.2)           # r=1.0, σ=0.2

prob = SDEProblem(f, g, u0, tspan, p)
sol = solve(prob, SRIW1())

using Plots
plot(sol, label="Biomass X(t)", xlabel="Time", ylabel="X(t)")
