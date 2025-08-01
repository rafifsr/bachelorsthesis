using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations
using Statistics
using Plots
using LaTeXStrings
using Random

# Drift term
function f(du, u, p, t) # Reaction kinetics x -> y -> z
    k1 = p[1]
    k2 = p[2]
    du[1] = -k1*u[1] 
    du[2] = k1*u[1] - k2*u[2]
    du[3] = k2*u[2]
end

# Diffusion term
function g(du, u, p, t)
    σ1 = p[3]
    σ2 = p[4]
    σ3 = p[5]
    du[1] = 0
    du[2] = σ2*u[2]
    du[3] = 0
end

# Initial conditions and parameters
u0 = [1.0, 0.0, 0.0]  # Initial concentrations of A, B, and C
T = 20.0             # Final time
tspan = (0.0, T)      # Time span
p = (0.5, 0.1, 0.1, 0.1, 0.1)   # Parameters k1, k2, σ1, σ2, σ3

# Problem definition
prob = SDEProblem(f, g, u0, tspan, p)
sol = solve(prob, EM(), dt=0.01, saveat=0.1)

# Plot
plot(sol, label=["[X]" "[Y]" "[Z]"], xlabel="Time", ylabel="Concentration", lw=2, legend=:topright, 
     title="Stochastic Reaction Model", fontfamily="Computer Modern", ylims=(0, 1.5))