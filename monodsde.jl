using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, LinearAlgebra, Statistics, Plots

# Drift term
function monod(du, u, p, t)
    X, S, P = u               # u[1] = biomass X, u[2] = substrate S
    μ_max, K_s, Y_s, q_max, K_sp, Y_p = p      # unpack parameters

    μ = μ_max * S / (K_s + S)  # Monod growth rate with small epsilon to avoid div by zero
    q = q_max * S / (K_sp + S)  # Monod product formation rate
    dX = μ * X
    dS = - (1 / Y_s) * μ * X - (1 / Y_p) * q * X  # Monod substrate consumption rate
    dP = q * X

    du[1] = dX
    du[2] = dS
    du[3] = dP
end

# Diffusion term
function g(du, u, p, t)
    σ1 = p[7]
    σ2 = p[8]
    du[1] = -σ1*u[1]
    du[2] = σ1*u[1] - σ2*u[2]
    du[3] = σ2*u[2]
end

# Initial conditions and parameters
u0 = [10.0, 0.01, 0.0]  # Initial concentrations of X, Y, and z
T = 20.0             # Final time
tspan = (0.0, T)      # Time span
p = (0.4, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1)   # Parameters µmax, qpmax, ks, ksp, yxs, yps, σ1, σ2

# Problem definition
prob = SDEProblem(f, g, u0, tspan, p)
sol = solve(prob, EM(), dt=0.01, saveat=0.1)

# Plot
plot(sol, label=["X" "Y" "Z"], xlabel="Time", ylabel="Concentration", lw=2, legend=:topright, 
     title="Stochastic Monod Model", fontfamily="Computer Modern", ylims=(0, 10))