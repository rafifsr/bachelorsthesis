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
    du[1] = - σ1*u[1]
    du[2] = σ1*u[1] - σ2*u[2] 
    du[3] = σ2*u[2]
end

# Initial conditions and parameters
u0 = [1.0, 0.0, 0.0]  # Initial concentrations of A, B, and C
T = 20.0             # Final time
tspan = (0.0, T)      # Time span
p = (0.5, 0.5, 0.1, 0.5)   # Parameters k1, k2, σ1, σ2

# Problem definition
prob = SDEProblem(f, g, u0, tspan, p)
sol = solve(prob, EM(), dt=0.01, saveat=0.1)

# Plot
plot(sol, label=["x" "y" "z"], xlabel="Time", ylabel="Concentration", lw=2, legend=:topright, 
     title="Stochastic Reaction Model", fontfamily="Computer Modern", ylims=(0, 1.5))

# # Monte Carlo simulation
# N = 10
# dt = 0.01
# tsteps = 0.0:dt:T
# Y = zeros(length(tsteps), N)
# for i in 1:N
#     sol = solve(prob, EM(), dt=dt, saveat=tsteps)
#     Y[:, i] .= sol[2, :]
# end

# plot(tsteps, Y, xlabel="Time", ylabel="Concentration Y", lw=2, legend=:topright, 
#      title="Stochastic Reaction Model", fontfamily="Computer Modern")