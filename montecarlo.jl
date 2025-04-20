using Pkg, DifferentialEquations, Statistics, Plots, LaTeXStrings
Pkg.activate(@__DIR__)

# Parameters
r = 1.0      # intrinsic growth rate
sigma = 0.2  # noise intensity
K = 2.0      # carrying capacity
x_0 = 0.01   # initial biomass
T = 100.0    # final time
N = 10     # number of Monte Carlo simulations
dt = 0.01    # time step for Monte Carlo simulation
tsteps = 0.0:dt:T 

# Drift and diffusion functions
function f!(dx, x, p, t)
    dx[1] = r * x[1] * (1 - x[1]/K)
end

function g!(dx, x, p, t)
    dx[1] = sigma * x[1]
end

# SDE problem definition
prob = SDEProblem(f!, g!, [x_0], (0.0, T))

# Deterministic solution for comparison
det_sol = K * x_0 * exp.(r * tsteps) ./ (K .+ x_0 * (exp.(r * tsteps) .- 1))

# Monte Carlo simulation
X = zeros(length(tsteps), N)
for i in 1:N
    sol = solve(prob, EM(), dt=dt, saveat=tsteps)
    X[:, i] .= sol[1, :]
end
terminal_values = X[end, :]

# # Create a plot with two subfigures
# p1 = plot(tsteps, X, label="", xlabel="Time", ylabel="Biomass", 
#           title="Monte Carlo Simulations", legend=false, xlims=(0, T), ylims=(0, 2))
# plot!(p1, tsteps, det_sol, label="Deterministic Solution", linestyle=:dash, color=:black)

# p2 = histogram(terminal_values, bins=20, xlabel="Terminal Biomass", ylabel="Frequency", 
#                title="Distribution of Terminal Biomass", color=:blue, legend=false)

# plot(p1, p2, layout=(2,1), fontfamily="Computer Modern")

# Plot the Monte Carlo simulations
plot(tsteps, X, label="", xlabel="Time", ylabel="Biomass", title="Monte Carlo Simulations of Stochastic Logistic Growth Model", legend=false)
plot!(tsteps, det_sol, label="Deterministic Solution", linestyle=:dash, color=:black)
plot!(xlabel="Time", ylabel="Biomass", title="Monte Carlo Simulations of Stochastic Logistic Growth Model", fontfamily="Computer Modern")
plot!(xlims=(0, T), ylims=(0, K+1))

# # Plot the distribution of the terminal values from the Monte Carlo simulations
# terminal_values = X[end, :]
# histogram(terminal_values, bins=20, xlabel="Terminal Biomass", ylabel="Frequency", 
#           title="Distribution of Terminal Biomass from Monte Carlo Simulations", 
#           color=:blue, alpha=0.7, fontfamily="Computer Modern")