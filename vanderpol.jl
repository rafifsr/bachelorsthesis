using DifferentialEquations

# Define the Van der Pol oscillator SDE
function vanderpol_sde!(du, u, p, t)
    μ = p[1]
    du[1] = u[2]
    du[2] = μ * (1 - u[1]^2) * u[2] - u[1]
end

function noise!(du, u, p, t)
    du[1] = 0.0
    du[2] = p[2] # Noise intensity
end

# Initial conditions and parameters
u0 = [2.0, 0.0] # Initial state
p = [1.0, 0.5]  # μ and noise intensity
tspan = (0.0, 50.0)

# Define the SDE problem
sde_prob = SDEProblem(vanderpol_sde!, noise!, u0, tspan, p)

# Solve the SDE
sol = solve(sde_prob, EM(), dt=0.01)

# Plot the solution
using Plots
plot(sol, vars=(1, 2), xlabel="x", ylabel="v", title="Van der Pol Oscillator SDE")