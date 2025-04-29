using DifferentialEquations
using Plots

# Monod kinetics ODE function
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

# function noise(du, u, p, t)
#     σ1 = p[4]  # unpack noise parameters
#     du[1] = -σ1 * u[2]    # noise in biomass
#     du[2] = σ1 * u[2]    # noise in substrate
# end

# Initial conditions: biomass and substrate
X₀ = 0.001  # g/L
S₀ = 10.0 # g/L
P₀ = 0.0  # g/L
u₀ = [X₀, S₀, P₀]  # Initial concentrations of X, S, and P

# Parameters: μ_max, K_s, Y
p = (0.4, 0.5, 0.5, 0.1, 0.1, 0.1)

# Time span
tspan = (0.0, 30.0)

# Define ODE problem and solve
prob = ODEProblem(monod, u₀, tspan, p)
sol = solve(prob, Tsit5(), dt=0.01, saveat=0.1)

# prob = SDEProblem(monod, noise, u₀, tspan, p)
# sol = solve(prob, EM(), dt=0.01, saveat=0.1)

# Plot the solution
plot(sol, label=["Biomass (X)" "Substrate (S)"], xlabel="Time (h)", ylabel="Concentration (g/L)", lw=2, legend=:topright, 
     title="Stochastic Monod Model", fontfamily="Computer Modern", ylims=(0, 10), xlims=(0, 30), grid=true)
