using DifferentialEquations, Plots

# Parameters
r = 1.0
σ = 1.0

# Drift term
function f!(dx, x, p, t)
    dx[1] = r * x[1]
end

# Diffusion term
function g!(dx, x, p, t)
    dx[1] = σ * x[1]
end

# Initial condition
x0 = [1.0]

# Time span
tspan = (0.0, 10.0)

# Problem definition
prob = SDEProblem(f!, g!, x0, tspan)

# Solve the SDE
sol = solve(prob, EM(), dt=0.01)

# Plot the solution
plot(sol, label="Geometric Brownian Motion", xlabel="Time", ylabel="X(t)", title="Geometric Brownian Motion", 
     legend=:topright, grid=true)