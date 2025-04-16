using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations
using Plots

# Parameters
r = 1.0
σ = 0.5
q = 1.0
x_0 = 0.0
dt = 0.01

# Drift term
function f!(dx, x, p, t)
    dx[1] = r * x[1] - q * x[1]^3
end

# Diffusion term
function g!(dx, x, p, t)
    dx[1] = σ
end

# Initial condition
x0 = [x_0]

# Time span
tspan = (0.0, 100.0)

# Problem definition
prob = SDEProblem(f!, g!, x0, tspan)

# Solve the SDE
sol = solve(prob, EM(), dt=dt)

# Plot the solution
plot(sol, label="Double Well Path", xlabel="Time", ylabel="X(t)", title="Double Well Potential SDE", 
     legend=:topright, grid=true)
xlims!(0, maximum(sol.t))
ylims!(-2, 2)