using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations, Plots, LaTeXStrings

# Parameters
r = 1.0
σ = 1.0

# Drift term
function f!(dx, x, p, t)
    dx[1] = r
end

# Diffusion term
function g!(dx, x, p, t)
    dx[1] = σ
end

# Initial condition
x0 = [0.0]

# Time span
tspan = (0.0, 10.0)

# Problem definition
prob = SDEProblem(f!, g!, x0, tspan)

# Solve the SDE
sol = solve(prob, EM(), dt=0.01)
sol2 = solve(prob, EM(), dt=0.01)
sol3 = solve(prob, EM(), dt=0.01)
sol4 = solve(prob, SRIW1(), dt=0.01)
sol5 = solve(prob, SRIW1(), dt=0.01)

# Plot the solution
plot(sol, label="Brownian Motion", xlabel="t", ylabel=L"B_t", title="Brownian Motion", 
     legend=false, grid=true, fontfamily="Computer Modern", linecolor=:blue)
plot!(sol2, linecolor=:red, style=:dash)
plot!(sol3, linecolor=:green, style=:dot)
# plot!(sol4, linecolor=:orange, style=:dashdot)
# plot!(sol5, linecolor=:purple)
ylims!(-5, 5)
xlims!(0, 5)
savefig("Figures/brownian_motion_plot.pdf")