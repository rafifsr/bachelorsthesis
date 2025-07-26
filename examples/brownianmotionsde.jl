using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations, Plots, LaTeXStrings

# Parameters
r = 0.0
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
sol4 = solve(prob, EM(), dt=0.01)
sol5 = solve(prob, EM(), dt=0.01)

# Plot the solution
plot(legend=false, grid=true, fontfamily="Computer Modern",)
plot!(sol, linecolor=:blue, linestyle=:solid)
plot!(sol2, linecolor=:red, linestyle=:dash)
# plot!(sol3, linecolor=:green, linestyle=:dot)
# plot!(sol4, linecolor=:orange, linestyle=:dashdot)
xlabel!(L"\textrm{Time}~t")
ylabel!(L"\textrm{Position}~X(t)")
ylims!(-5, 5)
xlims!(0, 10)
savefig("Figures/brownian_motion_plot2.pdf")