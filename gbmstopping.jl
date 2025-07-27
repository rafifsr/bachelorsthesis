using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations, Plots, LaTeXStrings, Measures

# Parameters
r = 0.05
σ = 0.15

# Drift term
function f!(dx, x, p, t)
    dx[1] = r * x[1]
end

# Diffusion term
function g!(dx, x, p, t)
    dx[1] = σ * x[1]
end

# Initial condition
x0 = [2.0]

# Time span
tspan = (0.0, 10.0)
dt = 0.01
tsteps = 0.0:dt:10.0

# Problem definition
prob = SDEProblem(f!, g!, x0, tspan)

# Solve the SDE
sol = solve(prob, EM(), dt=dt)
sol2 = solve(prob, EM(), dt=dt)
sol3 = solve(prob, EM(), dt=dt)
sol4 = solve(prob, EM(), dt=dt)

# Exercise boundary
y(x) = (4/81) * x.^2 .+ 0.5
boundary = y(tsteps)

# Define your intersection logic
function find_first_intersection(sol, boundary, tsteps)
    for i in 1:length(sol)
        if sol[i][1] <= boundary[i]  # adjust condition if needed
            return (tsteps[i], sol[i][1], boundary[i], i)
        end
    end
    return nothing  # if no intersection found
end



# Plot the solution
p = plot(legend=:false, grid=true, fontfamily="Computer Modern", size = (900,600), margins = 5mm, tickfont = 12, guidefont = 16, legendfont = 11, xticks = false, yticks = false)
plot!(sol, label="Asset price", color=:blue, linewidth=2)
plot!(sol2, label="Asset price", color=:red, linewidth=2)
plot!(sol3, label="Asset price", color=:orange, linewidth=2)
#plot!(tsteps, boundary, label="Optimal Exercise Boundary", color=:black, linestyle =:dashdot, linewidth=2)
#hline!([4.5], color=:black, linestyle=:dash, label="Strike Price")
#annotate!(-0.3, 4.5, text(L"K", :center, 14, :black))
#vline!([9], color=:black, linestyle=:solid, label=false)
#annotate!(9, -0.25, text(L"T_N", :center, 14, :black))

# result = find_first_intersection(sol, boundary, tsteps)

# if result !== nothing
#     t_int, x_sol, x_bound, i = result
#     vline!([t_int], color=:black, linestyle=:solid, label="Optimal Exercise Time")
#     annotate!(t_int, -0.25, text(L"\tau", :center, 20, :black))
# else
#     println("No intersection found between solution and boundary.")
# end

xlabel!("Time")
ylabel!(L"X_t")
ylims!(0, 6)
xlims!(0, 10)
display(p)
savefig(p, "scenarioslsmc.pdf")