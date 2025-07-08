using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations, LinearAlgebra, Statistics, Random
using Plots, Distributions, KernelDensity, StatsBase

# Parameters
p = (0.5, 0.5, 0.5)  # k1, k2, σ
u0 = [2.0, 0.0, 0.0]  # Initial concentrations of X, Y, Z
T = 60.0
dt = 0.05
N = Int(T/dt)
M = 50_000  # number of trajectories
tspan = (0.0, T)
tsteps = 0:dt:T

# Define drift and diffusion for the components
function f!(du, u, p, t)
    X, Y, Z = u
    k1, k2 = p
    du[1] = -k1 * X
    du[2] = k1 * X - k2 * Y
    du[3] = k2 * Y
end

function g!(du, u, p, t)
    X, Y, Z = u
    σ = p[3]
    du[1] = 0
    du[2] = σ * Y
    du[3] = 0
end

# Laguerre basis functions up to degree 3
function laguerre_design_matrix(y::Vector{Float64}, d::Int)
    Φ = zeros(length(y), d + 1)
    for i in 1:length(y)
        Φ[i,1] = 1.0
        if d >= 1
            Φ[i,2] = 1 - y[i]
        end
        if d >= 2
            Φ[i,3] = 1 - 2*y[i] + 0.5*y[i]^2
        end
        if d >= 3
            Φ[i,4] = 1 - 3*y[i] + 1.5*y[i]^2 - (1/6)*y[i]^3
        end
    end
    return Φ
end

# Store all trajectories
Xs = zeros(M, length(tsteps))
Ys = zeros(M, length(tsteps))

# Store stopping times
τ = fill(length(tsteps), M)

# Simulate all trajectories
prob = SDEProblem(f!, g!, u0, tspan, p)
Random.seed!(42)
for i in 1:M
    sol = solve(prob, EM(), dt=dt, saveat=tsteps)
    Xs[i, :] .= sol[1,:]
    Ys[i, :] .= sol[2,:]
end

# Initialize matrices for Longstaff-Schwartz
V = copy(Ys) # copy(Xs) if we're using Xs
degree = 3  # Degree of polynomial basis
β_matrix = zeros(N, degree + 1)

# Longstaff-Schwartz backward induction in time
for n in (length(tsteps)-1):-1:2
    y_now = Ys[:, n]
    value_future = V[:, n+1]

    # Optional: Filter out non-positive values (in-the-money paths for options)
    # all of the values are supposed to be positive
    itm_indices = findall(y -> y > 0, y_now)
    if length(itm_indices) < 10
        continue
    end

    y_itm = y_now[itm_indices]
    vf_itm = value_future[itm_indices]

    # Fit polynomial regression using Laguerre basis functions
    Φ = laguerre_design_matrix(y_itm, degree)
    β = Φ \ vf_itm
    
    # Compute continuation value
    # Note: This is where the polynomial regression is used to compute the continuation value
    continuation_value = Φ * β
    stop_now = y_itm .> continuation_value

    # Update stopping times
    # Note: This is where we update the stopping times based on the stopping condition
    for (idx, stop) in zip(itm_indices, stop_now)
        if stop
            V[idx,n] = y_now[idx]
            τ[idx] = min(τ[idx], n)
        else
            V[idx,n] = V[idx,n+1]
        end
    end

    # Save regression coefficients
    β_matrix[n, :] .= β

end

# Compile outputs
Y_opt_values = [Ys[i, τ[i]] for i in 1:M]
τ_times = [tsteps[τ[i]] for i in 1:M]

println("Estimated optimal expected value of Y: ", round(mean(Y_opt_values), digits=4))
println("Expected optimal stopping time: ", round(mean(τ_times), digits=4), " seconds")
println("Variance of stopping time: ", round(var(τ_times), digits=4), " seconds^2")
println("Standard deviation of stopping time: ", round(std(τ_times), digits=4), " seconds")
println("95% confidence interval for stopping time: ", round(quantile(τ_times, 0.025), digits=4), " to ", round(quantile(τ_times, 0.975), digits=4), " seconds")
println("95% confidence interval for expected value of Y: ", round(quantile(Y_opt_values, 0.025), digits=4), " to ", round(quantile(Y_opt_values, 0.975), digits=4))

# Plotting reaction trajectories
# plot(tsteps, Ys[1,:], 
#      xlabel="Time (seconds)", ylabel="Y Concentration", 
#      title="Trajectories of Y Concentration", 
#      legend=false, lw=0.5, alpha=0.1, color=:blue, 
#      fontfamily="Computer Modern")

# Plot: Histogram + Gamma fit + KDE
histogram(τ_times, bins=30, normalize=true, label="Histogram", xlabel="Stopping Time", ylabel="Density", title="Optimal Stopping Time PDF", 
          legend=:topright, fontfamily="Computer Modern", lw=2, alpha=0.5)

# Gamma distribution fit
fit_gamma = fit(Gamma, τ_times)
plot!(tsteps, pdf.(fit_gamma, tsteps), lw=2, label="Gamma Fit")

# KDE overlay
kde_est = kde(τ_times)
plot!(kde_est.x, kde_est.density, lw=2, linestyle=:dash, label="KDE")

xlims!(0, 10)

# # 2D KDE
# data = hcat(τ_times, Y_opt_values)
# kde2d = kde(data)

# # Heatmap
# heatmap(kde2d.x, kde2d.y, kde2d.density',
#     xlabel="Stopping Time", ylabel="Y at Stopping",
#     colorbar_title="Density",
#     fontfamily="Computer Modern")

# # Save the plot
# savefig("Figures/optimal_stopping_time.pdf")

# # 2D histogram (time vs Y value at stopping)
# time_bins = range(0, T, length=30)
# y_bins = range(0, maximum(Y_opt_values), length=30)

# h = fit(Histogram, (τ_times, Y_opt_values), (time_bins, y_bins))

# # Normalize to get a density
# density = h.weights / sum(h.weights) / ((time_bins[2]-time_bins[1]) * (y_bins[2]-y_bins[1]))

# # Plot heatmap
# heatmap(h.edges[1], h.edges[2], density', 
#     xlabel="Stopping Time", ylabel="Y at Stopping", 
#     title="Joint PDF of Stopping Time and Y", 
#     colorbar_title="Density")