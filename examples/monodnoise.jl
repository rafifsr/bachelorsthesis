import Pkg
Pkg.activate(@__DIR__)
using Plots, LaTeXStrings, Measures, Distributions

# Parameters
μ_max = 0.5     # 1/h
K_s = 2.0  # For visual display and annotation
µ_ln = log(K_s) + μ_max^2 # Mean of the log-normal distribution

# Equations
mu(S) = μ_max * S / (K_s + S)
gauss(S) = μ_max * pdf(LogNormal(log(K_s)+ μ_max^2, μ_max), S) 
# gauss(S) = μ_max^2 * pdf(Normal(K_s, μ_max), S)

# Values
S_vals = 0:0.01:10
μ_vals = mu.(S_vals)
g_vals = gauss.(S_vals)

# Plot Monod
plt = plot(S_vals, μ_vals,
    xlabel = L"\mathrm{Substrate~Concentration}~S",
    ylabel = L"\mathrm{Specific~Growth~Rate}~\mu",
    lw = 2,
    legend = false,
    color = :blue,
    xlims = (0, 10),
    ylims = (0.0, 0.6),
    ticks = false,
    left_margin = 7mm,
    bottom_margin = 7mm,
    right_margin = 7mm,
    yguidefont = font(10, "Computer Modern", :blue))

# Add visual markers (on main axis only)
hline!(plt, [μ_max], linestyle = :dash, color = :black, label = "")
annotate!(plt, 0.7, μ_max + 0.015, text(L"\mu_{\max}", :right, 10, :blue))

plot!(plt, 0:0.01:K_s, fill(μ_max/2, Int(K_s / 0.01) + 1),
    linestyle = :dash, color = :black, label = "")
annotate!(plt, 1, μ_max/2 + 0.02, text(L"\mu_{\max}/2", :right, 10, :blue))

vline!(plt, [K_s], linestyle = :dash, color = :black, label = "")
annotate!(plt, K_s + 0.04, 0.015, text(L"K_s", :left, 10, :black))

# Plot Gaussian on twin axis
gauss_ax = twinx(plt)
plot!(gauss_ax, S_vals, g_vals,
    ylabel = L"\mathrm{Noise~Intensity}~\sigma",
    lw = 2,
    color = :green,
    linestyle = :dot,
    legend = false,
    xlims = (0, 10),
    ylims = (0, 0.6),
    ticks = false,
    yguidefont = font(10, "Computer Modern", :green))

x_interval = K_s:0.01:10  # Define the specific interval for x
y_value = maximum(g_vals)
plot!(plt, x_interval, fill(y_value, length(x_interval)),
    linestyle = :dash, color = :black, label = "")
annotate!(plt, 10, y_value + 0.0125, text(L"\sigma_{\max}", :right, 10, :green))

# savefig(plt, "Figures/musigma_normal.pdf")
# savefig(plt, "Figures/musigma_lognormal.pdf")
