import Pkg
Pkg.activate(@__DIR__)
using Plots, LaTeXStrings, Measures, Distributions

# Parameters
μ_max = 0.5     # 1/h
μ_max_display = 0.45  # For visual display and annotation
K_s_display = 2.0  # For visual display and annotation
K_s_monod = 1    # For Monod curve shape
σ = 0.5
gauss_scale = 0.3

# Equations
mu(S) = μ_max * S / (K_s_monod + S)
@info "K_s_display = $K_s_display"
gauss(S) = gauss_scale * pdf(Normal(K_s_display, σ), S)

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
    xlims = (0, 4),
    ylims = (0.0, 0.5),
    ticks = false,
    left_margin = 7mm,
    bottom_margin = 7mm,
    right_margin = 7mm,
    yguidefont = font(10, "Computer Modern", :blue))

# Add visual markers (on main axis only)
hline!(plt, [μ_max_display], linestyle = :dash, color = :black, label = "")
annotate!(plt, 0.3, μ_max_display + 0.015, text(L"\mu_{\max}", :right, 10, :blue))

vline!(plt, [K_s_display], linestyle = :dash, color = :black, label = "")
annotate!(plt, K_s_display + 0.04, 0.015, text(L"K_s", :left, 10, :black))

# Plot Gaussian on twin axis
gauss_ax = twinx(plt)
plot!(gauss_ax, S_vals, g_vals,
    ylabel = L"\mathrm{Noise~Intensity}~\sigma",
    lw = 2,
    color = :green,
    linestyle = :dot,
    legend = false,
    xlims = (0, 4),
    ylims = (0, 0.5),
    ticks = false,
    yguidefont = font(10, "Computer Modern", :green))

hline!(plt, [maximum(g_vals)], linestyle = :dash, color = :black, label = "")
annotate!(plt, 4, maximum(g_vals) + 0.0125, text(L"\sigma_{\max}", :right, 10, :green))

savefig(plt, "Figures/monod_equation.pdf")
