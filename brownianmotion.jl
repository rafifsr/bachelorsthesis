using Random, Plots

# Define the Brownian motion generator
function rBM(t::Vector{Float64})
    dt = diff([0.0; t])                     # time increments, prepend 0
    dW = randn(length(t)) .* sqrt.(dt)      # scaled normal increments
    return cumsum(dW)                       # Brownian path
end

# Time vector
t = collect(0:0.01:1)

# Generate Brownian path
path = rBM(t)

# Plot the path
plot(t, path, title="Sample Path of Brownian Motion", xlabel="Time", ylabel="W(t)",
     label="Brownian Motion", linewidth=2, legend=:topright, grid=true)
xlims!(0, maximum(t))