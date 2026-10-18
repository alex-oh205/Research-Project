using LinearAlgebra
using Plots
using Printf

## Calculate
N = 100
a = 1
e = Tridiagonal(fill(-1, N - 1), fill(2, N), fill(-1, N - 1))
x = [0; [a / (N + 1) * i for i in 1:N]; a]

values, vectors = eigen(e)

# println((values .* (n + 1)^2 ./ 2)[1:5])

## Plot
n = 1
E = n^2 * pi^2 / (N + 1)^2
xcenter = a / 2
ycenter = maximum(vectors[:, n]) / 2
plt = plot(x, [0; vectors[:, n] * sign(vectors[1, n]); 0], title="Infinite Potential Well", xlabel="x", ylabel="ψ")
annotate!(plt, xcenter, ycenter, text(@sprintf("ΔE/E: %.2e\nn = %d\nN = %d", values[n] / E - 1, n, N), :blue, :center, 12))
display(plt)

## Save
savefig(plt, "infinite_well_n_$(n)_N_$(N).png")