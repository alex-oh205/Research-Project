## Initialize
using Quantica, GLMakie, LinearAlgebra, Optim, Printf

r = 1.2
ep = 0
t = -0.25

positions = [(r * cos(i * pi / 3), r * sin(i * pi / 3), 0.0) for i in 0:5]

lat = lattice(sublat(positions...; name = :C))

## Functions
function solve(p, doplot=false)
    ep, t = p

    if doplot
        display(qplot(lat, title = "Benzene Geometric Structure"))
    end

    model = onsite(ep) + hopping(t; range = 1.01 * r)

    H = hamiltonian(lat, model)
    Hmat = Matrix(H[()])

    return solve_ham(Hmat)
end

function solve_ham(H)
    evals, evecs = eigen(Hermitian(H))

    evecs = real.(evecs)

    return evals, evecs
end

## Final solution
evals, evecs = solve([ep, t], true)

println("  n   eigval   eigvec")
for n in 1:length(evals)
    fmt = Printf.Format(" %2i  %7.3f   [" * "%6.3f " ^ (length(evals) - 1) * "%6.3f]\n")
    Printf.format(stdout, fmt, n, evals[n], evecs[:, n]...)
end

## Plot
pos = sites(lat)
xs = [p[1] for p in pos]
ys = [p[2] for p in pos]

fig = Figure(size = (900, 650))
Label(fig[0, 1:3], "Benzene Eigen Energies and States", fontsize = 20, halign = :center)

for n in 1:6
    row, col = divrem(n - 1, 3) .+ (1, 1)
    ax = Axis(fig[row, col], title = "n=$n,  E=$(round(evals[n], digits=2))",
              aspect = DataAspect())
    hidedecorations!(ax)

    for k in 1:6
        k2 = k == 6 ? 1 : k + 1
        lines!(ax, [xs[k], xs[k2]], [ys[k], ys[k2]], color = :gray, linewidth = 2)
    end

    coeffs = real.(evecs[:, n])
    scatter!(ax, xs, ys, color = sign.(coeffs), colormap = cgrad([:blue, :red]),
             colorrange = (-1, 1), markersize = 0.35, markerspace = :data, strokecolor = :black, strokewidth = 1.5)
end

display(fig)

## Save
save("Theory/benzene.png", fig)