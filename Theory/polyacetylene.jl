using Quantica, GLMakie, LinearAlgebra, Printf

## Lattice
sub1 = sublat((-1/4, 1/2), name = :C1)
sub2 = sublat((1/4, -1/2), name = :C2)

lat = lattice(sub1, sub2, bravais = ((1.0, 0.0),))

display(qplot(lat, title = "Polyacetylene Unit Cell"))

## Model
ΔE = 0.4
t = -1.0
δt = -0.3

t1 = t + δt
t2 = t - δt

model = onsite(ΔE; sublats = :C1) +
        onsite(-ΔE; sublats = :C2) +
        plusadjoint(hopping((r, dr) -> dr[1] > 0 ? t1 : t2; sublats = :C1 => :C2, range = 1.2))

H = hamiltonian(lat, model)

## Path Plot
phis = range(0, π, length = 300)
b = bands(H, phis)

fig = qplot(b, title = "Polyacetylene Band Structure", figure = (size = (900, 600),))

ax = content(fig[1, 1])
ax.xticks = ([0, π], ["0", "π"])
ax.xlabel = "k"

# vlines!(ax, [0], color = :black, linewidth = 1)
hlines!(ax, [0], color = :black, linewidth = 1, linestyle = :dash)

Label(fig[0, :], "Polyacetylene Band Structure",
      fontsize = 22, font = :bold, halign = :center, tellwidth = false)

display(fig)

## Save
save("Theory/polyacetylene.png", fig)