using Quantica, GLMakie, LinearAlgebra, Printf

## Lattice
d_nn = √3 / 2
A1 = (-0.5,  0.5,  0.5)
A2 = ( 0.5, -0.5,  0.5)
A3 = ( 0.5,  0.5, -0.5)

sub = sublat((0.0, 0.0, 0.0), name = :Li)

lat = lattice(sub, bravais = (A1, A2, A3))

display(qplot(lat, title = "Lithium Unit Cell"))

## Model
es = 4.5
t = -1.4

model = onsite(es) + hopping(t; range = d_nn * 1.01)

H = hamiltonian(lat, model)

## Path Plot
# cartesian
kΓ = (0.0, 0.0, 0.0)
kH = (2pi, 0.0, 0.0)
kP = (pi, pi, pi)

# specified in Bloch phases = reciprocal lattice coordinates * 2π
ϕΓ = (0.0, 0.0, 0.0) .* 2pi
ϕH = (-0.5, 0.5, 0.5) .* 2pi
ϕP = (0.25, 0.25, 0.25) .* 2pi

d_ΓH = norm(kH .- kΓ)
d_ΓP = norm(kP .- kΓ)

npts = 40

bcut = bands(H, subdiv((-d_ΓP, 0, d_ΓH), npts); mapping = (-d_ΓP, 0, d_ΓH) => (ϕP, ϕΓ, ϕH))

fig = qplot(bcut, title = "Lithium Band Structure", figure = (size = (900, 600),))

ax = content(fig[1, 1])
ax.xticks = ([-d_ΓP, 0, d_ΓH], ["P", "Γ", "H"])
ax.xlabel = "k"

vlines!(ax, [0], color = :black, linewidth = 1)
hlines!(ax, [0], color = :black, linewidth = 1, linestyle = :dash)

Label(fig[0, :], "Lithium Band Structure",
      fontsize = 22, font = :bold, halign = :center, tellwidth = false)

display(fig)

## Save
save("Theory/lithium.png", fig)