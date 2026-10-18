using Quantica, GLMakie, LinearAlgebra, Printf

## Lattice
A1 = (cos(π/3), sin(π/3))
A2 = (-cos(π/3), sin(π/3))

sub_A = sublat((0.0, -0.5/√3), name = :A)
sub_B = sublat((0.0,  0.5/√3), name = :B)

lat = lattice(sub_A, sub_B, bravais = (A1, A2))

display(qplot(lat, title = "Graphene Unit Cell"))

## Model
t = -1.0

model = hopping(t; range = 1.01/√3)

H = hamiltonian(lat, model)

evals_Γ, evecs_Γ = eigen(Hermitian(Matrix(H((0, 0)))))
println("Eigenvalues at Γ: ", round.(evals_Γ, digits = 3))

## Path Plot
Amat = bravais_matrix(lat)
Bmat = 2pi * inv(Amat)'
B1 = Bmat[:, 1]
B2 = Bmat[:, 2]

kΓ = [0.0; 0.0]
kK = (B1 - B2) / 3
kM = B1 / 2

k_to_ϕ(k) = Tuple(k' * Amat)

ϕΓ = k_to_ϕ(kΓ)
ϕK = k_to_ϕ(kK)
ϕM = k_to_ϕ(kM)

d_ΓK = norm(kK - kΓ)
d_KM = norm(kM - kK)
d_MΓ = norm(kΓ - kM)

t0 = 0.0
t1 = t0 + d_ΓK
t2 = t1 + d_KM
t3 = t2 + d_MΓ

npts = 40

bcut = bands(H, subdiv((t0, t1, t2, t3), npts); mapping = (t0, t1, t2, t3) => (ϕΓ, ϕK, ϕM, ϕΓ))

fig = qplot(bcut, title = "Graphene Bands: Γ → K → M → Γ", figure = (size = (900, 600),))

ax = content(fig[1, 1])

ax.xticks = ([t0, t1, t2, t3], ["Γ", "K", "M", "Γ"])
ax.xlabel = "k"
ax.yticks = -9:1:9
ax.xticklabelsize = 16
ax.yticklabelsize = 14
ax.xgridvisible = false

vlines!(ax, [t0, t1, t2, t3], color = :black, linewidth = 1)
hlines!(ax, [0], color = :black, linewidth = 1, linestyle = :dash)

Label(fig[0, :], "Graphene Band Structure",
      fontsize = 22, font = :bold, halign = :center, tellwidth = false)

display(fig)

## Save
save("Theory/graphene.png", fig)