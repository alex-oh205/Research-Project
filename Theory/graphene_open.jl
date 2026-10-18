using GLMakie, LinearAlgebra, Printf

## Lattice
A = [0, 0]
B = [0, √3/3]
B_zz = [1/2, √3/6]
B_ac = [√3/3, 0]

R1 = [1, 0]
R2 = [cos(π/3), sin(π/3)]

τ_ac1 = [-√3/6, 1/2]
τ_ac2 = [√3/6, 1/2]

t = -1.0
N = 1

m = 2 * N
R = [1, 0]

## Hamiltonian
function h_p(k)
      hop = t * (exp(1im * k ⋅ (B_zz - A)) + exp(1im * k ⋅ (B_zz - A - R1)) + exp(1im * k ⋅ (B_zz - A - R2)))
      H = [
            0.0  hop;
            hop  0.0
      ]
      return H
end

function h_zz(k)
      H = zeros(ComplexF64, m, m)
      for i in 1:(m - 1)
            if i % 2 == 1
                  H[i, i + 1] = t * (exp(1im * k ⋅ (B_zz - A)) + exp(1im * k ⋅ (B_zz - A - R)))
            else
                  H[i, i + 1] = t
            end
      end
      return H
end

function h_ac(k)
      H = zeros(ComplexF64, m, m)
      for i in 1:(m - 1)
            if i % 2 == 1
                  H[i, i + 1] = t * exp(1im * k ⋅ (B_ac - A))
                  if i + 3 <= m
                        H[i, i + 3] = t * exp(1im * k ⋅ τ_ac1)
                  end
            else
                  H[i, i + 1] = t * exp(1im * k ⋅ τ_ac2)
            end
      end
      return H
end

## Band Structure
K = 4π/3
M = 2√3π/3
kmax = K

show(stdout, "text/plain", eigen(Hermitian(h_zz([0.999 * π, 0]))).vectors)

x = range(-kmax, kmax, length = 300)
Y = []
for kx in x
      global Y
      evals, _ = eigen(Hermitian(h_zz([kx, 0])))
      if isempty(Y)
            Y = evals
      else
            Y = hcat(Y, evals)
      end
end

blue = fill(:blue, length(x))
fig = series(x, Y, color = blue)
fig.axis.xticks = ([-kmax, 0, kmax], ["-2√3π/3", "0", "2√3π/3"])
display(fig)

## Save
save("Theory/graphene_periodic_zigzag.png", fig)