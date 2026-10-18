using GLMakie, LinearAlgebra, Printf

# m = 0.28
# A1 = 2.2
# A2 = 4.1
# B1 = 10.0
# B2 = 56.6
# C = -0.0068
# D1 = 1.3
# D2 = 19.6

m = 0.145
A1 = 2.2
A2 = 3.15
B1 = 6.8
B2 = 42.5
C = 0
D1 = -1.0
D2 = 11.0
a = 9.4

Nz = 100

ϵ0(k) = C + D2 * (k[1]^2 + k[2]^2)
M(k) = m - B2 * (k[1]^2 + k[2]^2)

function H0(k)
      kp = Complex(k[1], k[2])
      km = conj(kp)
      Mval = M(k)
      H_mat = ϵ0(k) * I + [
            Mval  0        0        A2 * km;
            0    -Mval     A2 * km  0;
            0     0        Mval     0;
            0     0        0       -Mval
    ]
    return Hermitian(H_mat)
end

H1 = Hermitian([
      0   A1  0    0;
      0   0   0    0;
      0   0   0   -A1;
      0   0   0    0
])

H2 = [
      D1 - B1  0        0        0;
      0        D1 + B1  0        0;
      0        0        D1 - B1  0;
      0        0        0        D1 + B1
]

h0(k) = H0(k) + (2 / a^2) * H2
T = -1im / (2a) * H1 - (1 / a^2) * H2

function H(k)
      Htot = zeros(ComplexF64, 4Nz, 4Nz)
      hk = Matrix(h0(k))
      
      for n in 1:Nz
            idx = (4n-3):(4n)
            Htot[idx, idx] .= hk
      end
      
      for n in 1:Nz-1
            Htot[(4n-3):(4n), (4n+1):(4n+4)] .= T
      end
      
      return Hermitian(Htot)
end

function kpm_reconstruct(μ::Vector{Float64}, Es)
      M = length(μ)
      return map(Es) do E
            T0, T1 = 1.0, E
            s = μ[1]*T0 + 2*μ[2]*T1
            for m in 2:M-1
                  T2 = 2*E*T1 - T0
                  s += 2*μ[m+1]*T2
                  T0, T1 = T1, T2
            end
            s / (π*sqrt(1-E^2))
      end
end

function kpm_dos(M::Int; nk=400, npts=2000, ϵ=0.05, rmax=0.1)
      bound = maximum(sum(abs.(Matrix(H([rmax, 0.0]))), dims=2))  # Gershgorin bound at worst-case k (r=rmax)

      a = (1 + ϵ/2) * bound

      μ = zeros(Float64, M)
      wsum = 0.0

      for r in range(0, rmax, length=nk)
            Hk = H([r, 0.0]) ./ a
            w = r  # Jacobian for polar-coordinate area element r dr dθ
            wsum += w

            A0 = Matrix{ComplexF64}(I, 4Nz, 4Nz)
            A1 = Hk
            μ[1] += w * real(tr(A0)) / 4Nz
            μ[2] += w * real(tr(A1)) / 4Nz
            for m in 3:M
                  A2 = 2*Hk*A1 - A0
                  μ[m] += w * real(tr(A2)) / 4Nz
                  A0, A1 = A1, A2
            end
      end
      μ ./= wsum

      # Jackson kernel
      alpha = pi / (M + 1)
      g = [
            begin
                  theta = m * alpha
                  ((M-m+1)*cos(theta) + sin(theta)*(cos(alpha) / sin(alpha))) / (M+1)
            end for m in 0:M-1
      ]
      μ .*= g

      Es = range(-0.99, 0.99, length=npts)
      ρ = kpm_reconstruct(μ, Es)
      return Es .* a, ρ ./ a .* (rmax^2/4π) * 10^4
end

Es, ρ = kpm_dos(300; nk=200)

## Plot
fig = lines(Es, ρ)
fig.axis.xlabel = "Energy (eV)"
fig.axis.ylabel = "DOS (states/eV/Å^2)"
fig.axis.title = "Bi2Se3 Density of States ($(Nz) QL)"
xlims!(fig.axis, -0.5, 0.5)

display(fig)

## Save
save("Theory/bi2se3_$(Nz)ql_dos.png", fig)