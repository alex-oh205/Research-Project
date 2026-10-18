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
C = 0.0215
D1 = -1.0
D2 = 11.0
a = 9.4

Nz = 20

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

## Calculate
kmax = 0.2
x = range(-kmax, kmax, length = 300)

spinup_weight(v) = sum(abs2, v[41:44])

eigs = [eigen(H([kx, 0])) for kx in x]
Y = reduce(hcat, [e.values for e in eigs])
W = reduce(hcat, [spinup_weight.(eachcol(e.vectors)) for e in eigs])

## Plot
cmap = cgrad([RGBAf(0, 0, 1, 0), RGBAf(0, 0, 1, 1)])

# fig, ax, _ = series(x, Y, color = fill(:blue, size(Y, 1)), linewidth = 2)
fig, ax, _ = lines(x, Y[1, :], color = W[1, :], colormap = cmap, colorrange = (0, 1), linewidth = 2)
for b in 2:80
      lines!(ax, x, Y[b, :], color = W[b, :], colormap = cmap, colorrange = (0, 1), linewidth = 2)
end
ax.xticks = ([-kmax, 0, kmax], ["-0.2", "0", "0.2"])
ax.xlabel = "kx (1/Å)"
ax.ylabel = "Energy (eV)"
ax.title = "Bi2Se3 Thin Film Band Structure Weighted Middle Layer Contribution ($(Nz) QL)"
xlims!(ax, -kmax, kmax)
ylims!(ax, -0.3, 0.5)

display(fig)

## Save
save("Theory/bi2se3_$(Nz)ql_layermiddle.png", fig)