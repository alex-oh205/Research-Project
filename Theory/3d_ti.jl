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

ϵ0(k) = C + D1 * k[3]^2 + D2 * (k[1]^2 + k[2]^2)

M(k) = m - B1 * k[3]^2 - B2 * (k[1]^2 + k[2]^2)

H(k) = begin
      kp = complex(k[1:2]...)
      km = conj(kp)
      Mval = M(k)
      Hermitian(ϵ0(k) * I + [
            Mval A1 * k[3] 0       A2 * km;
            0    -Mval     A2 * km 0;
            0    0         Mval    -A1 * k[3];
            0    0         0       -Mval
      ])
end

kmax = 0.2
x = range(-kmax, kmax, length = 300)
Y1 = []
Y2 = []
colors1 = []
colors2 = []
colors3 = []
colors4 = []
for kx in x
      global Y1, Y2, colors1, colors2, colors3, colors4
      evals1, evecs = eigen(Hermitian(H([kx, 0, 0])))
      evals2, _ = eigen(Hermitian(H([kx, 0, π/9.4/2])))
      if isempty(Y1)
            Y1 = evals1
            Y2 = evals2
            colors1 = abs2.(evecs[:, 1])
            colors2 = abs2.(evecs[:, 2])
            colors3 = abs2.(evecs[:, 3])
            colors4 = abs2.(evecs[:, 4])
      else
            Y1 = hcat(Y1, evals1)
            Y2 = hcat(Y2, evals2)
            colors1 = hcat(colors1, abs2.(evecs[:, 1]))
            colors2 = hcat(colors2, abs2.(evecs[:, 2]))
            colors3 = hcat(colors3, abs2.(evecs[:, 3]))
            colors4 = hcat(colors4, abs2.(evecs[:, 4]))
      end
end

## Plot
o = 3

fig = scatterlines(x, Y1[1, :], color = colors1[o, :], colormap = cgrad([RGBAf(1, 1, 1, 0), RGBAf(0, 0, 1, 1)]), colorrange = (0, 1), linewidth = 2, markersize = 0)
scatterlines!(fig.axis, x, Y1[2, :], color = colors2[o, :], colormap = cgrad([RGBAf(1, 1, 1, 0), RGBAf(0, 0, 1, 1)]), colorrange = (0, 1), linewidth = 2, markersize = 0)
scatterlines!(fig.axis, x, Y1[3, :], color = colors3[o, :], colormap = cgrad([RGBAf(1, 1, 1, 0), RGBAf(0, 0, 1, 1)]), colorrange = (0, 1), linewidth = 2, markersize = 0)
scatterlines!(fig.axis, x, Y1[4, :], color = colors4[o, :], colormap = cgrad([RGBAf(1, 1, 1, 0), RGBAf(0, 0, 1, 1)]), colorrange = (0, 1), linewidth = 2, markersize = 0)
# fig = series(x, Y1, color = fill(:blue, length(x)), linewidth = 2)
# series!(fig.axis, x, Y2, color = fill(:blue, length(x)), linewidth = 2)
fig.axis.xticks = ([-kmax, 0, kmax], ["-0.2", "0", "0.2"])
fig.axis.xlabel = "kx (1/Å)"
fig.axis.ylabel = "Energy (eV)"
fig.axis.title = "Bi2Se3 Band Structure P1z+, ↓ (Bi) Contribution"
xlims!(fig.axis, -kmax, kmax)
ylims!(fig.axis, -0.3, 0.5)

display(fig)

## Save
save("Theory/p1z+_down.png", fig)