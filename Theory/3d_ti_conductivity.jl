using GLMakie, LinearAlgebra, Printf

# m = 0.28
# A1 = 2.2
# A2 = 4.1
# B1 = 10.0
# B2 = 56.6
# C = -0.0068
# D1 = 1.3
# D2 = 19.6

const m = 0.145
const A1 = 2.2
const A2 = 3.15
const B1 = 6.8
const B2 = 42.5
const C = 0
const D1 = -1.0
const D2 = 11.0
const a = 9.4
const e_charge = 1.602176634e-19  # C
const hbar_SI = 1.054571817e-34   # J*s
const e2_over_hbar = e_charge^2 / hbar_SI  # Siemens (Ω⁻¹); the 2D quantum-conductance unit

Nz = 10

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

function dH0_dkx(kx)
    dϵ = 2 * D2 * kx
    dM = -2 * B2 * kx
    dH = dϵ * I + [
        dM   0    0   A2;
        0  -dM   A2   0;
        0    0   dM   0;
        0    0    0 -dM
    ]
    return dH
end

function VelocityOperator_X(kx)
    v_tot = zeros(ComplexF64, 4Nz, 4Nz)
    vk = dH0_dkx(kx)
    for n in 1:Nz
        idx = (4n-3):(4n)
        v_tot[idx, idx] .= vk
    end
    return v_tot
end

# --- Exact Kubo Kernel at a specific kx point ---
function kubo_kernel_at_kx(kx, mu, T, eta)
    Hk = H([kx, 0.0])
    dec = eigen(Hermitian(Matrix(Hk)))
    Es = dec.values
    U = dec.vectors
    N = length(Es)

    Vx = U' * VelocityOperator_X(kx) * U

    kB = 8.617333e-5
    beta = 1.0 / (kB * T)

    f(E) = 1.0 / (exp((E - mu) * beta) + 1.0)
    df_dE(E) = -beta / (4 * cosh((E - mu) * beta / 2)^2)

    kernel_sum = 0.0
    for m in 1:N
        fm = f(Es[m])
        for n in 1:N
            fn = f(Es[n])
            dE = Es[m] - Es[n]

            vx_mn = Vx[m, n]
            vx_nm = Vx[n, m]

            if abs(dE) < 1e-6
                weight_xx = -df_dE(Es[m])
            else
                weight_xx = (fn - fm) / dE
            end
            lorentzian_delta = (1.0 / pi) * (eta / (dE^2 + eta^2))
            kernel_sum += real(vx_mn * vx_nm) * weight_xx * lorentzian_delta
        end
    end
    return kernel_sum
end

function calculate_macroscopic_conductivity(mu, T_kelvin, eta; nk=100, kmax=0.1)
    kxs = range(1e-5, kmax, length=nk)
    dk = step(kxs)
    
    total_conduct = 0.0
    
    for kx in kxs
        local_kernel = kubo_kernel_at_kx(kx, mu, T_kelvin, eta)
        
        total_conduct += kx * local_kernel * dk / 2.0
    end
    
    area_normalization = 1.0 / (2*pi)
    return total_conduct * area_normalization * e2_over_hbar  # Ω⁻¹ (Siemens)
end

## Eta parameters
η0 = 0.005
γac = 5e-5
γop = 0.01
θ = 90
eta(T) = η0 + γac * T + γop / (exp(θ / T) - 1)

## Plot conductivity vs mu
mus = [0.03, 0.2, 0.3, 0.4]
σ_10K = [calculate_macroscopic_conductivity(mu, 10.0, eta(10.0)) for mu in mus]
σ_300K = [calculate_macroscopic_conductivity(mu, 300.0, eta(300.0)) for mu in mus]

fig, ax, _ = scatterlines(mus, σ_10K, label = "10 K", linewidth = 2)
scatterlines!(ax, mus, σ_300K, label = "300 K", linewidth = 2)
ax.xlabel = "μ (eV)"
ax.ylabel = "Conductivity (Ω⁻¹)"
ax.title = "Bi2Se3 Conductivity vs Chemical Potential"
axislegend(ax)

display(fig)

## Plot: conductivity vs temperature at fixed mu
Ts = range(5.0, 50.0, length=30)
mus_T = [0.2, 0.35]

σ_vs_T = [[calculate_macroscopic_conductivity(mu, Tk, eta(Tk)) for Tk in Ts] for mu in mus_T]

fig2, ax2, _ = lines(Ts, σ_vs_T[1], label = "μ = 0.2 eV", linewidth = 2)
for i in 2:length(mus_T)
    lines!(ax2, Ts, σ_vs_T[i], label = "μ = $(mus_T[i]) eV", linewidth = 2)
end
ax2.xlabel = "Temperature (K)"
ax2.ylabel = "Conductivity (Ω⁻¹)"
ax2.title = "Bi2Se3 Conductivity vs Temperature"
axislegend(ax2)

display(fig2)

## Plot: conductivity vs thickness (number of layers) at fixed mu, T
Nz_range = 10:20:100
mu_thick = 0.2
T_thick = 300.0
eta_thick = eta(T_thick)
Nz_saved = Nz

σ_vs_Nz = Float64[]*
for Nz_val in Nz_range
    global Nz = Nz_val
    push!(σ_vs_Nz, calculate_macroscopic_conductivity(mu_thick, T_thick, eta_thick))
end
global Nz = Nz_saved

fig3, ax3, _ = scatterlines(collect(Nz_range), σ_vs_Nz, linewidth = 2)
ax3.xlabel = "Number of layers (QL)"
ax3.ylabel = "Conductivity (Ω⁻¹)"
ax3.title = "Bi2Se3 Conductivity vs Thickness (μ = 0.2 eV, T = 300 K)"

display(fig3)