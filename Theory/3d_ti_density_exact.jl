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
const C = 0.021325
const D1 = -1.0
const D2 = 11.0
const a = 9.4

const V0 = -0.2
const λ = 20

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
            Htot[idx, idx] .= hk + V0 * (exp(-(n - 1) * a / λ) + exp(-(Nz - n) * a / λ)) * I
      end
      
      for n in 1:Nz-1
            Htot[(4n-3):(4n), (4n+1):(4n+4)] .= T
      end
      
      return Hermitian(Htot)
end

function compute_dos(E_grid, k_grid, E_m_k, Gamma)
    Nk = length(k_grid)
    NE = length(E_grid)
    N_bands = size(E_m_k, 1)
    
    dos = zeros(NE)
    dk = k_grid[2] - k_grid[1]
    weights = fill(dk, Nk)
    weights[1] /= 2.0
    weights[end] /= 2.0
    prefactor = 1.0 / (2.0 * pi)
    
    for ie in 1:NE
        E = E_grid[ie]
        sum_k = 0.0
        for m in 2Nz+1:N_bands
            for i in 1:Nk
                k_i = k_grid[i]
                E_mk = E_m_k[m, i]
                gaussian = exp(-(E - E_mk)^2 / (2 * Gamma^2)) / (Gamma * sqrt(2 * pi))
                sum_k += k_i * gaussian * weights[i]
            end
        end
        dos[ie] = prefactor * sum_k
    end
    return dos
end

function calculate_density_from_sum(μ, T, energy_grid, g_upper, energy_weights)
    # Boltzmann constant in eV/K
    kB = 8.617333e-5
    
    # Absolute zero safety check
    if T < 1e-4
        T = 1e-4
    end
    
    n_calc = 0.0
    NE = length(energy_grid)
    
    for j in 1:NE
        E = energy_grid[j]
        Ω_j = energy_weights[j]
        g_j = g_upper[j]
        
        # Calculate Fermi-Dirac distribution factor safely to avoid numerical overflow
        x = (E - μ) / (kB * T)
        f_FD = 1.0 / (1.0 + exp(x))
        
        # Discrete integration step accumulator
        n_calc += Ω_j * g_j * f_FD
    end
    
    return n_calc
end

function solve_chemical_potential(T, target_density, energy_grid, g_upper, energy_weights)
    # Define conservative search boundaries for μ [eV]
    # μ must lie somewhere within or slightly outside your energy grid boundaries
    μ_lower = energy_grid[1]
    μ_upper = energy_grid[end]
    
    tolerance = 1e-8  # Precision limit for the root finder
    max_iter = 100
    
    μ_guess = 0.0
    n_current = 0.0
    
    for iter in 1:max_iter
        μ_guess = (μ_lower + μ_upper) / 2.0
        n_current = calculate_density_from_sum(μ_guess, T, energy_grid, g_upper, energy_weights)
        
        if abs(n_current - target_density) < tolerance
            return μ_guess, n_current
        end
        
        if n_current < target_density
            μ_lower = μ_guess
        else
            μ_upper = μ_guess
        end
    end
    
    return μ_guess, n_current # Return best guess if max iterations are reached
end

## Calculate
E_grid = collect(-2.0:0.005:4.0)
k_grid = collect(0.0:0.0005:0.2)   # k·p model is only valid for small k near Γ
Gamma = 0.005


Nk = length(k_grid)
N_bands = 4 * Nz
E_m_k = zeros(N_bands, Nk)

for i in 1:Nk
    k_val = k_grid[i]
    H_k = H([k_val, 0.0])
    E_m_k[:, i] .= eigvals(H_k)
end

dos_results = compute_dos(E_grid, k_grid, E_m_k, Gamma)

NE = length(E_grid)

ΔE = E_grid[2] - E_grid[1]
energy_weights = fill(ΔE, NE)
energy_weights[1] /= 2.0
energy_weights[end] /= 2.0

ns = 1.5e-3 # Angstrom^-2, surface carrier density per surface
nb = 5.0e-7 # Angstrom^-3, volume carrier density
target_density = 2ns + Nz * a * nb

temperatures = [5.0, 50.0, 150.0, 300.0] # Kelvin
println("Temperature (K) | Chemical Potential μ (eV)")
println("-------------------------------------------")

for T in temperatures
    μ_T, n = solve_chemical_potential(T, target_density, E_grid, dos_results, energy_weights)
    @printf("%15.1f | %24.5f | %10.5f\n ", T, μ_T, n)
end

## Plot

fig = Figure(resolution = (1200, 600), fontsize = 18)

ax_bands = Axis(fig[1, 1],
    title = "Electronic Band Structure",
    xlabel = "Wavevector k (Å⁻¹)",
    ylabel = "Energy E (eV)"
)

ax_dos = Axis(fig[1, 2],
    title = "Density of States (DOS)",
    xlabel = "DOS (eV⁻¹ · Å⁻²)",
    ylabel = "Energy E (eV)"
)
linkyaxes!(ax_bands, ax_dos)

for m in 1:N_bands
    lines!(ax_bands, k_grid, E_m_k[m, :], color = :blue, linewidth = 1.5)
end

lines!(ax_dos, dos_results, E_grid, color = :crimson, linewidth = 2.5)

tightlimits!(ax_bands)
ylims!(ax_bands, -0.5, 0.7)

display(fig)

## Save
save("Theory/bi2se3_$(Nz)ql_dos_exact.png", fig)