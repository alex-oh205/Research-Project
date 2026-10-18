using LinearAlgebra, Printf, GLMakie

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

const V0 = -0.2
const λ = 20
const ns_surface = 1.5e-3 # 1.5e-3 = 1.5e13 cm^-2
const nb_volume = 0 # 1e-5 = 1e19 cm^-3
const qw = 0

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

function H(N, k)
      Htot = zeros(ComplexF64, 4N, 4N)
      hk = Matrix(h0(k))
      
      for n in 1:N
            idx = (4n-3):(4n)
            if qw == 0
                Htot[idx, idx] .= hk + V0 * (exp(-(n - 1) * a / λ) + exp(-(N - n) * a / λ)) * I
            else
                Htot[idx, idx] .= hk + ((n <= qw || n > N - qw) ? V0 : 0) * I
            end
      end
      
      for n in 1:N-1
            Htot[(4n-3):(4n), (4n+1):(4n+4)] .= T
      end
      
      return Hermitian(Htot)
end

function compute_dos(E_grid, k_grid, E_m_k, Gamma)
    Nk = length(k_grid)
    NE = length(E_grid)
    N_bands = size(E_m_k, 1)
    n_layers = N_bands ÷ 4
    
    dos = zeros(NE)
    dk = k_grid[2] - k_grid[1]
    weights = fill(dk, Nk)
    weights[1] /= 2.0
    weights[end] /= 2.0
    prefactor = 1.0 / (2.0 * pi)
    
    for ie in 1:NE
        E = E_grid[ie]
        sum_k = 0.0
        for m in 2*n_layers+1:N_bands
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
            return μ_guess
        end
        
        if n_current < target_density
            μ_lower = μ_guess
        else
            μ_upper = μ_guess
        end
    end
    
    return μ_guess # Return best guess if max iterations are reached
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

function VelocityOperator_X(N, kx)
    v_tot = zeros(ComplexF64, 4N, 4N)
    vk = dH0_dkx(kx)
    for n in 1:N
        idx = (4n-3):(4n)
        v_tot[idx, idx] .= vk
    end
    return v_tot
end

function kubo_kernel(Nz, kx, mu, T, eta, weight_kind)
    Hk = H(Nz, [kx, 0.0])
    dec = eigen(Hermitian(Matrix(Hk)))
    Es = dec.values
    U = dec.vectors
    Nstates = size(U, 1)
    start_band = Nstates ÷ 2 + 1

    Vx = U' * VelocityOperator_X(Nz, kx) * U

    kB = 8.617333e-5
    beta = 1.0 / (kB * T)

    f(E) = 1.0 / (exp((E - mu) * beta) + 1.0)
    df_dE(E) = -beta / (4 * cosh((E - mu) * beta / 2)^2)

    kernel_sum = 0.0
    for m in start_band:Nstates
        fm = f(Es[m])
        for n in start_band:Nstates
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

            if weight_kind == :seebeck
                energy_weight = (Es[m] + Es[n]) / 2 - mu
            else
                energy_weight = 1.0
            end

            kernel_sum += real(vx_mn * vx_nm) * weight_xx * lorentzian_delta * energy_weight
        end
    end
    return kernel_sum
end

function calculate_macroscopic_conductivity(N, mu, T_kelvin, eta; nk=100, kmax=0.1)
    kxs = range(1e-5, kmax, length=nk)
    dk = step(kxs)
    
    total = 0.0
    
    for kx in kxs
        local_kernel = kubo_kernel(N, kx, mu, T_kelvin, eta, :conductivity)
        
        total += kx * local_kernel * dk
    end
    
    area_normalization = 1.0 / (2*pi)
    return total * area_normalization * e2_over_hbar  # Ω⁻¹ (Siemens)
end

function calculate_conductivity_and_seebeck(N, mu, T_kelvin, eta; nk=100, kmax=0.1)
    kxs = range(1e-5, kmax, length=nk)
    dk = step(kxs)

    total_cond = 0.0
    total_seebeck = 0.0

    for kx in kxs
        local_cond_kernel = kubo_kernel(N, kx, mu, T_kelvin, eta, :conductivity)
        local_seebeck_kernel = kubo_kernel(N, kx, mu, T_kelvin, eta, :seebeck)

        total_cond += kx * local_cond_kernel * dk
        total_seebeck += kx * local_seebeck_kernel * dk
    end

    area_normalization = 1.0 / (2*pi)
    conductivity = total_cond * area_normalization * e2_over_hbar

    if abs(conductivity) < 1e-30
        seebeck = NaN
    else
        seebeck = -1 / T_kelvin * total_seebeck * area_normalization * e2_over_hbar / conductivity * 1e6
    end

    return conductivity, seebeck
end

function calculate_macroscopic_seebeck(N, mu, T_kelvin, eta; nk=100, kmax=0.1)
    conductivity, seebeck = calculate_conductivity_and_seebeck(N, mu, T_kelvin, eta; nk=nk, kmax=kmax)
    return seebeck
end

## Eta parameters
const kB_ev = 8.617333e-5  # eV/K

const η_0   = 0.002      # Residual low-T bulk baseline (eV)
const α_ac  = 1.2e-5       # Acoustic phonon slope (eV/K)

# 1. Mediated through conducting bulk (Turns on gradually below ~50 QL)
const A_bulk          = 200   # Amplitude of bulk-mediated cross-talk (eV·Å) - 100 matches my experiment
const p_power         = 2     # Power law exponent for gradual rise

function eta(T::Float64)
    # Temperature-Dependent Phonon Background
    η_ph = η_0 + (α_ac * T)
    
    # Physical thickness of the film in Angstroms
    thickness = Nz * a
    
    # Gradual bulk-mediated inter-surface scattering
    η_mediated = A_bulk / (thickness^p_power)
    
    # Total combined scalar scattering rate
    return η_ph + η_mediated
end

## Calculate μs
E_grid = collect(-0.3:0.005:1.0)
k_grid = collect(0.0:0.0005:0.2)   # k·p model is only valid for small k near Γ
Gamma = 0.005

Nk = length(k_grid)

Ts = range(300, 400, length=3)
mus = Dict()

for N in [5, 10, 16, 30, 50, 70, 100]
    N_bands = 4N
    E_m_k = zeros(N_bands, Nk)

    for i in 1:Nk
        k_val = k_grid[i]
        H_k = H(N, [k_val, 0.0])
        E_m_k[:, i] .= eigvals(H_k)
    end

    dos_results = compute_dos(E_grid, k_grid, E_m_k, Gamma)

    NE = length(E_grid)

    ΔE = E_grid[2] - E_grid[1]
    energy_weights = fill(ΔE, NE)
    energy_weights[1] /= 2.0
    energy_weights[end] /= 2.0

    target_density = 2ns_surface + N * a * nb_volume

    # mus[N] = Dict(T => solve_chemical_potential(T, target_density, E_grid, dos_results, energy_weights) for T in Ts)
    mus[N] = solve_chemical_potential(0, target_density, E_grid, dos_results, energy_weights)
end

## Thickness sweep at fixed temperature
thicknesses = [5, 10, 16, 30, 50, 70, 100]
T_eval = 350.0

sheet_resistance_vs_thickness = Float64[]
sigma_vs_thickness = Float64[]
seebeck_vs_thickness = Float64[]
power_factor_vs_thickness = Float64[]

for N in thicknesses
    global Nz = N
    # mu_T = mus[N][T_eval]
    mu_T = mus[N]
    sigma_T, seebeck_T = calculate_conductivity_and_seebeck(N, mu_T, T_eval, eta(T_eval))
    sheet_resistance = isfinite(sigma_T) && sigma_T > 0 ? 1.0 / sigma_T : NaN
    power_factor = isfinite(sigma_T) && isfinite(seebeck_T) ? (seebeck_T^2) * sigma_T / N : NaN

    push!(sheet_resistance_vs_thickness, sheet_resistance)
    push!(sigma_vs_thickness, sigma_T / N)
    push!(seebeck_vs_thickness, seebeck_T)
    push!(power_factor_vs_thickness, power_factor)
end

# selected_thicknesses = [5, 10, 50, 100]
# temp_sweep_fig = Figure(size = (1600, 900))
# ax_seeb_temp = Axis(temp_sweep_fig[1, 1], xlabel = "Temperature (K)", ylabel = "Seebeck Coefficient (μV/K)", title = "Seebeck Coefficient vs Temperature")
# ax_sheet_temp = Axis(temp_sweep_fig[1, 2], xlabel = "Temperature (K)", ylabel = "Sheet Resistance (Ω/sq)", title = "Sheet Resistance vs Temperature")
# ax_cond_temp = Axis(temp_sweep_fig[2, 1], xlabel = "Temperature (K)", ylabel = "Conductivity (Ω⁻¹QL⁻¹)", title = "Conductivity vs Temperature")
# ax_pf_temp = Axis(temp_sweep_fig[2, 2], xlabel = "Temperature (K)", ylabel = "Power Factor (S²σ)", title = "Power Factor vs Temperature")

# temp_colors = Dict(5 => :royalblue, 10 => :darkorange, 50 => :seagreen, 100 => :purple)

# for N in selected_thicknesses
#     global Nz = N
#     sigma_vs_temp = Float64[]
#     seebeck_vs_temp = Float64[]
#     sheet_resistance_vs_temp = Float64[]
#     power_factor_vs_temp = Float64[]

#     for T in Ts
#         mu_T = mus[N][T]
#         sigma_T, seebeck_T = calculate_conductivity_and_seebeck(N, mu_T, T, eta(T))
#         sheet_resistance = isfinite(sigma_T) && sigma_T > 0 ? 1.0 / sigma_T : NaN
#         power_factor = isfinite(sigma_T) && isfinite(seebeck_T) ? (seebeck_T^2) * sigma_T / N : NaN

#         push!(sigma_vs_temp, sigma_T / N)
#         push!(seebeck_vs_temp, seebeck_T)
#         push!(sheet_resistance_vs_temp, sheet_resistance)
#         push!(power_factor_vs_temp, power_factor)
#     end

#     color = temp_colors[N]
#     lines!(ax_seeb_temp, Ts, seebeck_vs_temp, linewidth = 2, color = color, label = "$(N) QL")
#     scatter!(ax_seeb_temp, Ts, seebeck_vs_temp, color = color, markersize = 8)

#     lines!(ax_sheet_temp, Ts, sheet_resistance_vs_temp, linewidth = 2, color = color, label = "$(N) QL")
#     scatter!(ax_sheet_temp, Ts, sheet_resistance_vs_temp, color = color, markersize = 8)

#     lines!(ax_cond_temp, Ts, sigma_vs_temp, linewidth = 2, color = color, label = "$(N) QL")
#     scatter!(ax_cond_temp, Ts, sigma_vs_temp, color = color, markersize = 8)

#     lines!(ax_pf_temp, Ts, power_factor_vs_temp, linewidth = 2, color = color, label = "$(N) QL")
#     scatter!(ax_pf_temp, Ts, power_factor_vs_temp, color = color, markersize = 8)
# end

# axislegend(ax_seeb_temp; position = :rt)
# axislegend(ax_sheet_temp; position = :rt)
# axislegend(ax_cond_temp; position = :rt)
# axislegend(ax_pf_temp; position = :rt)

# output_path_temp = joinpath(@__DIR__, "temperature_dependence_selected_thicknesses.png")
# save(output_path_temp, temp_sweep_fig)
# @info "Saved temperature plot to $(output_path_temp)"

# display(temp_sweep_fig)

csv_output_path = joinpath(@__DIR__, "thickness_dependence_summary_A_bulk_200_const_mu.csv")
open(csv_output_path, "w") do io
    println(io, "# parameter,V0,$V0")
    println(io, "# parameter,ns,$ns_surface")
    println(io, "# parameter,nb,$nb_volume")
    println(io, "# parameter,A_bulk,$A_bulk")
    println(io, "# parameter,qw,$qw")
    println(io, "Thickness_QL,Conductivity,Seebeck_uV_per_K,Sheet_Resistance,Power_Factor")
    for (N, sigma, seeb, sheet, pf) in zip(thicknesses, sigma_vs_thickness, seebeck_vs_thickness, sheet_resistance_vs_thickness, power_factor_vs_thickness)
        println(io, "$N,$sigma,$seeb,$sheet,$pf")
    end
end
@info "Saved CSV data to $(csv_output_path)"

thickness_fig = Figure(size = (1400, 800))
ax_seeb_thickness = Axis(thickness_fig[1, 1], xlabel = "Thickness (QL)", ylabel = "Seebeck Coefficient (μV/K)", title = "Seebeck Coefficient vs Thickness")
ax_sheet_thickness = Axis(thickness_fig[1, 2], xlabel = "Thickness (QL)", ylabel = "Sheet Resistance (Ω/sq)", title = "Sheet Resistance vs Thickness")
ax_cond_thickness = Axis(thickness_fig[2, 1], xlabel = "Thickness (QL)", ylabel = "Conductivity (Ω⁻¹QL⁻¹)", title = "Conductivity vs Thickness")
ax_pf_thickness = Axis(thickness_fig[2, 2], xlabel = "Thickness (QL)", ylabel = "Power Factor (S²σ)", title = "Power Factor vs Thickness")

lines!(ax_seeb_thickness, thicknesses, seebeck_vs_thickness, linewidth = 2, color = :darkorange)
scatter!(ax_seeb_thickness, thicknesses, seebeck_vs_thickness, color = :darkorange, markersize = 8)

lines!(ax_sheet_thickness, thicknesses, sheet_resistance_vs_thickness, linewidth = 2, color = :purple)
scatter!(ax_sheet_thickness, thicknesses, sheet_resistance_vs_thickness, color = :purple, markersize = 8)

lines!(ax_cond_thickness, thicknesses, sigma_vs_thickness, linewidth = 2, color = :royalblue)
scatter!(ax_cond_thickness, thicknesses, sigma_vs_thickness, color = :royalblue, markersize = 8)

lines!(ax_pf_thickness, thicknesses, power_factor_vs_thickness, linewidth = 2, color = :seagreen)
scatter!(ax_pf_thickness, thicknesses, power_factor_vs_thickness, color = :seagreen, markersize = 8)

if any(isfinite, seebeck_vs_thickness)
    valid_seeb = filter(isfinite, seebeck_vs_thickness)
    if !isempty(valid_seeb)
        y_max = maximum(abs.(valid_seeb))
        ylims!(ax_seeb_thickness, -y_max * 1.1, 0.0)
    end
end

if any(isfinite, sheet_resistance_vs_thickness)
    valid_sheet = filter(isfinite, sheet_resistance_vs_thickness)
    if !isempty(valid_sheet)
        y_max = maximum(valid_sheet)
        ylims!(ax_sheet_thickness, 0.0, y_max * 1.1)
    end
end

if any(isfinite, sigma_vs_thickness)
    valid_sigma = filter(isfinite, sigma_vs_thickness)
    if !isempty(valid_sigma)
        y_max = maximum(valid_sigma)
        ylims!(ax_cond_thickness, 0.0, y_max * 1.1)
    end
end

if any(isfinite, power_factor_vs_thickness)
    valid_pf = filter(isfinite, power_factor_vs_thickness)
    if !isempty(valid_pf)
        y_max = maximum(valid_pf)
        ylims!(ax_pf_thickness, 0.0, y_max * 1.1)
    end
end

output_path = joinpath(@__DIR__, "thickness_dependence_A_bulk_200_const_mu.png")
save(output_path, thickness_fig)
@info "Saved thickness plot to $(output_path)"
display(thickness_fig)