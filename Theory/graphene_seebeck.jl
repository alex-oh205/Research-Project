using LinearAlgebra, Printf

# Graphene tight-binding parameters
const a_cc = 1.42          # C-C bond length in Å
const t = 2.7              # nearest-neighbor hopping in eV
const e_charge = 1.602176634e-19
const hbar_SI = 1.054571817e-34
const e2_over_hbar = e_charge^2 / hbar_SI
const kB_eV = 8.617333e-5

# Real-space nearest-neighbor vectors (in Å)
const a1 = [sqrt(3) * a_cc / 2, a_cc / 2]
const a2 = [sqrt(3) * a_cc / 2, -a_cc / 2]

function graphene_hamiltonian(k)
    phase1 = exp(1im * (k ⋅ a1))
    phase2 = exp(1im * (k ⋅ a2))
    f = -t * (1.0 + phase1 + phase2)
    return [
        0.0        f;
        conj(f)   0.0
    ]
end

function graphene_velocity_operator(k)
    phase1 = exp(1im * (k ⋅ a1))
    phase2 = exp(1im * (k ⋅ a2))
    df_dkx = -t * 1im * (a1[1] * phase1 + a2[1] * phase2)
    return [
        0.0       df_dkx;
        conj(df_dkx)  0.0
    ]
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
    prefactor = 1.0 / (2π)

    for ie in 1:NE
        E = E_grid[ie]
        sum_k = 0.0
        for m in 1:N_bands
            for i in 1:Nk
                k_i = k_grid[i]
                E_mk = E_m_k[m, i]
                gaussian = exp(-(E - E_mk)^2 / (2 * Gamma^2)) / (Gamma * sqrt(2π))
                sum_k += k_i * gaussian * weights[i]
            end
        end
        dos[ie] = prefactor * sum_k
    end
    return dos
end

function calculate_density_from_sum(μ, T, energy_grid, g_upper, energy_weights)
    if T < 1e-4
        T = 1e-4
    end

    n_calc = 0.0
    for j in 1:length(energy_grid)
        E = energy_grid[j]
        Ω_j = energy_weights[j]
        g_j = g_upper[j]
        x = (E - μ) / (kB_eV * T)
        f_FD = 1.0 / (1.0 + exp(x))
        n_calc += Ω_j * g_j * f_FD
    end
    return n_calc
end

function solve_chemical_potential(T, target_density, energy_grid, g_upper, energy_weights)
    μ_lower = energy_grid[1]
    μ_upper = energy_grid[end]
    tolerance = 1e-8
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

    return μ_guess
end

function kubo_kernel(kx, μ, T, eta, weight_kind)
    Hk = graphene_hamiltonian([kx, 0.0])
    dec = eigen(Hermitian(Matrix(Hk)))
    Es = dec.values
    U = dec.vectors

    Vx = U' * graphene_velocity_operator([kx, 0.0]) * U

    beta = 1.0 / (kB_eV * T)
    f(E) = 1.0 / (1.0 + exp((E - μ) * beta))
    df_dE(E) = -beta / (4 * cosh((E - μ) * beta / 2)^2)

    kernel_sum = 0.0
    for m in 1:length(Es)
        fm = f(Es[m])
        for n in 1:length(Es)
            fn = f(Es[n])
            dE = Es[m] - Es[n]
            vx_mn = Vx[m, n]
            vx_nm = Vx[n, m]

            if abs(dE) < 1e-6
                weight_xx = -df_dE(Es[m])
            else
                weight_xx = (fn - fm) / dE
            end

            lorentzian_delta = (1.0 / π) * (eta / (dE^2 + eta^2))
            energy_weight = weight_kind == :seebeck ? (Es[m] + Es[n]) / 2 - μ : 1.0
            kernel_sum += real(vx_mn * vx_nm) * weight_xx * lorentzian_delta * energy_weight
        end
    end
    return kernel_sum
end

function calculate_conductivity_and_seebeck(μ, T, eta; nk=200, kmax=0.8)
    kxs = range(-kmax, kmax, length=nk)
    dk = step(kxs)

    total_cond = 0.0
    total_seebeck = 0.0
    for kx in kxs
        local_cond_kernel = kubo_kernel(kx, μ, T, eta, :conductivity)
        local_seebeck_kernel = kubo_kernel(kx, μ, T, eta, :seebeck)
        total_cond += local_cond_kernel * dk
        total_seebeck += local_seebeck_kernel * dk
    end

    area_normalization = 1.0 / (2π)
    conductivity = total_cond * area_normalization * e2_over_hbar

    if abs(conductivity) < 1e-30
        seebeck = NaN
    else
        seebeck = -1.0 / T * total_seebeck * area_normalization * e2_over_hbar / conductivity * 1e6
    end

    return conductivity, seebeck
end

# Energy and k-space grids
E_grid = collect(-2.0:0.005:2.0)
k_grid = collect(-0.8:0.002:0.8)
Gamma = 0.02

Nk = length(k_grid)
N_bands = 2
E_m_k = zeros(N_bands, Nk)

for i in 1:Nk
    k_val = k_grid[i]
    H_k = graphene_hamiltonian([k_val, 0.0])
    E_m_k[:, i] .= eigvals(H_k)
end

dos_results = compute_dos(E_grid, k_grid, E_m_k, Gamma)

NE = length(E_grid)
ΔE = E_grid[2] - E_grid[1]
energy_weights = fill(ΔE, NE)
energy_weights[1] /= 2.0
energy_weights[end] /= 2.0

target_density = 1.0e-4  # Å^-2, roughly 1e13 cm^-2

temperatures = [100.0, 200.0, 300.0, 400.0]
println("Temperature (K) | μ (eV) | Seebeck (μV/K)")
println("----------------------------------------")

seebeck_vs_T = Float64[]

for T in temperatures
    μ_T = solve_chemical_potential(T, target_density, E_grid, dos_results, energy_weights)
    conductivity_T, seebeck_T = calculate_conductivity_and_seebeck(μ_T, T, 0.01; nk=200, kmax=0.8)
    push!(seebeck_vs_T, seebeck_T)
    @printf("%15.1f | %10.5f | %14.3f\n", T, μ_T, seebeck_T)
end

println("\nCompleted graphene Seebeck calculation.")
println("Results are printed above; if you want a plotted figure, install GLMakie and restore the plotting section.")