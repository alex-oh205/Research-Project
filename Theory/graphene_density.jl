using LinearAlgebra, GLMakie

A = [0, 0]
B = [0, √3/3]
B_zz = [1/2, √3/6]
B_ac = [√3/3, 0]

R1 = [1, 0]
R2 = [cos(π/3), sin(π/3)]
R = [1, 0]

τ_ac1 = [-√3/6, 1/2]
τ_ac2 = [√3/6, 1/2]

## Infinite
function h_graphene(k, t=1.0)
    a1 = [1.0, 0.0]
    a2 = [0.5, sqrt(3)/2]
    f = -t * (1 + exp(1im * (k ⋅ a1)) + exp(1im * (k ⋅ a2)))
    return [
        0       f;
        conj(f) 0
    ]
end

## Nanoribbon
function h_zz(k, N, t=1.0)
    H = zeros(ComplexF64, 2N, 2N)
    for i in 1:(2N - 1)
        if i % 2 == 1
            H[i, i + 1] = t * (exp(1im * k ⋅ (B_zz - A)) + exp(1im * k ⋅ (B_zz - A - R)))
        else
            H[i, i + 1] = t
        end
    end
    return Hermitian(H)
end

function h_ac(k, N, t=1.0)
    H = zeros(ComplexF64, 2N, 2N)
    for i in 1:(2N - 1)
        if i % 2 == 1
            H[i, i + 1] = t * exp(1im * k ⋅ (B_ac - A))
            if i + 3 <= 2N
                H[i, i + 3] = t * exp(1im * k ⋅ τ_ac1)
            end
        else
            H[i, i + 1] = t * exp(1im * k ⋅ τ_ac2)
        end
    end
    return Hermitian(H)
end

## KPM
function kpm_reconstruct(μ::Vector{Float64}, Es)
    M = length(μ)
    return map(Es) do E
        T0, T1 = 1.0, E
        s = μ[1] * T0 + 2 * μ[2] * T1
        for m in 2:M - 1
            T2 = 2 * E * T1 - T0
            s += 2 * μ[m + 1] * T2
            T0, T1 = T1, T2
        end
        s / (π * sqrt(1 - E^2))
    end
end

function calculate_moments!(μ::Vector{Float64}, Hk, dim::Int, norm::Real)
    A0 = Matrix{ComplexF64}(I, dim, dim)
    A1 = Hk
    μ[1] += real(tr(A0)) / dim / norm
    μ[2] += real(tr(A1)) / dim / norm
    for m in 3:length(μ)
        A2 = 2 * Hk * A1 - A0
        μ[m] += real(tr(A2)) / dim / norm
        A0, A1 = A1, A2
    end
end

function kpm_dos(system::Symbol, M::Int, N::Int=1, t=1.0; nk=400, npts=2000, ϵ=0.05)
    a = 3t / (1 - ϵ/2)
    μ = zeros(Float64, M)

    if system == :bulk
        b1 = 2π .* [1.0, 1/sqrt(3)]
        b2 = 2π .* [1.0, -1/sqrt(3)]
        for i in 1:nk, j in 1:nk
            u, v = i / nk, j / nk
            k = u * b1 + v * b2
            Hk = h_graphene(k, t) ./ a
            calculate_moments!(μ, Hk, 2, nk^2)
        end
    else
        kmax = system == :zigzag ? 2π : system == :armchair ? 2π * sqrt(3)/3 : error("unknown system: $system")
        h_ribbon = system == :zigzag ? h_zz : h_ac
        for i in 1:nk
            k = kmax * i / nk
            H = h_ribbon([k, 0], N, t) ./ a
            calculate_moments!(μ, H, 2N, nk)
        end
    end

    # Jackson kernel
    alpha = pi / (M + 1)
    g = [
        begin
            theta = m * alpha
            ((M - m + 1) * cos(theta) + sin(theta) * (cos(alpha) / sin(alpha))) / (M + 1)
        end for m in 0:M-1
    ]
    μ .*= g

    Es = range(-0.99, 0.99, length=npts)
    ρ = kpm_reconstruct(μ, Es)
    return Es .* a, ρ ./ a
end

N = 10
Es, ρ = kpm_dos(:bulk, 300, N; nk=200)

## Plot
fig = lines(Es, ρ)
display(fig)

## Save
# save("Theory/graphene_dos_zigzag_$N.png", fig)