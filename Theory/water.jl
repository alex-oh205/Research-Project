## Initialize
using Quantica, GLMakie, LinearAlgebra, Optim, Printf

b = 1.0; angle = deg2rad(52.25)

E_experimental = [-2.37, -1.37, -1.08, -0.93]
initial_params = [-1.5, -1.2, -1.0, -0.4, -0.3]

## Functions
function solve(p, doplot=false)
    eos, eop, eh, ts, tp = p

    pos_O  = [0.0, 0.0, 0.0]
    pos_H1 = [b * cos(angle),  b * sin(angle), 0.0]
    pos_H2 = [b * cos(angle), -b * sin(angle), 0.0]

    sub_O  = sublat(pos_O,  name = :O)
    sub_H1 = sublat(pos_H1, name = :H1)
    sub_H2 = sublat(pos_H2, name = :H2)

    lat = lattice(sub_O, sub_H1, sub_H2)
    if doplot
        display(qplot(lat, title = "H2O Geometric Structure"))
    end

    model_onsite =
        onsite(r -> Diagonal([eos, eop, eop, eop]); sublats = :O) +
        onsite(r -> Diagonal([eh]); sublats = (:H1, :H2))

    model_hopping = plusadjoint(hopping(
        (r, dr) -> begin
            d = normalize(dr)
            [ts tp * d[1] tp * d[2] 0.0]
        end;
        sublats = :O => (:H1, :H2),
    ))

    H = hamiltonian(lat, model_onsite + model_hopping, orbitals = (4, 1, 1))
    Hmat = Matrix(H[()])

    return solve_ham(Hmat)
end

function solve_ham(H)
    evals, evecs = eigen(Hermitian(H))

    evecs = real.(evecs)

    for j in 1:length(evals)
        if sum(evecs[2:4, j]) < 0
            evecs[:, j] .= .-evecs[:, j]
        end
    end

    return evals, evecs
end

function cost_function(p)
    evals, evecs = solve(p)
    return sum((evals[1:4] .- E_experimental).^2)
end

## Optimize
result = optimize(cost_function, initial_params, NelderMead(), Optim.Options(g_tol = 1e-6, iterations = 1000))

best_params = Optim.minimizer(result)

println("Params:")
for p in best_params
    @printf("%7.3f Ry\n", p)
end

## Final solution

# Manual Hamiltonian
# eos, eop, eh, ts, tp = best_params
# H_raw = [
#     eos 0   0   0   ts            ts;
#     0   eop 0   0   tp*cos(angle) tp*cos(angle);
#     0   0   eop 0   tp*sin(angle) -tp*sin(angle);
#     0   0   0   eop 0             0;
#     0   0   0   0   eh            0;
#     0   0   0   0   0             eh
# ]

# evals, evecs = solve_ham(H_raw)

evals, evecs = solve(best_params, true)

println("  n   eigval   eigvec")
for n in 1:length(evals)
    fmt = Printf.Format(" %2i  %7.3f   [" * "%6.3f " ^ (length(evals) - 1) * "%6.3f]\n")
    Printf.format(stdout, fmt, n, evals[n], evecs[:, n]...)
end