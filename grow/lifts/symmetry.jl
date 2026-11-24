# What training does to the duplicated coordinates after a width lift D 4 → 8.
#     julia --project=. -t 1 grow/lifts/symmetry.jl
# stream: the fraction of the stream's norm (after the block) outside the duplicated subspace;
# params: the fraction of |θ′| outside the fixed subspace of the copy-swap symmetry.
include(joinpath(@__DIR__, "Lifts.jl"))
using .Lifts, PicoGPT, Random, LinearAlgebra, Printf, Statistics

corpus = "to be or not to be"
x, y = windows(encode(corpus, alphabet(corpus)), 8)
const ℓfloor = (2log(2) + 3log(3)) / 80
c4 = Config(V = 7, D = 4, H = 2, L = 1, T = 8, F = 16)
net4 = Net(c4)
U = dup(4, 2)
at = [0, 1, 2, 5, 10, 20, 50, 100, 200, 400]
function grow(θ, net; asym, rng, mlp)
    θ′, net′ = lift_width(θ, net; asym, rng)
    mlp == :none ? (θ′, net′) : lift_mlp(θ′, net′; F′ = 32, mode = mlp, rng)
end

arms = [("width duplicated, F = 16 kept (a fixed point of the copy swap)", 0.0, 0.0, :none),
        ("width duplicated, MLP units duplicated 16 → 32 (also a fixed point)", 0.0, 0.0, :dup),
        ("width duplicated, 16 fresh MLP units (their random W1 rows read asymmetrically)", 0.0, 0.0, :fresh),
        ("width duplicated + noise 1e-3 on θ′ (inexact), F = 16", 0.0, 1e-3, :none),
        ("asymmetric readers ×0.1, 16 fresh MLP units", 0.1, 0.0, :fresh),
        ("asymmetric readers ×1, 16 fresh MLP units", 1.0, 0.0, :fresh)]
println("parent D = 4 trained 100 updates, lifted to D = 8, trained 400 more (η = 2e-2); median over 6 seeds")
for (name, asym, noise, mlp) in arms
    S, Pm, Ls = zeros(6, length(at)), zeros(6, length(at)), zeros(6, length(at))
    for seed in 1:6
        θ4 = init(c4; rng = Xoshiro(seed))
        train!(θ4, net4, x, y; steps = 100, η = 2e-2)
        θ′, net′ = grow(θ4, net4; asym, mlp, rng = Xoshiro(1000 + seed))
        θ′ .+= noise .* randn(Xoshiro(2000 + seed), length(θ′))
        each(t, θ) = (i = findfirst(==(t), at); i === nothing || begin
            S[seed, i] = stream_asymmetry(θ, x, net′, U).Xout
            Pm[seed, i] = norm(θ - symmetrise(θ, net′, U)) / norm(θ)
        end)
        l = train!(θ′, net′, x, y; steps = 400, η = 2e-2, each)
        Ls[seed, :] .= l[at .+ 1] ./ ℓfloor .- 1
    end
    println("\n", name)
    @printf("  %-22s %s\n", "updates after the lift", join([@sprintf("%8d", t) for t in at]))
    @printf("  %-22s %s\n", "stream asymmetry", join([@sprintf("%8.1e", v) for v in median(S, dims = 1)]))
    @printf("  %-22s %s\n", "parameter asymmetry", join([@sprintf("%8.1e", v) for v in median(Pm, dims = 1)]))
    @printf("  %-22s %s\n", "loss/floor − 1", join([@sprintf("%8.1e", v) for v in median(Ls, dims = 1)]))
end
