# Grow the specimen D 4 → 8 (and F 16 → 32) against training at full size from scratch.
#     julia --project=. -t 1 grow/lifts/grow_experiment.jl
# Work = Σ over updates of the number of parameters of the model being updated.
include(joinpath(@__DIR__, "Lifts.jl"))
using .Lifts, PicoGPT, Random, LinearAlgebra, Printf, Statistics

corpus = "to be or not to be"
abc = alphabet(corpus)
x, y = windows(encode(corpus, abc), 8)
const ℓfloor = (2log(2) + 3log(3)) / 80
const target = 1.01ℓfloor
const η, cap, seeds = 2e-2, 3000, 1:8
c4 = Config(V = 7, D = 4, H = 2, L = 1, T = 8, F = 16)      # 315 parameters
c8 = Config(V = 7, D = 8, H = 2, L = 1, T = 8, F = 32)      # 1007: the specimen's shape
net4, net8 = Net(c4), Net(c8)
net8d = Net(c8, dup(4, 2))                                   # the specimen's shape with the duplicated code
p4, p8 = nparams(c4), nparams(c8)

grow(θ, net; asym, rng) = lift_mlp(lift_width(θ, net; asym, rng)...; F′ = 32, rng)
J, _ = tangent(θ -> grow(θ, net4; asym = 0.0, rng = Xoshiro(0))[1], p4)   # the parent's directions
N = nullspace(J')                                            # the new directions: 1007 − 315 = 692
@assert size(N, 2) == p8 - p4

rows = []
run!(arm, n1, seed, s1, s2, wall, last, gap = 0.0) =
    push!(rows, (; arm, n1, seed, s1, s2, work = s1 * p4 + s2 * p8, wall, reached = last ≤ target, last, gap))

function experiment()
    for seed in seeds
        # full size from scratch: PicoGPT's code, and the duplicated code the child must use
        for (arm, net) in (("scratch D8, PicoGPT code", net8), ("scratch D8, duplicated code", net8d))
            θ = init(c8; rng = Xoshiro(seed))
            wall = @elapsed l = train!(θ, net, x, y; steps = cap, η, target)
            run!(arm, 0, seed, 0, length(l) - 1, wall, l[end])
        end
        # the parent alone, never lifted
        θ = init(c4; rng = Xoshiro(seed))
        wall = @elapsed l = train!(θ, net4, x, y; steps = cap, η, target)
        run!("parent D4 alone", 0, seed, length(l) - 1, 0, wall, l[end])
        # train the parent n1 steps, lift, continue
        for n1 in (25, 50, 100, 200)
            θ4 = init(c4; rng = Xoshiro(seed))
            w1 = @elapsed l1 = train!(θ4, net4, x, y; steps = n1, η, target)
            s1 = length(l1) - 1
            for (arm, asym, basis) in (("grown, plain duplication", 0.0, nothing),
                                       ("grown, asymmetric readers", 1.0, nothing),
                                       ("grown, parent frozen", 1.0, N))
                θ′, net′ = grow(θ4, net4; asym, rng = Xoshiro(1000 + seed))
                w2 = @elapsed l2 = train!(θ′, net′, x, y; steps = cap, η, target, basis)
                run!(arm, n1, seed, s1, length(l2) - 1, w1 + w2, l2[end], abs(l2[1] - l1[end]))
            end
        end
    end
end

experiment(); empty!(rows)          # once to compile, then for the record
experiment()

open(joinpath(@__DIR__, "out", "growth.tsv"), "w") do io
    println(io, "arm\tn1\tseed\tsteps_parent\tsteps_child\twork\twall_s\treached\tfinal_loss\tbirth_gap")
    for r in rows
        @printf(io, "%s\t%d\t%d\t%d\t%d\t%d\t%.4f\t%s\t%.6f\t%.1e\n", r.arm, r.n1, r.seed, r.s1, r.s2, r.work, r.wall, r.reached, r.last, r.gap)
    end
end

@printf("floor %.6f, target %.6f; η = %g, %d seeds, cap %d updates per phase; p = %d → %d\n", ℓfloor, target, η, length(seeds), cap, p4, p8)
@printf("%-28s %4s  %7s  %-17s %-17s %-22s %-14s %s\n", "arm", "n1", "reached", "parent updates", "child updates", "work/1000 med [min,max]", "wall ms (med)", "birth gap")
for key in unique((r.arm, r.n1) for r in rows)
    rs = [r for r in rows if (r.arm, r.n1) == key]
    med(f) = median(f.(rs))
    @printf("%-28s %4d  %3d/%-3d  %-17s %-17s %6.0f [%4.0f, %5.0f]    %7.0f        %.0e\n", key[1], key[2], count(r -> r.reached, rs), length(rs),
            @sprintf("%.0f", med(r -> r.s1)), @sprintf("%.0f [%d, %d]", med(r -> r.s2), minimum(r -> r.s2, rs), maximum(r -> r.s2, rs)),
            med(r -> r.work) / 1000, minimum(r -> r.work, rs) / 1000, maximum(r -> r.work, rs) / 1000, 1000med(r -> r.wall), maximum(r -> r.gap, rs))
end
