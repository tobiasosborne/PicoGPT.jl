# 2 · A two-level tower on the specimen's text.
#
#     julia --project=. -t 3 grow/tower/specimen_tower.jl
#
# T1 is trained on the coarse text (3 classes). T2 predicts the fine text (7 symbols) as
# p(v, b | h) = p(v | ·) q(b | v, h), in three ways:
#
#   two streams, parent frozen   p(v | πh) is T1 on the coarse history, untouched;
#                                q is a small new PicoGPT reading the fine history
#   two streams, parent thawed   the parent is lifted to read fine symbols and trained on
#                                the class term alone; q as before
#   one stream (the lift)        a specimen-shaped model born from T1 by `lift`, all trained
#
# against the 1,007-parameter specimen trained from scratch. Full batch (80 predictions),
# the specimen's recipe, 200 updates per phase.

include("floors.jl")

const SEEDS = 1:5
const EPS = (1e-2, 1e-3)                            # "reached the floor" = within ε of it
xc, yc = CLS[x], CLS[y]                             # the coarse windows: π applied to the fine ones

c3 = Config(V = 3, D = 8, H = 2, L = 1, T = 8, F = 32)        # T1:   939 parameters
c7 = Config(V = 7, D = 8, H = 2, L = 1, T = 8, F = 32)        # the specimen's shape: 1,007
qshapes = (small  = Config(V = 7, D = 4, H = 2, L = 1, T = 8, F = 8),
           medium = Config(V = 7, D = 4, H = 2, L = 1, T = 8, F = 16),
           full   = c7)

F = map(f -> f.value, floors)

# loss-and-gradient closures; each returns ((class term, refinement term), θ̄)
function split_lossgrad(c, x, y; c̄ = 1, r̄ = 1)
    function (θ)
        Z, back_Z = gpt(θ, x, c)
        ℓc, ℓr, back = splitloss(Z, y, CLS)
        (ℓc, ℓr), back_Z(back(c̄, r̄))
    end
end
plain_lossgrad(c, x, y) = θ -> ((ℓ, back) = loss(θ, x, y, c); (ℓ, back()))

# ── checks: pullbacks by finite differences ──────────────────────────────────────────
println("── finite-difference checks (relative error of the hand-written pullbacks) ──")
let rng = Xoshiro(7)
    Z = randn(rng, 7, 12)
    t = rand(rng, 1:7, 12)
    _, _, back = splitloss(Z, t, CLS)
    @printf("splitloss, class term       %.1e\n", gradcheck(Z -> splitloss(Z, t, CLS)[1], Z, back(1, 0)))
    @printf("splitloss, refinement term  %.1e\n", gradcheck(Z -> splitloss(Z, t, CLS)[2], Z, back(0, 1)))
    ℓc, ℓr, _ = splitloss(Z, t, CLS)
    @printf("ℓc + ℓr − crossentropy      %+.1e\n", ℓc + ℓr - first(crossentropy(Z, t)))
    θ = init(c7; rng)
    for (name, c̄, r̄, k) in (("class term", 1, 0, 1), ("refinement term", 0, 1, 2))
        _, θ̄ = split_lossgrad(c7, x, y; c̄, r̄)(θ)
        err = gradcheck(θ -> first(split_lossgrad(c7, x, y)(θ))[k], θ, θ̄)
        @printf("θ̄ of the %-16s through gpt, all 1007 coordinates  %.1e\n", name, err)
    end
end

# ── scratch: the specimen ────────────────────────────────────────────────────────────
scratch = map(SEEDS) do seed
    values, θ = train(init(c7; rng = Xoshiro(seed)), c7, split_lossgrad(c7, x, y))
    (; total = sum.(values), class = first.(values), refine = last.(values), θ)
end

# ── T1 on the coarse text ────────────────────────────────────────────────────────────
parents = map(SEEDS) do seed
    losses, θ = train(init(c3; rng = Xoshiro(seed)), c3, plain_lossgrad(c3, xc, yc))
    (; losses, θ)
end

# ── consistency at birth, to rounding ────────────────────────────────────────────────
println("\n── consistency and the split loss (seed 1), to rounding ──")
θ1 = parents[1].θ
P1 = softmax(first(gpt(θ1, xc, c3)); dims = 1)                    # T1's predictions, 3 × 80
ℓ1 = first(loss(θ1, xc, yc, c3))
sizes = [count(==(v), CLS) for v in 1:3]
let
    # one stream: the lift
    θ2, c2 = lift(θ1, c3, CLS)
    @assert c2 == c7
    Z2 = first(gpt(θ2, x, c7))
    P2 = softmax(Z2; dims = 1)
    @printf("lift:        max |Σ_b T2(v,b|h) − T1(v|πh)|            = %.1e\n", maximum(abs, classprobs(Z2, CLS) - P1))
    @printf("lift:        max |T2(a|h) − T1(cls a|πh)/|cls a||       = %.1e\n", maximum(abs, P2 - P1[CLS, :] ./ sizes[CLS]))
    ℓc, ℓr, _ = splitloss(Z2, vec(y), CLS)
    born = sum(log(sizes[CLS[a]]) for a in y) / length(y)
    @printf("lift:        class term − T1's loss                     = %+.1e\n", ℓc - ℓ1)
    @printf("lift:        refinement term at birth − mean ln|class|  = %+.1e   (= %.6f nats)\n", ℓr - born, born)
    @printf("lift:        ℓc + ℓr − loss(θ2)                         = %+.1e\n", ℓc + ℓr - first(loss(θ2, x, y, c7)))

    # two streams: frozen parent × an arbitrary (random) q
    cq = qshapes.medium
    Zq = first(gpt(init(cq; rng = Xoshiro(99)), x, cq)) .* 3        # any logits will do
    Q = similar(Zq)                                                  # q(a | cls a, h)
    for n in axes(Zq, 2), v in 1:3
        m = findall(==(v), CLS)
        Q[m, n] .= softmax(Zq[m, n])
    end
    J = P1[CLS, :] .* Q                                              # the joint p(v, b | h), 7 × 80
    @printf("two streams: max |Σ_a p(a|h) − 1|                       = %.1e\n", maximum(abs, sum(J, dims = 1) .- 1))
    marginal = [sum(J[a, n] for a in 1:7 if CLS[a] == v) for v in 1:3, n in axes(J, 2)]
    @printf("two streams: max |Σ_b T2(v,b|h) − T1(v|πh)|, random q   = %.1e\n", maximum(abs, marginal - P1))
    total = -sum(log(J[a, n]) for (n, a) in enumerate(y)) / length(y)
    _, ℓr, _ = splitloss(Zq, vec(y), CLS)
    @printf("two streams: −mean log p(a|h) − (T1's loss + ℓr of q)   = %+.1e\n", total - (ℓ1 + ℓr))
end

# ── two streams, parent frozen: train q alone ────────────────────────────────────────
qruns = map(qshapes) do cq
    map(SEEDS) do seed
        values, θ = train(init(cq; rng = Xoshiro(100 + seed)), cq, split_lossgrad(cq, x, y; c̄ = 0, r̄ = 1))
        (; refine = last.(values), θ)
    end
end

# ── two streams, parent thawed: the lifted parent learns the class from the fine history ──
thawed = map(SEEDS) do seed
    θ2, _ = lift(parents[seed].θ, c3, CLS)
    values, θ = train(θ2, c7, split_lossgrad(c7, x, y; c̄ = 1, r̄ = 0))
    (; class = first.(values), θ)
end

# ── one stream: the lift, then train ─────────────────────────────────────────────────
headmask = let m = zeros(Bool, nparams(c7)), p = carve(m, c7)      # embedding and readout only
    p.E .= true; p.Wout .= true; p.bout .= true
    m
end
function grown(mask)
    map(SEEDS) do seed
        θ2, _ = lift(parents[seed].θ, c3, CLS)
        values, θ = train(θ2, c7, split_lossgrad(c7, x, y); mask)
        (; total = sum.(values), class = first.(values), refine = last.(values), θ)
    end
end
lifted_all, lifted_head = grown(nothing), grown(headmask)

# ── report ───────────────────────────────────────────────────────────────────────────
med(v) = (s = sort(collect(v)); isodd(length(s)) ? s[(end+1)÷2] : (s[end÷2] + s[end÷2+1]) / 2)
function reach(curves, target)                      # steps to target + ε, per ε: median and range
    map(EPS) do ε
        ks = [steps_to(c, target + ε) for c in curves]
        any(isnothing, ks) ? "never ($(count(!isnothing, ks))/$(length(ks)))" :
            @sprintf("%d [%d–%d]", med(ks), minimum(ks), maximum(ks))
    end
end
function row(name, p, curves, target)
    r = reach(curves, target)
    @printf("%-44s %5d  %.5f  %.5f  %+.5f  %-14s %-14s\n", name, p, target, med(last.(curves)),
            med(last.(curves)) - target, r...)
end

println("\n── training: loss after 200 updates and updates needed to come within ε of the floor ──")
println("   (median over seeds $(first(SEEDS))–$(last(SEEDS)), [min–max]; ε = $(EPS[1]) and $(EPS[2]))")
@printf("%-44s %5s  %-7s  %-7s  %-8s  %-14s %-14s\n", "phase", "p", "floor", "final", "excess", "to ε=1e-2", "to ε=1e-3")
row("scratch specimen, fine text", nparams(c7), getfield.(scratch, :total), F.fine)
row("  its class term", nparams(c7), getfield.(scratch, :class), F.coarse_fine)
row("  its refinement term", nparams(c7), getfield.(scratch, :refine), F.refine_fine)
row("T1, coarse text", nparams(c3), getfield.(parents, :losses), F.coarse_coarse)
for (name, cq) in pairs(qshapes)
    row("q alone ($name: D=$(cq.D), F=$(cq.F))", nparams(cq), getfield.(qruns[name], :refine), F.refine_fine)
end
row("thawed parent, class term, fine history", nparams(c7), getfield.(thawed, :class), F.coarse_fine)
row("lift, all trained: total", nparams(c7), getfield.(lifted_all, :total), F.fine)
row("  its class term", nparams(c7), getfield.(lifted_all, :class), F.coarse_fine)
row("  its refinement term", nparams(c7), getfield.(lifted_all, :refine), F.refine_fine)
row("lift, only E, Wout, bout trained: total", count(headmask), getfield.(lifted_head, :total), F.fine)

println("\n── whole lineages against scratch: where they end, and the work to get there ──")
println("   work = Σ over phases of (parameters trained) × (updates), in units of the specimen's")
println("   1007 × 200; every update is one full-batch pass over the same 80 predictions")
unit = nparams(c7) * RECIPE.steps
ends(curves...) = med(sum(last.(c) for c in curves))
@printf("%-58s %-9s %-9s %-8s\n", "lineage", "ends at", "its floor", "work")
@printf("%-58s %.5f   %.5f   %.2f\n", "scratch specimen", ends(getfield.(scratch, :total)), F.fine, 1.0)
for (name, cq) in pairs(qshapes)
    @printf("%-58s %.5f   %.5f   %.2f\n", "T1 → frozen parent × q ($name)",
            ends(getfield.(parents, :losses), getfield.(qruns[name], :refine)),
            F.coarse_coarse + F.refine_fine, (nparams(c3) + nparams(cq)) * RECIPE.steps / unit)
end
@printf("%-58s %.5f   %.5f   %.2f\n", "T1 → thawed parent (class term) × q (medium)",
        ends(getfield.(thawed, :class), getfield.(qruns.medium, :refine)), F.fine,
        (nparams(c3) + nparams(c7) + nparams(qshapes.medium)) * RECIPE.steps / unit)
@printf("%-58s %.5f   %.5f   %.2f\n", "T1 → lift → all trained",
        ends(getfield.(lifted_all, :total)), F.fine, (nparams(c3) + nparams(c7)) * RECIPE.steps / unit)
@printf("%-58s %.5f   %.5f   %.2f\n", "T1 → lift → only E, Wout, bout trained",
        ends(getfield.(lifted_head, :total)), F.fine, (nparams(c3) + count(headmask)) * RECIPE.steps / unit)

println("\n   the same, counting updates only until the loss is within ε = 1e-2 of the fine floor")
println("   (T1 charged the updates it needed to come within 1e-2 of its own floor)")
k1 = [steps_to(p.losses, F.coarse_coarse + 1e-2) for p in parents]
ks = [steps_to(s.total, F.fine + 1e-2) for s in scratch]
kl = [steps_to(s.total, F.fine + 1e-2) for s in lifted_all]
@printf("   scratch: %s updates × 1007 = %.0f      T1 → lift: %s × 939 + %s × 1007 = %.0f\n",
        show_steps(med(ks)), med(ks) * 1007, show_steps(med(k1)), show_steps(med(kl)), med(k1 .* 939 .+ kl .* 1007))
println("   per seed, scratch: ", ks, "   T1: ", k1, "   lifted: ", kl)

# ── what thawing does to the consistency condition ───────────────────────────────────
println("\n── consistency after training (seed 1) ──")
let θ = lifted_all[1].θ
    P2 = classprobs(first(gpt(θ, x, c7)), CLS)                     # Σ_b T2(v, b | h), 3 × 80
    @printf("lift, all trained: max |Σ_b T2(v,b|h) − T1(v|πh)|                  = %.3f\n", maximum(abs, P2 - P1))
    # the weaker statement: T1(v | c) = the average of Σ_b T2(v, b | h) over the fine histories h with πh = c
    groups = Dict{Vector{Int},Vector{Int}}()
    for (n, case) in enumerate(cases)
        push!(get!(groups, case.πh, Int[]), n)
    end
    worst = maximum(maximum(abs, vec(sum(P2[:, ns], dims = 2)) ./ length(ns) - P1[:, first(ns)]) for ns in values(groups))
    @printf("lift, all trained: max |mean_{h: πh = c} Σ_b T2(v,b|h) − T1(v|c)|   = %.3f\n", worst)
end
