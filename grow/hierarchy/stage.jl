# Lift V=3 → V=6 and compare staged training with scratch.
# usage: julia --project=../.. stage.jl [nseeds] [quick]
include("world.jl")
using PicoGPT, Random, Printf
const T = 8
cfg(V) = Config(; V, D = 8, H = 2, L = 1, T, F = 32)

"Lift: child embedding = class embedding; child logit = class logit (+ a within-class logit that is 0)."
function lift(θ3)
    c3, c6 = cfg(3), cfg(6)
    θ6 = zeros(nparams(c6)); p3 = carve(θ3, c3); p6 = carve(θ6, c6)
    for v in 1:6
        p6.E[:, v] = p3.E[:, cls(v)]; p6.Wout[v, :] = p3.Wout[cls(v), :]; p6.bout[v] = p3.bout[cls(v)]
    end
    for (a, b) in zip(leaves(p6.blocks), leaves(p3.blocks)); a .= b; end
    p6.γf .= p3.γf; p6.βf .= p3.βf
    θ6
end

probs(θ, X, c) = (Z = first(gpt(θ, X, c)); softmax(Z; dims = 1))
evalloss(θ, X, Y, c) = first(loss(θ, X, Y, c))

"Train until test excess over the true process ≤ tol·floor. Returns (steps or nothing, curve)."
function train!(θ, c, draw, Xt, Yt, floor_emp, target; rng, maxsteps, η = 0.01, B = 32, every = 5)
    opt = AdamW(θ); curve = Tuple{Int,Float64}[]
    ex = evalloss(θ, Xt, Yt, c) - floor_emp; push!(curve, (0, ex))
    ex ≤ target && return 0, curve
    for s in 1:maxsteps
        X, Y = draw(B, rng)
        ℓ, back = loss(θ, X, Y, c)
        update!(opt, θ, clip!(back(), 1.0), η)
        if s % every == 0
            ex = evalloss(θ, Xt, Yt, c) - floor_emp; push!(curve, (s, ex))
            ex ≤ target && return s, curve
        end
    end
    nothing, curve
end

function run(; ε, κ, δ, seed, maxsteps = 1500, B = 32, tol = 0.01)
    w = world(; ε, κ, δ); e = entropies(w; T)
    rng = MersenneTwister(1000 + seed)
    Xt, Yt = batch(w, T, 500, MersenneTwister(7))             # fixed test set, 4000 tokens
    ffloor, cfloor = truefloors(w, Xt, Yt)                    # surprise of the true process on the test tokens
    Xc, Yc = cls.(Xt), cls.(Yt)
    fine(B, rng) = batch(w, T, B, rng)
    coarse(B, rng) = ((X, Y) = batch(w, T, B, rng); (cls.(X), cls.(Y)))
    c3, c6 = cfg(3), cfg(6)
    # parent
    θ3 = init(c3; rng)
    s3, _ = train!(θ3, c3, coarse, Xc, Yc, cfloor, tol * e.coarse; rng, maxsteps, B)
    par = evalloss(θ3, Xc, Yc, c3)
    # lift, and the test at birth
    θ6 = lift(θ3)
    p3 = probs(θ3, Xc, c3); p6 = probs(θ6, Xt, c6)
    marg = maximum(abs, [p6[2c-1, n] + p6[2c, n] - p3[c, n] for c in 1:3, n in axes(p3, 2)])
    birth = evalloss(θ6, Xt, Yt, c6)
    s6, _ = train!(θ6, c6, fine, Xt, Yt, ffloor, tol * e.fine; rng, maxsteps, B)
    # scratch
    θs = init(c6; rng)
    ss, _ = train!(θs, c6, fine, Xt, Yt, ffloor, tol * e.fine; rng, maxsteps, B)
    tok = T * B
    wk(p, s) = s === nothing ? NaN : Float64(p) * tok * s
    (; e, s3, s6, ss, marg, birth, par, birthgap = birth - (par + log(2)),
       Wparent = wk(nparams(c3), s3), Wchild = wk(nparams(c6), s6), Wscratch = wk(nparams(c6), ss),
       birth_excess = birth - ffloor)
end

med(x) = (y = sort(filter(!isnan, x)); isempty(y) ? NaN : length(y) < length(x) / 2 + 0.5 ? NaN : y[(length(y) + 1) ÷ 2])

function main()
    nseeds = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 5
    settings = length(ARGS) ≥ 2 ? [(1.0, 1.0, 0.0)] :
        [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 2.0, 0.0), (1.0, 1.0, 0.25), (1.0, 1.0, 0.5),
         (1.0, 1.0, 1.0), (1.0, 1.0, 2.0), (0.0, 1.0, 1.0), (0.0, 1.0, 2.0)]
    out = open(length(ARGS) ≥ 2 ? "quick.tsv" : "results.tsv", "w")
    println(out, "eps\tkappa\tdelta\tseed\tdefect\tsplit\tfine\tcoarse\tsteps_parent\tsteps_child\tsteps_scratch\tW_parent\tW_child\tW_scratch\tbirth_loss\tparent_loss\tbirth_gap\tmarg_err\tbirth_excess")
    for (ε, κ, δ) in settings
        rs = [run(; ε, κ, δ, seed) for seed in 1:nseeds]
        for (seed, r) in enumerate(rs)
            println(out, join((ε, κ, δ, seed, r.e.defect, r.e.split, r.e.fine, r.e.coarse, r.s3, r.s6, r.ss, r.Wparent, r.Wchild, r.Wscratch, r.birth, r.par, r.birthgap, r.marg, r.birth_excess), "\t"))
        end
        flush(out)
        e = rs[1].e
        @printf("ε=%.2f κ=%.2f δ=%.2f defect=%.4f split=%.4f fine=%.4f | steps parent %s child %s scratch %s | median W child/scratch %.3f, (parent+child)/scratch %.3f | max marg err %.1e, max |birth−(parent+ln2)| %.1e, birth excess %.4f\n",
            ε, κ, δ, e.defect, e.split, e.fine, string([r.s3 for r in rs]), string([r.s6 for r in rs]), string([r.ss for r in rs]),
            med([r.Wchild for r in rs]) / med([r.Wscratch for r in rs]), med([r.Wparent + r.Wchild for r in rs]) / med([r.Wscratch for r in rs]),
            maximum(r.marg for r in rs), maximum(abs(r.birthgap) for r in rs), sum(r.birth_excess for r in rs) / nseeds)
        flush(stdout)
    end
    close(out)
end
main()
