# Parent (1 pass, coarse) → child (2 passes, gate from 0, fine) against scratch (2 passes, fine).
# Run: OPENBLAS_NUM_THREADS=1 julia --project -t2 grow/doors/run.jl [nseeds] [maxsteps]
include("doors.jl")
nseeds = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 3
maxsteps = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 6000
const B = 64
c = config(parse(Int, get(ENV, "DOORS_D", "24")), parse(Int, get(ENV, "DOORS_H", "1")), parse(Int, get(ENV, "DOORS_F", "48")))
say0 = println("D, H, F = ", (c.D, c.H, c.F))
out = open(joinpath(@__DIR__, "results.tsv"), "w")
println(out, "seed\tarm\tparent_steps\tparent_acc\tbirth_maxdiff\tgate_grad_fd\tgate_grad_analytic\tzero_shot_g1_acc\tsteps_to_90\tsteps_to_99\tfinal_acc\tfinal_gate\twork_to_99\twork_incl_parent")
curves = open(joinpath(@__DIR__, "curves.tsv"), "w"); println(curves, "seed\tarm\tstep\tloss\tacc")
say(args...) = (println(args...); flush(stdout))
say("parameters: ", nparams(c) + 1, "   chance on fine: 1/4 = 0.25 (loss ln 4 = 1.386) with doors known, perfect 1.0 (loss 0)")
for seed in 1:nseeds
    rng = Xoshiro(seed)
    testf = fine(Xoshiro(1000 + seed), 1000)
    function report(arm, θ0, k; parent_steps = 0, parent_acc = NaN, mask = trues(length(θ0)), cap = maxsteps, birth = NaN, gfd = NaN, gan = NaN, zs = NaN)
        θ, cv = train!(copy(θ0), fine, c, k; rng, test = testf, steps = cap, stop = 0.995, trainmask = mask)
        for (s, l, a) in cv; println(curves, seed, "\t", arm, "\t", s, "\t", l, "\t", a); end
        s90, s99 = reach(cv, 0.90), reach(cv, 0.99)
        w99 = ismissing(s99) ? missing : s99 * B * k
        wall = ismissing(w99) ? missing : w99 + parent_steps * B
        println(out, join((seed, arm, parent_steps, parent_acc, birth, gfd, gan, zs, s90, s99, cv[end][3], θ[end], w99, wall), "\t")); flush(out)
        say(@sprintf("seed %d %-22s steps→90%%: %s  →99%%: %s  final acc %.3f  gate %.3f  work→99%%: %s (with parent %s)", seed, arm, s90, s99, cv[end][3], θ[end], w99, wall))
        θ
    end
    report("scratch-2pass", newθ(c, rng; g = 1.0), 2)
    report("scratch-1pass", newθ(c, rng; g = 0.0), 1)
    for (pname, world) in (("coarse", coarse), ("hops", hops))
        testp = world(Xoshiro(2000 + seed), 1000)
        θp, cvp = train!(newθ(c, rng; g = 0.0), world, c, 1; rng, test = testp, steps = maxsteps, stop = 0.995)
        psteps, pacc = cvp[end][1], cvp[end][3]
        # birth: the child (k = 2, g = 0) equals the parent (k = 1) on both worlds
        birth = maximum(maximum(abs, looped(θp, d..., c, 2)[2] - looped(θp, d..., c, 1)[2]) for d in (testp, testf))
        # gate gradient at birth on the fine world, by central differences and by the pullback
        f(g) = looped(vcat(θp[1:end-1], g), testf..., c, 2)[1]
        gfd = (f(1e-5) - f(-1e-5)) / 2e-5
        θ̄ = looped(θp, testf..., c, 2)[3]()
        gan = θ̄[end]
        zs = evaluate(vcat(θp[1:end-1], 1.0), testf, c, 2)[2]      # open the gate fully, no training
        a0 = evaluate(θp, testf, c, 2)[2]
        say(@sprintf("seed %d parent-%s: %d steps, acc %.3f | birth max|ΔZ| %.1e | fine acc at birth %.3f | ∂ℓ/∂g FD %+.5f analytic %+.5f | rest of ∇ at birth via pass 2: |θ̄| %.3f | zero-shot g=1 fine acc %.3f",
            seed, pname, psteps, pacc, birth, a0, gfd, gan, norm(θ̄[1:end-1]), zs))
        if pname == "hops"       # after pass 1 on the fine world, is the key decodable by the parent's own readout?
            Z = looped(θp, testf..., c, 1)[2]
            say(@sprintf("   parent-hops pass-1 readout on fine queries names a key: %.3f", sum(argmax(Z[:, n]) in KEYS for n in 1:1000) / 1000))
        end
        kw = (; parent_steps = psteps, parent_acc = pacc, birth, gfd, gan, zs)
        report("child-of-$pname", θp, 2; kw...)
        gateonly = falses(length(θp)); gateonly[end] = true
        report("child-of-$pname-gate-only", θp, 2; mask = gateonly, cap = 300, kw...)
    end
end
close(out); close(curves)
