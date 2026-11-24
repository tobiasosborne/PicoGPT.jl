# The same cut on the 810,049-parameter Shakespeare model (read-only checkpoint, not retrained
# from scratch):       julia --project -t 3 grow/truncate/shakespeare.jl
#
# Statistics (C, M) come from training text, losses from validation text. Writes shakespeare.tsv.

include("truncate.jl")

const HOME_REPO = "/home/tobias/Projects/PicoGPT.jl/artifacts"
const CK = joinpath(HOME_REPO, "shakespeare", "checkpoints", "best.bin")
const TXT = joinpath(HOME_REPO, "data", "tinyshakespeare.txt")
isfile(CK) && isfile(TXT) || (println("checkpoint or text missing; skipped"); exit())

function load_checkpoint(path)                    # the format of scripts/common.jl
    open(path) do io
        String(read(io, 8)) == "PICOGPT1" || error("not a PicoGPT checkpoint")
        V, D, H, L, T, F, step, t = ltoh.(read!(io, Vector{Int64}(undef, 8)))
        c = Config(V, D, H, L, T, F)
        abc = collect(String(read(io, ltoh(read(io, Int64)))))
        k = ltoh(read(io, Int64))
        ltoh(read(io, Int64)) == nparams(c) || error("invalid parameter count")
        θ = copy(reinterpret(Float32, ltoh.(read!(io, Vector{UInt32}(undef, nparams(c))))))
        (; c, abc, step, θ)
    end
end

ck = load_checkpoint(CK)
c, θ = ck.c, ck.θ
data = encode(read(TXT, String), ck.abc)
n = round(Int, 0.9 * length(data))
train, val = data[1:n], data[n+1:end]
xs, ys = windows(train, c.T, rand(Xoshiro(1), 1:length(train) - c.T, 16))     # for the statistics
xv, yv = windows(val, c.T, rand(Xoshiro(2), 1:length(val) - c.T, 24))         # for the losses
xt, yt = windows(train, c.T, rand(Xoshiro(3), 1:length(train) - c.T, 64))     # for retraining
full = cutloss(θ, xv, yv, c)
@printf("Shakespeare: p = %d, step %d, D = %d, L = %d; validation loss on %d windows %.4f\n",
        nparams(c), ck.step, c.D, c.L, size(xv, 2), full)

stats = Dict(m => streamstats(θ, xs, c; metric = m, probes = 6, rng = Xoshiro(4)) for m in (:logit, :fisher))
C = pooled(stats[:logit].C)
rows = String[]
show(name, λ) = (λ = max.(λ, 0) ./ maximum(λ);
    println(rpad(name, 14), "rank ", join((1, 2, 4, 8, 16, 32, 64, 96, 127, 128), " "), ":  ",
            join((@sprintf("%.1e", λ[i]) for i in (1, 2, 4, 8, 16, 32, 64, 96, 127, 128)), " "));
    push!(rows, join(("spectrum", name, λ...), '\t')))
show("C", spectrum(C))
show("M/logit", spectrum(pooled(stats[:logit].M)))
show("M/fisher", spectrum(pooled(stats[:fisher].M)))
show("hankel/logit", hankel(C, pooled(stats[:logit].M)))
show("hankel/fisher", hankel(C, pooled(stats[:fisher].M)))

const RULES = ((:pca, :logit), (:jac, :logit), (:jac, :fisher), (:balanced, :logit), (:balanced, :fisher), (:random, :logit))
label(rule, metric) = rule in (:pca, :random) ? string(rule) : "$rule/$metric"
println("\nvalidation loss of the cut model, no retraining (full model ", round(full, digits = 4), ", uniform ", round(log(c.V), digits = 3), ")")
println("| r | ", join((label(r...) for r in RULES), " | "), " |")
for r in (8, 16, 32, 48, 64, 96, 112, 127)
    ls = [cutloss(θ, xv, yv, c, choose(rule, r, stats[metric]; R = Float32, rng = Xoshiro(5))) for (rule, metric) in RULES]
    println("| ", r, " | ", join((@sprintf("%.4f", l) for l in ls), " | "), " |")
    foreach(((rm, l),) -> push!(rows, join(("cut", label(rm...), r, l), '\t')), zip(RULES, ls))
    flush(stdout)
end

println("\nafter 30 AdamW steps (batch 16, η = 3e-4) with the cut fixed: validation loss before → after")
for r in (32, 64), (rule, metric) in ((:pca, :logit), (:balanced, :fisher), (:random, :logit))
    q = choose(rule, r, stats[metric]; R = Float32, rng = Xoshiro(5))
    θ′, = retrain(θ, xt, yt, c, q; steps = 30, η = 3f-4, batch = 16, λ = 0.1, rng = Xoshiro(6))
    b, a = cutloss(θ, xv, yv, c, q), cutloss(θ′, xv, yv, c, q)
    @printf("r = %3d  %-16s %.4f → %.4f\n", r, label(rule, metric), b, a)
    push!(rows, join(("retrained", label(rule, metric), r, b, a), '\t'))
    flush(stdout)
end
write(joinpath(@__DIR__, "shakespeare.tsv"), join(rows, "\n") * "\n")
