# Truncating the specimen and its relatives:   julia --project -t 3 grow/truncate/specimen.jl
#
# Trains D = 4, 8, 16 on "to be or not to be" (the specimen's recipe, seed 1), then cuts the
# stream to rank r by each rule and reports the loss before and after 50 steps of retraining,
# the spectra, and the intertwiner defect. Writes specimen.tsv, spectra.tsv, defects.tsv.

include("truncate.jl")

const corpus = "to be or not to be"
const abc = alphabet(corpus)
const data = encode(corpus, abc)
const floor_ = (2log(2) + 3log(3)) / 80
const hyper = (η = 2e-2, ηmin = 2e-3, warmup = 20, steps = 200, λ = 0.01, clip = 1.0)
const RULES = ((:pca, :logit), (:pcaperp, :logit), (:jac, :logit), (:balanced, :logit), (:balanced, :fisher))
label(rule, metric) = rule in (:pca, :pcaperp, :random) ? string(rule) : "$rule/$metric"

function train(c, x, y; seed = 1, steps = hyper.steps)
    θ = init(c; rng = Xoshiro(seed))
    opt = AdamW(θ; λ = decay(c, hyper.λ))
    for t in 1:steps
        step!(θ, opt, x, y, c; η = warmup_cosine(t; hyper.η, hyper.warmup, steps, hyper.ηmin), hyper.clip)
    end
    θ, first(loss(θ, x, y, c))
end

rows, spec, defs = String[], String[], String[]
for D in (4, 8, 16)
    c = Config(V = length(abc), D = D, H = 2, L = 1, T = 8, F = 4D)
    x, y = windows(data, c.T)
    θ, ℓ = train(c, x, y)
    steps = hyper.steps
    while ℓ - floor_ > 1e-3 && steps < 3200          # the narrow model may need longer
        steps *= 2
        θ, ℓ = train(c, x, y; steps)
    end
    Zfull, = tgpt(θ, x, c)
    @printf("\n## D = %d: p = %d, %d steps, loss %.6f, floor %.6f, excess %.1e\n", D, nparams(c), steps, ℓ, floor_, ℓ - floor_)
    stats = Dict(m => streamstats(θ, x, c; metric = m) for m in (:logit, :fisher))
    C = pooled(stats[:logit].C)
    for (name, λ) in (("C", spectrum(C)), ("M/logit", spectrum(pooled(stats[:logit].M))),
                      ("M/fisher", spectrum(pooled(stats[:fisher].M))),
                      ("hankel/logit", hankel(C, pooled(stats[:logit].M))),
                      ("hankel/fisher", hankel(C, pooled(stats[:fisher].M))))
        λ = max.(λ, 0) ./ maximum(λ)
        println(rpad(name, 14), join((@sprintf("%.1e", v) for v in λ), " "))
        push!(spec, join((D, name, λ...), '\t'))
    end

    println("\n| r | ", join((label(r...) for r in RULES), " | "), " | random (mean of 5) |   loss − floor: before → after 50 steps")
    for r in 1:D
        cells = String[]
        function trial(rule, metric, rng)
            q = choose(rule, r, stats[metric]; rng)
            before = cutloss(θ, x, y, c, q)
            kl = meankl(Zfull, first(tgpt(θ, x, c, q)))
            θ′, ls = retrain(θ, x, y, c, q)
            (; before, kl, after = last(ls), q, θ′)
        end
        for (rule, metric) in RULES
            t = trial(rule, metric, Xoshiro(0))
            push!(cells, @sprintf("%.4f → %.4f", t.before - floor_, t.after - floor_))
            push!(rows, join((D, r, label(rule, metric), t.before, t.kl, t.after), '\t'))
            if r in (D ÷ 2, D - 2) && D > 4 || D == 4 && r == 2
                for (tag, θ₁) in (("cut", θ), ("retrained", t.θ′))
                    d = defects(θ, θ₁, x, c, t.q)
                    push!(defs, join((D, r, label(rule, metric), tag, d.loc..., d.klocal, d.drift..., d.kdrift), '\t'))
                end
            end
        end
        ts = [trial(:random, :logit, Xoshiro(s)) for s in 1:5]
        mean(f) = sum(f, ts) / length(ts)
        push!(cells, @sprintf("%.4f → %.4f", mean(t -> t.before) - floor_, mean(t -> t.after) - floor_))
        push!(rows, join((D, r, "random", mean(t -> t.before), mean(t -> t.kl), mean(t -> t.after)), '\t'))
        if r in (D ÷ 2, D - 2) && D > 4 || D == 4 && r == 2
            for (tag, θ₁) in (("cut", θ), ("retrained", ts[1].θ′))
                d = defects(θ, θ₁, x, c, ts[1].q)
                push!(defs, join((D, r, "random", tag, d.loc..., d.klocal, d.drift..., d.kdrift), '\t'))
            end
        end
        println("| ", r, " | ", join(cells, " | "), " |")
    end
    sc = smallcount(c, D ÷ 2)
    println("cut model at r = ", D ÷ 2, ": ", sc.learned, " weights + ", sc.fixed, " fixed numbers; a native model of width ",
            D ÷ 2, " has ", D ÷ 2 ≥ 2 && iseven(D ÷ 2) ? nparams(Config(V = c.V, D = D ÷ 2, H = 2, L = 1, T = 8, F = 2D)) : "—")
end

# native models trained from scratch at each width, for comparison (best of 3 seeds)
println("\n## native widths, from scratch, 200 steps (loss − floor, best of seeds 1–3)")
for D in (2, 4, 6, 8)
    c = Config(V = length(abc), D = D, H = 2, L = 1, T = 8, F = 4D)
    x, y = windows(data, c.T)
    ls = [last(train(c, x, y; seed)) for seed in 1:3]
    @printf("D = %d: p = %d, %.4f  (seeds: %s)\n", D, nparams(c), minimum(ls) - floor_, join((@sprintf("%.4f", l - floor_) for l in ls), ", "))
    push!(rows, join((D, D, "native", minimum(ls), "", ""), '\t'))
end

here = @__DIR__
write(joinpath(here, "specimen.tsv"), "D\tr\trule\tloss_cut\tkl_to_full\tloss_retrained\n" * join(rows, "\n") * "\n")
write(joinpath(here, "spectra.tsv"), "D\tquantity\teigenvalues, relative to the largest\n" * join(spec, "\n") * "\n")
write(joinpath(here, "defects.tsv"), "D\tr\trule\tweights\tlocal: site 1, 2, 3; KL at readout\tdrift: site 1, 2, 3; KL at readout\n" * join(defs, "\n") * "\n")
@printf("\nfloor = %.6f nats\n", floor_)
