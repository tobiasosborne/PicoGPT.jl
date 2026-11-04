# Train the character-level model on Tiny Shakespeare.
#
#     julia --project -t auto scripts/train_shakespeare.jl
#
# Writes checkpoints and the loss history to artifacts/shakespeare/checkpoints/ (then run
# scripts/export_book_data.jl). Settings may be overridden by environment variables:
# D H L T B STEPS WARMUP LR LR_MIN DECAY CLIP SEED EVAL_EVERY EVAL_BATCHES. RESUME=1
# continues from checkpoints/latest.bin.

include("common.jl")

BLAS.set_num_threads(1)                 # we parallelise over sequences, with Julia's threads

const S = (D = env("D", 128), H = env("H", 4), L = env("L", 4), T = env("T", 256),
           B = env("B", 32), steps = env("STEPS", 12_000), warmup = env("WARMUP", 200),
           η = env("LR", 1e-3), ηmin = env("LR_MIN", 1e-4), λ = env("DECAY", 0.1),
           clip = env("CLIP", 1.0), seed = env("SEED", 1234),
           eval_every = env("EVAL_EVERY", 250), eval_batches = env("EVAL_BATCHES", 8))

# Checkpoints kept for the book (samples "as training goes"); others are overwritten.
const KEEP = Set([0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000])
evaluate_now(step) = step % S.eval_every == 0 || step in KEEP || step == S.steps

function main()
    text = tiny_shakespeare()
    abc = alphabet(text)
    train, val = shakespeare_split(text, abc)
    c = Config(; V = length(abc), S.D, S.H, S.L, S.T)
    η(t) = warmup_cosine(t; S.η, S.warmup, S.steps, S.ηmin)

    θ = init(c, Float32; rng = Xoshiro(S.seed))
    opt = AdamW(θ; λ = decay(c, S.λ))
    start = 0
    if get(ENV, "RESUME", "0") == "1" && isfile(joinpath(CHECKPOINTS, "latest.bin"))
        ck = load_checkpoint(joinpath(CHECKPOINTS, "latest.bin"))
        θ .= ck.θ; opt.m .= ck.adam[1]; opt.v .= ck.adam[2]; opt.t = ck.t
        start = ck.step
    end
    rng = Xoshiro(S.seed + start)

    # the same evaluation windows every time, so the curves are comparable
    evalset(data, k) = rand(Xoshiro(S.seed + k), 1:length(data) - c.T, S.B * S.eval_batches)
    train_eval, val_eval = evalset(train, 1), evalset(val, 2)

    @printf("Tiny Shakespeare: %d characters, V = %d; %d train / %d val tokens\n",
            length(text), c.V, length(train), length(val))
    @printf("model: D=%d H=%d L=%d T=%d F=%d → p = %d parameters (Float32)\n",
            c.D, c.H, c.L, c.T, c.F, nparams(c))
    @printf("training: %d steps of B=%d sequences (%d tokens/step), AdamW η=%g→%g, λ=%g, clip=%g, %d threads\n",
            S.steps, S.B, S.B * c.T, S.η, S.ηmin, S.λ, S.clip, Threads.nthreads())
    println("uniform guessing: ln V = ", round(log(c.V), digits = 4))
    flush(stdout)

    history = joinpath(CHECKPOINTS, "history.tsv")
    mkpath(CHECKPOINTS)
    if start == 0
        write(history, "step\ttokens\tseconds\tlr\tbatch_loss\ttrain_loss\tval_loss\n")
        write(joinpath(CHECKPOINTS, "settings.txt"), string(S, "\n", c, "\n"))
    end

    best = Inf
    seconds, recent = 0.0, Float64[]
    for t in start:S.steps
        if t > start
            x, y = windows(train, c.T, rand(rng, 1:length(train) - c.T, S.B))
            seconds += @elapsed ℓ = step!(θ, opt, x, y, c; η = η(t), S.clip)
            push!(recent, ℓ)
            if t % 50 == 0
                @printf("step %6d  loss %.4f  η %.2e  %6.0f tok/s  %6.1f min\n",
                        t, sum(recent[max(end - 49, 1):end]) / min(length(recent), 50), η(t),
                        S.B * c.T * (t - start) / seconds, seconds / 60)
                flush(stdout)
            end
        end
        evaluate_now(t) || continue

        ℓtrain = meanloss(θ, train, train_eval, c)
        ℓval = meanloss(θ, val, val_eval, c)
        batch = isempty(recent) ? ℓtrain : sum(recent) / length(recent)
        open(history, "a") do io
            println(io, join((t, t * S.B * c.T, round(seconds, digits = 2), η(t), batch, ℓtrain, ℓval), '\t'))
        end
        empty!(recent)
        @printf("── step %6d  train %.4f  val %.4f%s\n", t, ℓtrain, ℓval, ℓval < best ? "  (best)" : "")
        if t in KEEP || t == S.steps
            save_checkpoint(checkpoint_path(t); c, abc, step = t, θ)
        end
        if ℓval < best
            best = ℓval
            save_checkpoint(joinpath(CHECKPOINTS, "best.bin"); c, abc, step = t, θ)
        end
        save_checkpoint(joinpath(CHECKPOINTS, "latest.bin"); c, abc, step = t, θ, opt)
        if t % 1000 == 0 || t == S.steps
            prompt = encode("\n", abc)
            sample = decode(generate(θ, c, prompt, 240; τ = 0.8, rng = Xoshiro(t)), abc)
            println(replace(sample, r"^\n" => ""), "\n")
        end
        flush(stdout)
    end
    @printf("done: %d steps in %.1f min; best val loss %.4f\n", S.steps, seconds / 60, best)
end

main()
