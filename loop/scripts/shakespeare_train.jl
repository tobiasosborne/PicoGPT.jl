# Run the Float64 check first, then train the three models in sequence:
# julia --project -t 12 loop/scripts/shakespeare_gradcheck.jl
# for model in one looped stock; do julia --project -t 12 loop/scripts/shakespeare_train.jl "$model" || break; done
# RESUME=1 restores the last completed evaluation checkpoint.
include("shakespeare_common.jl")

const EVAL_HEADER = "step\ttokens\tupdate_seconds\twall_seconds\tlr\tbatch_loss\ttrain_loss\tval_loss"
const UPDATE_HEADER = "step\tpasses\tloss\tgradient_norm"
const KEEP_STEPS = Set([0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000])

evaluate_now(t, S) = t % S.eval_every == 0 || t in KEEP_STEPS || t == S.steps

function trial_step!(θ, opt, x, y, c, K, supervision, opts, η)
    B = size(x, 2)
    parts = sequence_groups(B) do J
        ℓ, pull = windowloss(θ, x[:, J], y[:, J], c; K, supervision, opts...)
        weight = Float32(length(J)) / B
        weight * ℓ, pull(weight)
    end
    g = last(first(parts))
    for (_, part) in Iterators.drop(parts, 1)
        g .+= part
    end
    ℓ, gnorm = sum(first, parts), norm(g)
    isfinite(ℓ) && isfinite(gnorm) || error("nonfinite loss or gradient norm: loss=$ℓ norm=$gnorm")
    gnorm > SETTINGS.clip && (g .*= Float32(SETTINGS.clip) / gnorm)
    update!(opt, θ, g, η)
    return ℓ, gnorm
end

function trim_history(path, step)
    header, rows... = readlines(path)
    filter!(row -> parse(Int, first(split(row, '\t'; limit = 2))) ≤ step, rows)
    open(path, "w") do io
        println(io, header)
        foreach(row -> println(io, row), rows)
    end
    rows
end

function train_trial(model)
    BLAS.set_num_threads(1)
    S = SETTINGS
    recipe = MODELS[model]
    (; abc, train, val) = trial_data()
    c = Config(; V = length(abc), S.D, S.H, S.L, S.T, S.F)
    @assert c.V == 65 && length(train) == 1003855 && length(val) == 111539
    @assert 2 * c.V * c.D + 2 * c.D + c.V == 16961
    @assert nparams(c) - 16961 == 198272
    opts = model_options(recipe, c)
    train_eval, val_eval = evalset(train, 1, c), evalset(val, 2, c)
    η(t) = warmup_cosine(t; S.η, S.warmup, S.steps, S.ηmin)
    θ = init(c, Float32; rng = Xoshiro(S.seed))
    opt = AdamW(θ; λ = decay(c, S.λ))

    path = mkpath(run_dir(model))
    history, updates = joinpath.(path, ("history.tsv", "updates.tsv"))
    latest = joinpath(path, "latest.bin")
    settings_path = joinpath(path, "settings.txt")
    settings = string((; S, model, recipe, opts, julia = VERSION, threads = Threads.nthreads()), '\n')
    start, seconds, previous_wall, best = 0, 0.0, 0.0, Inf
    if get(ENV, "RESUME", "0") == "1" && isfile(latest)
        read(settings_path, String) == settings || error("resume settings differ from the original run")
        ck = load_checkpoint(latest)
        ck.c == c && ck.abc == abc && length(ck.adam) == 2 && ck.t == ck.step ||
            error("resume needs matching parameters, alphabet and Adam state")
        θ .= ck.θ
        opt.m .= ck.adam[1]
        opt.v .= ck.adam[2]
        opt.t = start = ck.step
        rows = split.(trim_history(history, start), '\t')
        trim_history(updates, start)
        seconds = parse(Float64, last(rows)[3])
        previous_wall = parse(Float64, last(rows)[4])
        best = minimum(parse(Float64, row[8]) for row in rows)
    else
        isfile(latest) && error("existing trial at $path; set RESUME=1")
        write(settings_path, settings)
        write(history, EVAL_HEADER * "\n")
        write(updates, UPDATE_HEADER * "\n")
    end

    @printf("%s: %d parameters, %d steps, passes %s, %d threads (Julia %s)\n",
            model, nparams(c), S.steps, recipe.passes, Threads.nthreads(), VERSION)
    flush(stdout)
    wall_start = time_ns()
    previous_seconds = seconds
    recent = NamedTuple{(:step, :passes, :loss, :gradient_norm), Tuple{Int, Int, Float32, Float32}}[]
    for t in start:S.steps
        if t > start
            rng = Xoshiro(S.seed + t)
            x, y = windows(train, c.T, rand(rng, 1:length(train) - c.T, S.B))
            K = rand(rng, recipe.passes)
            seconds += @elapsed ℓ, gnorm = trial_step!(θ, opt, x, y, c, K, recipe.supervision, opts, η(t))
            push!(recent, (; step = t, passes = K, loss = ℓ, gradient_norm = gnorm))
            if t % 50 == 0
                last50 = view(recent, max(1, length(recent) - 49):length(recent))
                @printf("step %5d  loss %.4f  η %.2e  %6.0f tok/s  %5.1f min\n",
                        t, sum(r -> r.loss, last50) / length(last50), η(t),
                        S.B * c.T * (t - start) / (seconds - previous_seconds), seconds / 60)
                flush(stdout)
            end
        end
        (t == start && isfile(latest) || !evaluate_now(t, S)) && continue

        train_loss = selected_loss(θ, train, train_eval, c, opts, recipe.eval_pass)
        val_loss = selected_loss(θ, val, val_eval, c, opts, recipe.eval_pass)
        isfinite(train_loss) && isfinite(val_loss) || error("nonfinite evaluation at step $t")
        batch_loss = isempty(recent) ? train_loss : sum(r -> r.loss, recent) / length(recent)
        wall = previous_wall + (time_ns() - wall_start) / 1e9
        open(updates, "a") do io
            foreach(row -> println(io, join(row, '\t')), recent)
        end
        empty!(recent)
        open(history, "a") do io
            println(io, join((t, t * S.B * c.T, seconds, wall, η(t), batch_loss, train_loss, val_loss), '\t'))
        end
        @printf("── step %5d  pass %d  train %.4f  val %.4f%s\n",
                t, recipe.eval_pass, train_loss, val_loss, val_loss < best ? "  (best)" : "")
        best = min(best, val_loss)
        save_checkpoint(joinpath(path, @sprintf("step_%06d.bin", t)); c, abc, step = t, θ)
        save_checkpoint(latest; c, abc, step = t, θ, opt)
        flush(stdout)
    end
    @printf("done: %d steps in %.2f update minutes; best val loss %.6f\n", S.steps, seconds / 60, best)
end

abspath(PROGRAM_FILE) == (@__FILE__) && train_trial(Symbol(only(ARGS)))
