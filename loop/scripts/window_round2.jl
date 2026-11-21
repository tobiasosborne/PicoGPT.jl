# Run from the repository root, for example:
# julia --project -O1 -t 4 loop/scripts/window_round2.jl diagnosis
include(joinpath(@__DIR__, "window_parity.jl"))

const MIXED_LENGTHS = 4:12
const FIXED = [(8, 8)]
const MIXED = [(T, K) for T in MIXED_LENGTHS for K in T-1:2T]
const ONE = [(T, 1) for T in MIXED_LENGTHS]

# Sample T uniformly, then K uniformly conditional on T. For one-pass and
# fixed-depth controls the inner mean is over a singleton.
function distribution_mean(f, pairs)
    lengths = unique(first.(pairs))
    mean(lengths) do T
        mean(f(T, K) for (t, K) in pairs if t == T)
    end
end

floor_fraction(T, K) = sum(max(T - k - 1, 0) for k in 1:K) // (T * K)

function context_counts(T, K; position)
    # The low bits store the earliest visible letters. Plain NoPE with singleton
    # attention is equivalent to padding the word on the left with its first bit.
    outcomes = map(1:K) do k
        S = min(T, k + 1)
        counts = zeros(Int, 2, 1 << S, position ? T : 1)
        for t in 1:T, word in 0:(1 << T)-1
            L = min(t, k + 1)
            code = (word >> (t - L)) & ((1 << L) - 1)
            if !position && L < S
                padding = isodd(word) ? (1 << (S - L)) - 1 : 0
                code = (code << (S - L)) | padding
            end
            label = isodd(count_ones(word & ((1 << t) - 1))) ? 2 : 1
            counts[label, code + 1, position ? t : 1] += 1
        end
        n = T * (1 << T)
        entropy = 0.0
        mistakes = 0
        for j in axes(counts, 3), i in axes(counts, 2)
            a, b = counts[1, i, j], counts[2, i, j]
            mistakes += min(a, b)
            a == 0 || (entropy -= a * log(a / (a + b)))
            b == 0 || (entropy -= b * log(b / (a + b)))
        end
        (; entropy = entropy / n, accuracy_ceiling = 1 - mistakes / n)
    end
    outcomes
end

function round2_floors()
    by_T = map(MIXED_LENGTHS) do T
        positioned = context_counts(T, 2T; position = true)
        nope = context_counts(T, 2T; position = false)
        @assert all(k -> positioned[k].entropy ≈ per_pass_floor(T, k), 1:2T)
        (; T, words = 1 << T, positioned, nope)
    end
    distributions = map((FIXED, MIXED, ONE)) do pairs
        fraction = distribution_mean(floor_fraction, pairs)
        enumerated = distribution_mean(pairs) do T, K
            mean(x.entropy for x in by_T[T - first(MIXED_LENGTHS) + 1].positioned[1:K])
        end
        @assert enumerated ≈ fraction * log(2)
        (; numerator = numerator(fraction), denominator = denominator(fraction), nats = enumerated)
    end
    # With a complete receptive field, leading runs of bit 1 with odd and even
    # lengths give the same padded context. An odd run has the larger total weight.
    nope_ceiling(T) = 1 - sum(2.0^-r * (1 + (T - r)/2) for r in 2:2:T) / T
    @assert all(row -> row.nope[end].accuracy_ceiling ≈ nope_ceiling(row.T), by_T)
    nope_deep_floor = distribution_mean(MIXED) do T, K
        mean(x.entropy for x in by_T[T - first(MIXED_LENGTHS) + 1].nope[1:K])
    end
    (; fixed = distributions[1], mixed = distributions[2], one = distributions[3],
       nope_deep_floor, nope_terminal = [(; T, ceiling = nope_ceiling(T)) for T in (8, 32, 64)],
       by_T)
end

function validation_prefix(θ, c, pairs, forward_passes; chunk = 128)
    mean(unique(first.(pairs))) do T
        horizons = [K for (t, K) in pairs if t == T]
        K = maximum(horizons)
        words = binary_words(T)
        pass_totals = zeros(K)
        for cols in Iterators.partition(axes(words, 2), chunk)
            x = words[:, cols]
            y = prefix(x)
            forward_passes(θ, x, c; K) do k, Z, X, previous, weights
                pass_totals[k] += length(cols) * first(crossentropy(Z, y))
            end
        end
        cumulative = cumsum(pass_totals)
        mean(cumulative[k] / k for k in horizons) / size(words, 2)
    end
end

function fit_round2(c, seed, pairs, objective, initialize, validate; steps = 900, batch = 12,
                    η = 0.02, ηmin = η/10, λ = 0.01, warmup = 20, rates = decay(c, λ),
                    log_every = steps)
    rng = Xoshiro(1000 + seed)
    θ = initialize(c; rng = Xoshiro(seed))
    opt = AdamW(θ; β1 = 0.9, β2 = 0.95, ε = 1e-8, λ = rates)
    lengths = unique(first.(pairs))
    passes = Dict(T => [K for (t,K) in pairs if t == T] for T in lengths)
    function batch_at(rng)
        T = rand(rng, lengths)
        K = rand(rng, passes[T])
        x = rand(rng, 1:2, T, batch)
        x, K
    end
    x, K = batch_at(copy(rng))
    _, pull = objective(θ, x, K)
    pull()
    curve = [(step = 0, loss = validate(θ))]
    losses = zeros(steps)
    elapsed = 0.0
    for start in 1:log_every:steps
        stop = min(start + log_every - 1, steps)
        elapsed += @elapsed for step in start:stop
            x, K = batch_at(rng)
            losses[step], back = objective(θ, x, K)
            rate = warmup_cosine(step; η, ηmin, warmup, steps)
            update!(opt, θ, clip!(back(), 1.0), rate)
        end
        push!(curve, (step = stop, loss = validate(θ)))
    end
    (; θ, curve, elapsed, minibatch_loss_before_update = losses,
       recipe = (; steps, batch, η, ηmin, warmup, λ, log_every, β1 = 0.9, β2 = 0.95, ε = 1e-8, clip = 1.0))
end

function config2(D, F, pairs, tied)
    Config(; V = 2, D, H = 1, F, T = maximum(first, pairs), L = tied ? 1 : maximum(last, pairs))
end

function stock_spec(name, D, F, seed; pairs = MIXED, position = :boundary,
                    injection = :refresh, tied = true, stream_stats = false, options...)
    (; name, D, F, seed, pairs, position, injection, tied, stream_stats, options = (; options...))
end

function run_stock(spec)
    (; name, D, F, seed, pairs, position, injection, tied, stream_stats, options) = spec
    c = config2(D, F, pairs, tied)
    objective(θ, x, K) = windowloss(θ, x, prefix(x), c; K, tied, position, injection, conditional = true)
    visitor(f, θ, x, c; K) = WindowLoop.visit_passes(f, θ, x, c; K, tied, position, injection)
    validate(θ) = validation_prefix(θ, c, pairs, visitor)
    fitted = fit_round2(c, seed, pairs, objective, init, validate; options...)
    Kmax = maximum(last, pairs)
    evaluations = [evaluate_window(fitted.θ, c, T, tied && Kmax > 1 ? 2T : Kmax;
                                   tied, position, injection, stream_stats)
                   for T in (32, 64)]
    config = (; c.V, c.D, c.H, c.F, c.L, c.T)
    floor = distribution_mean(floor_fraction, pairs) * log(2)
    result = (; name, seed, pairs, position, injection, tied, config, parameters = nparams(c),
              floor, gap = last(fitted.curve).loss - floor, fitted..., evaluations)
    report_run(result)
    result
end

function init_register(c, transition; rng)
    θ = zeros(WindowLoop.register_stride(c, transition) * c.L)
    for block in WindowLoop.register_parameters(θ, c, transition)
        if transition == :gain
            randn!(rng, block)
        else
            p = WindowLoop.register_mlp(block, c.F)
            randn!(rng, p.W1) ./= sqrt(2)
            randn!(rng, p.W2) ./= sqrt(c.F)
        end
    end
    θ
end

function decay_register(c, transition, λ)
    rates = zeros(WindowLoop.register_stride(c, transition) * c.L)
    for block in WindowLoop.register_parameters(rates, c, transition)
        if transition == :gain
            block .= λ
        else
            p = WindowLoop.register_mlp(block, c.F)
            p.W1 .= λ
            p.W2 .= λ
        end
    end
    rates
end

function register_spec(name, seed; F = 1, transition = :gain, pairs = MIXED,
                       tied = true, steps = 650, batch = 12, η = 0.035, λ = 0.001, options...)
    (; name, seed, F, transition, pairs, tied, options = (; steps, batch, η, λ, options...))
end

function run_register(spec)
    (; name, seed, F, transition, pairs, tied, options) = spec
    c = config2(2, F, pairs, tied)
    objective(θ, x, K) = register_windowloss(θ, x, prefix(x), c; K, tied, transition, conditional = true)
    visitor(f, θ, x, c; K) = WindowLoop.visit_register(f, θ, x, c; K, tied, transition)
    validate(θ) = validation_prefix(θ, c, pairs, visitor)
    initialize(c; rng) = init_register(c, transition; rng)
    rates = decay_register(c, transition, options.λ)
    fitted = fit_round2(c, seed, pairs, objective, initialize, validate; rates, options...)
    Kmax = maximum(last, pairs)
    evaluations = [evaluate_window(fitted.θ, c, T, tied && Kmax > 1 ? 2T : Kmax;
                                   tied, transition, visitor = WindowLoop.visit_register)
                   for T in (32, 64)]
    config = (; c.V, c.D, c.H, c.F, c.L, c.T)
    floor = distribution_mean(floor_fraction, pairs) * log(2)
    result = (; name, seed, pairs, transition, tied, config, parameters = length(fitted.θ),
              floor, gap = last(fitted.curve).loss - floor, fitted..., evaluations)
    report_run(result)
    result
end

at_rollout(evaluations) = [e.passes[k].accuracy for e in evaluations for k in (e.T, 2e.T)]

function report_run(run)
    base = @sprintf("%-22s seed=%d P=%d loss=%.7f gap=%.7f %.2fs",
                    run.name, run.seed, run.parameters, last(run.curve).loss, run.gap, run.elapsed)
    rollout = if all(e -> e.K == 2e.T, run.evaluations)
        @sprintf(" T32 %.4f/%.4f T64 %.4f/%.4f", at_rollout(run.evaluations)...)
    else
        @sprintf(" pass %d T32 %.4f T64 %.4f", run.evaluations[1].K,
                 (last(e.passes).accuracy for e in run.evaluations)...)
    end
    println(base * rollout)
    flush(stdout)
end

function save_round2(mode, result)
    path = joinpath(@__DIR__, "..", "artifacts", "window-round2-$mode.json")
    writejson(path, (; julia = string(VERSION), result...))
    println("Wrote ", normpath(path))
end

function stock_diagnosis()
    [
        [stock_spec("fixed-$position", 8, 12, seed; pairs = FIXED, position, injection = :after)
         for position in (:sinusoidal, :none, :boundary) for seed in (1, 2)];
        [stock_spec("mixed-$injection", 8, 12, seed; injection)
         for injection in (:after, :refresh, :normalized) for seed in (1, 2)];
        stock_spec("fixed-refresh", 8, 12, 1; pairs = FIXED);
        stock_spec("fixed-normalized", 8, 12, 1; pairs = FIXED, injection = :normalized);
    ]
end

function register_screen()
    [
        [register_spec("gain", seed) for seed in 1:5];
        [register_spec("mlp-$F", seed; F, transition = :mlp, η = 0.02)
         for F in (1, 2, 3, 4, 8) for seed in 1:5];
    ]
end

function register_controls()
    [
        [register_spec("one-pass", seed; pairs = ONE) for seed in 1:5];
        [register_spec("untied", seed; tied = false) for seed in 1:5];
    ]
end

function register_probe()
    run = run_register(register_spec("gain", 1; log_every = 130))
    c = Config(; run.config...)
    evaluations = [evaluate_window(run.θ, c, T, 2T; details = true, visitor = WindowLoop.visit_register)
                   for T in (8, 32, 64)]
    (; run, evaluations, cone_T64 = tensor(stack(row.accuracy_by_position for row in last(evaluations).passes)),
       transport = transport(run.θ, c; forward = register_window))
end

function stock_probe()
    run = run_stock(stock_spec("mixed-normalized", 8, 12, 2; injection = :normalized, log_every = 225))
    c = Config(; run.config...)
    opts = (; run.position, run.injection)
    evaluations = [evaluate_window(run.θ, c, T, 2T; details = true, opts...) for T in (8, 32, 64)]
    (; run, evaluations, cone_T64 = tensor(stack(row.accuracy_by_position for row in last(evaluations).passes)),
       transport = transport(run.θ, c; opts...))
end

function stock_normalized()
    [
        [stock_spec("normalized-baseline", 8, 12, seed; injection = :normalized)
         for seed in 3:5];
        [stock_spec("normalized-tuned", D, F, seed; injection = :normalized,
                    steps = 1250, batch = 8, η = 0.02)
         for (D, F) in ((8, 12), (6, 16), (4, 16), (8, 24))
         for seed in (D == 8 && F == 12 ? (1:5) : (1:2))];
    ]
end

function rule60_step(x)
    next = copy(x)
    @views next[2:end, :] .= 1 .+ xor.(x[2:end, :] .- 1, x[1:end-1, :] .- 1)
    next
end

function rule60_targets(x, K)
    values = [copy(x)]
    for k in 1:K
        push!(values, rule60_step(last(values)))
    end
    values[2:end]
end

function lucas_window(x, k)
    y = ones(Int, size(x))
    for j in 0:min(k, size(x, 1) - 1)
        (j & k) == j || continue
        @views y[j+1:end, :] .= 1 .+ xor.(y[j+1:end, :] .- 1, x[1:end-j, :] .- 1)
    end
    y
end

function run_rule60(seed)
    c = config2(2, 1, MIXED, true)
    λ = 0.001
    rates = decay_register(c, :gain, λ)
    initialize(c; rng) = init_register(c, :gain; rng)
    objective(θ, x, K) =
        register_windowloss(θ, x, rule60_targets(x, K), c; K, task = :rule60)
    function evaluate(θ, x, K)
        iterates = rule60_targets(x, K)
        rows = NamedTuple[]
        WindowLoop.visit_register(θ, x, c; K, task = :rule60) do k, Z, X, previous, weights
            y = lucas_window(x, k)
            @assert y == iterates[k]
            hit = reshape(Z[2, :] .> Z[1, :], size(x)) .== (y .== 2)
            push!(rows, (; k, accuracy = mean(hit), loss = first(crossentropy(Z, y))))
        end
        rows
    end
    validate(θ) = mean(MIXED_LENGTHS) do T
        rng = Xoshiro(7100 + T)
        rows = evaluate(θ, rand(rng, 1:2, T, 128), 2T)
        mean(mean(r.loss for r in rows[1:k]) for k in T-1:2T)
    end
    fit = fit_round2(c, seed, MIXED, objective, initialize, validate;
                     λ, rates, steps = 650, batch = 12, η = 0.035)
    K = 4 * maximum(last, MIXED)
    evaluation = evaluate(fit.θ, eval_words(64), K)
    impulse = ones(Int, K + 1, 1)
    impulse[1] = 2
    impulse_evaluation = evaluate(fit.θ, impulse, K)
    println(@sprintf("rule60 seed=%d deep loss=%.7f minimum accuracy=%.5f impulse=%.5f %.2fs",
                     seed, last(fit.curve).loss, minimum(r.accuracy for r in evaluation),
                     minimum(r.accuracy for r in impulse_evaluation), fit.elapsed))
    (; seed, floor = 0, evaluation, impulse_evaluation, fit...)
end

function run_batch(f, specs)
    runs = Vector{Any}(undef, length(specs))
    Threads.@threads for i in eachindex(specs)
        runs[i] = f(specs[i])
    end
    runs
end

function round2_main(args)
    BLAS.set_num_threads(1)
    mode = only(args)
    if mode == "floor"
        save_round2(mode, round2_floors())
    elseif mode == "diagnosis"
        runs = run_batch(run_stock, stock_diagnosis())
        save_round2(mode, (; runs))
    elseif mode == "register"
        runs = run_batch(run_register, register_screen())
        save_round2(mode, (; runs))
    elseif mode == "controls"
        runs = run_batch(run_register, register_controls())
        save_round2(mode, (; runs))
    elseif mode == "probe"
        save_round2(mode, register_probe())
    elseif mode == "stock-probe"
        save_round2(mode, stock_probe())
    elseif mode == "stock-normalized"
        save_round2(mode, (; runs = run_batch(run_stock, stock_normalized())))
    elseif mode == "stock-narrow"
        specs = [stock_spec("normalized-tuned", 4, 16, seed; injection = :normalized,
                            steps = 1250, batch = 8, η = 0.02)
                 for seed in 3:5]
        save_round2(mode, (; runs = run_batch(run_stock, specs)))
    elseif mode == "distribution"
        stock = [stock_spec("normalized-$position-$distribution", 8, 12, 2; pairs, position, injection = :normalized)
                 for position in (:sinusoidal, :boundary) for (distribution, pairs) in
                 (("fixed", FIXED), ("mixed", MIXED))]
        register = [register_spec("gain-fixed", seed; pairs = FIXED) for seed in 1:5]
        stock_runs = run_batch(run_stock, stock)
        register_runs = run_batch(run_register, register)
        save_round2(mode, (; runs = vcat(stock_runs, register_runs)))
    elseif mode == "stream"
        specs = [stock_spec("mixed-$injection", 8, 12, 1; injection, stream_stats = true)
                 for injection in (:after, :refresh, :normalized)]
        save_round2(mode, (; runs = run_batch(run_stock, specs)))
    elseif mode == "stock-tune"
        specs = [stock_spec("refresh-tuned", D, F, seed; steps = 1500, batch = 8, η)
                 for (D, F, η) in ((8, 12, 0.02), (6, 16, 0.02), (8, 24, 0.015))
                 for seed in 1:5]
        save_round2(mode, (; runs = run_batch(run_stock, specs)))
    elseif mode == "rule60"
        save_round2(mode, (; runs = run_batch(run_rule60, collect(1:5))))
    else
        error("unknown window Round 2 experiment: $mode")
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && round2_main(ARGS)
