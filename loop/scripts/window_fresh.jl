# From the repository root: julia --project -O1 -t 4 loop/scripts/window_fresh.jl [lag_bias [seed ...]]
include(joinpath(@__DIR__, "window_round3.jl"))
using SHA, TOML

const FRESH_CONFIG = joinpath(@__DIR__, "window_fresh.toml")

read_section(recipe, key) = (; (Symbol(k) => v for (k, v) in recipe[key])...)

function fresh_words(evaluation, ids)
    rng = Xoshiro(evaluation.word_seed)
    [rand(rng, ids, T, evaluation.words_per_length) for T in evaluation.lengths]
end

function fresh_train(c, seed, training, lengths, opts, validate)
    (; batch, steps, K_min_offset, K_max_multiplier, rng_offset, supervision, conditional,
       warmup, peak_rate, final_rate, beta1, beta2, epsilon, matrix_decay, clip) = training
    rng = Xoshiro(rng_offset + seed)
    θ = init(c, Float64; rng = Xoshiro(seed))
    opt = AdamW(θ; β1 = beta1, β2 = beta2, ε = epsilon, λ = decay(c, matrix_decay))
    function batch_at(rng)
        T = rand(rng, lengths)
        K = rand(rng, T + K_min_offset:K_max_multiplier*T)
        rand(rng, 1:c.V, T, batch), K
    end
    objective(θ, x, K) = windowloss(θ, x, prefix(x), c; K, supervision = Symbol(supervision),
                                    conditional, opts...)
    # Match Round 3's untimed update on copies to warm up the timed path.
    x, K = batch_at(copy(rng))
    update!(deepcopy(opt), copy(θ), clip!(last(objective(θ, x, K))(), clip), peak_rate)
    initial_loss = validate(θ)
    minibatch_loss_before_update = zeros(steps)
    elapsed = @elapsed for step in 1:steps
        x, K = batch_at(rng)
        minibatch_loss_before_update[step], pull = objective(θ, x, K)
        rate = warmup_cosine(step; η = peak_rate, ηmin = final_rate, warmup, steps)
        update!(opt, θ, clip!(pull(), clip), rate)
    end
    (; θ, elapsed, initial_loss, final_loss = validate(θ), minibatch_loss_before_update)
end

function fresh_rollout(θ, c, x, opts, evaluation)
    T, B = size(x)
    horizons = evaluation.horizon_multipliers .* T
    K = maximum(horizons)
    y = prefix(x) .== 2
    accuracy_by_pass = zeros(K)
    fronts = zeros(Int, K)
    diagnostics = NamedTuple[]
    arrival = NamedTuple[]
    diagnostic_passes = (1, 4, 16, horizons...)
    WindowLoop.visit_passes(θ, x, c; K, opts...) do k, Z, X, previous, weights
        logits = reshape(view(Z, 2, :) - view(Z, 1, :), T, B)
        hit_by_site = vec(mean((logits .> 0) .== y; dims = 2))
        accuracy_by_pass[k] = mean(hit_by_site)
        fronts[k] = something(findfirst(<(evaluation.front_accuracy), hit_by_site), T + 1) - 1
        if k < T
            site = k + 1
            signed = ifelse.(view(y, site, :), view(logits, site, :), .-view(logits, site, :))
            push!(arrival, (; k, site, accuracy = hit_by_site[site],
                             mean_target_probability = mean(probability, signed)))
        end
        if k in diagnostic_passes
            prev = view(reshape(weights, 2, c.H, T, B), 2, :, 2:T, :)
            posterior = [t ≤ k + 1 ? y[t, b] : 0.5 for t in 1:8, b in 1:B]
            errors = abs.(probability.(view(logits, 1:8, :)) .- posterior)
            push!(diagnostics, (; k, front = fronts[k], accuracy = accuracy_by_pass[k],
                                accuracy_inside = mean(hit_by_site[1:min(k+1,T)]),
                                centered_state_rms = norm(center(X)) / sqrt(B*T),
                                centered_step_rms = norm(center(X - previous)) / sqrt(B*T),
                                predecessor_mean = mean(prev), predecessor_min = minimum(prev),
                                predecessor_max = maximum(prev), hard_self = mean(prev .≤ 0.05),
                                hard_previous = mean(prev .≥ 0.95), first_eight_accuracy = hit_by_site[1:8],
                                first_eight_inside_cone_error = vec(mean(errors; dims = 2))))
        end
    end
    (; T, horizons, accuracy = accuracy_by_pass[horizons], accuracy_by_pass, fronts, diagnostics, arrival)
end

# Invert the equal-tailed binomial test. Reflect the lower endpoint to
# obtain the upper endpoint, including the all-success and all-failure cases.
function clopper_pearson(successes, n, level)
    tail = (1 - level) / 2
    function lower(k)
        iszero(k) && return 0.0
        terms = k:n
        coefficients = [binomial(big(n), j) for j in terms]
        lo, hi = 0.0, 1.0
        for _ in 1:60
            p = big((lo + hi) / 2)
            probability = sum(a * p^j * (1 - p)^(n-j) for (a, j) in zip(coefficients, terms))
            if probability < tail
                lo = Float64(p)
            else
                hi = Float64(p)
            end
        end
        (lo + hi) / 2
    end
    (; level, lower = lower(successes), upper = 1 - lower(n - successes))
end

function fresh_main(args)
    BLAS.set_num_threads(1)
    recipe = TOML.parsefile(FRESH_CONFIG)
    model = read_section(recipe, "model")
    training = read_section(recipe, "training")
    evaluation = read_section(recipe, "evaluation")
    default_seeds = collect(recipe["seed_first"]:recipe["seed_last"])
    seeds = length(args) ≤ 1 ? default_seeds : parse.(Int, args[2:end])
    allunique(seeds) || throw(ArgumentError("seeds must be distinct"))
    bias = isempty(args) ? model.lag_bias : parse(Float64, first(args))
    isfinite(bias) || throw(ArgumentError("lag bias must be finite"))
    @assert model.window == 2
    c = Config(; read_section(recipe, "shape")...)
    @assert nparams(c) == 270
    opts = (; model.tied, position = Symbol(model.position), injection = Symbol(model.injection),
            start = Symbol(model.start), lag_bias = bias)
    lengths = training.T_min:training.T_max
    pairs = [(T, K) for T in lengths for K in T + training.K_min_offset:training.K_max_multiplier*T]
    @assert pairs == MIXED
    floor = floor_record()
    visitor(f, θ, x, c; K) = WindowLoop.visit_passes(f, θ, x, c; K, opts...)
    validate(θ) = validation_prefix(θ, c, pairs, visitor; chunk = evaluation.exhaustive_chunk)
    test_words = fresh_words(evaluation, 1:c.V)

    runs = run_batch(seeds) do seed
        fitted = fresh_train(c, seed, training, lengths, opts, validate)
        rollouts = [fresh_rollout(fitted.θ, c, x, opts, evaluation) for x in test_words]
        scores = vcat(getproperty.(rollouts, :accuracy)...)
        gap = fitted.final_loss - floor.nats
        relative_gap = gap / floor.nats
        qualified = abs(relative_gap) ≤ evaluation.max_relative_loss_gap &&
                    all(==(evaluation.required_accuracy), scores)
        @printf("bias=%+g seed=%d CE=%.9f T32 %.6f/%.6f T64 %.6f/%.6f %6.2fs %s\n",
                bias, seed, fitted.final_loss, scores..., fitted.elapsed, qualified ? "PASS" : "FAIL")
        flush(stdout)
        (; seed, minibatch_seed = training.rng_offset + seed, fitted..., gap, relative_gap, qualified, rollouts)
    end
    successes = count(r -> r.qualified, runs)
    interval = clopper_pearson(successes, length(runs), evaluation.interval_level)
    bias_label = @sprintf("%g", bias)
    suffix = seeds == default_seeds ? "" : "-seeds-" * bytes2hex(sha256(join(seeds, ',')))[1:16]
    path = joinpath(@__DIR__, "../artifacts/window-fresh-bias-$bias_label$suffix.json")
    words = [(; T = size(x, 1), words = tensor(x)) for x in test_words]
    writejson(path, (; julia = string(VERSION), julia_threads = Threads.nthreads(),
                     blas_threads = BLAS.get_num_threads(), recipe, lag_bias = bias,
                     parameters = nparams(c), seeds, floor, words, successes, interval, runs))
    @printf("%d/%d qualified, %.0f%% Clopper–Pearson interval [%.4f, %.4f]\n",
            successes, length(runs), 100interval.level, interval.lower, interval.upper)
    println("Wrote ", normpath(path))
end

abspath(PROGRAM_FILE) == (@__FILE__) && fresh_main(ARGS)
