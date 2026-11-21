# From the repository root: julia --project loop/scripts/rule60.jl
include(joinpath(@__DIR__, "common.jl"))

function rule60_batch(T, K, target_k, supervision)
    function make_batch(rng, B)
        x = rand(rng, BIT_IDS, T, B)
        targets = supervision == :deep ? [lucas(x, k) for k in 1:K] : lucas(x, target_k)
        x, targets
    end
end

function evaluate_rule60(θ, c, opts, T, eval_K, target_k)
    x = T ≤ 8 ? binary_words(T) : rand(Xoshiro(3200 + T), BIT_IDS, T, 256)
    ps = pass_predictions(θ, x, c, (; opts..., K = eval_K))
    fixed_target = lucas(x, target_k)
    expected = x
    passes = map(1:eval_K) do k
        expected = rule60_step(expected)
        @assert expected == lucas(x, k)
        by_position = position_accuracy(ps[k], expected)
        (; pass = k, lucas_accuracy = mean(by_position), by_position,
           trained_target_accuracy = accuracy(ps[k], fixed_target),
           exact_sequences = mean(vec(all(ps[k] .== expected; dims = 1))))
    end
    (; length = T, examples = size(x, 2), exhaustive = T ≤ 8, passes)
end

function run_rule60(name, c, K; seed, recipe = RECIPE, tied = true, injection = :none,
                    target_k = K, supervision = :last, horizon = 12, lengths = (c.T, 2c.T), transition = (;), θ_initial = nothing)
    opts = (; K, tied, injection, transition...)
    make_batch = rule60_batch(c.T, K, target_k, supervision)
    fitted = fit(make_batch, c, seed, opts; recipe, supervision, θ_initial)
    eval_K = tied ? max(K, horizon) : K
    curves = [evaluate_rule60(fitted.θ, c, opts, T, eval_K, target_k) for T in lengths]
    impulse = ones(Int, 2horizon + 1, 1)
    impulse[1] = 2
    endpoint = endpoint_losses(make_batch, c, seed, opts, fitted; supervision, θ_initial)
    @printf("%s seed %d P=%d loss %.4f → %.4f target acc %.4f %.2fs\n",
            name, seed, nparams(c), endpoint.initial, endpoint.final,
            first(curves).passes[K].trained_target_accuracy, fitted.elapsed)
    println("  Lucas ", round.(getproperty.(first(curves).passes, :lucas_accuracy); digits = 3))
    (; name, seed, config = config(c), nparams = nparams(c), options = opts, target_k, supervision,
       recipe, update_loss = fitted.losses, endpoint, train_seconds = fitted.elapsed,
       evaluations = curves, θ = fitted.θ, impulse = (; input = vec(impulse),
       exact = [vec(lucas(impulse, k)) for k in 1:eval_K],
       predicted = [vec(p) for p in pass_predictions(fitted.θ, impulse, c, (; opts..., K = eval_K))]),
       trace = example_trace(fitted.θ, impulse, c, (; opts..., K = eval_K)))
end

function rule60_experiment()
    c = Config(V = 2, D = 8, H = 1, L = 1, T = 6, F = 12)
    K = 3
    (; untied, wide) = control_configs(c, K)
    runs = [
        run_rule60(name, shape, passes; seed, target_k = K, tied, supervision)
        for (name, shape, passes, tied, supervision) in (
            ("tied", c, K, true, :last),
            ("tied_deep", c, K, true, :deep),
            ("one_pass", c, 1, true, :last),
            ("untied", untied, K, false, :last),
            ("wide_one_pass", wide, 1, true, :last),
        ) for seed in SEEDS
    ]
    save_experiment("rule60", (; task = "Rule 60 versus Lucas' theorem", runs))
end

abspath(PROGRAM_FILE) == (@__FILE__) && rule60_experiment()
