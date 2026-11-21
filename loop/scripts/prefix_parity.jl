# From the repository root: julia --project loop/scripts/prefix_parity.jl
include(joinpath(@__DIR__, "common.jl"))

prefix(x) = accumulate(bit_xor, x; dims = 1)

function windowed_parity(x, k)
    y = prefix(x)
    if k + 1 < size(x, 1)
        @views y[k+2:end, :] .= bit_xor.(y[k+2:end, :], copy(y[1:end-k-1, :]))
    end
    y
end

function prefix_batch(T, K, window_targets)
    function make_batch(rng, B)
        x = rand(rng, BIT_IDS, T, B)
        x, window_targets ? [windowed_parity(x, k) for k in 1:K] : prefix(x)
    end
end

function evaluate_prefix(θ, c, opts, T, eval_K)
    x = T ≤ 8 ? binary_words(T) : rand(Xoshiro(3100 + T), BIT_IDS, T, 256)
    target = prefix(x)
    flipped = copy(x)
    flipped[1, :] .= 3 .- flipped[1, :]
    ps = pass_predictions(θ, x, c, (; opts..., K = eval_K))
    ps_flipped = pass_predictions(θ, flipped, c, (; opts..., K = eval_K))
    passes = map(1:eval_K) do k
        predicted = ps[k]
        by_position = position_accuracy(predicted, target)
        # Require every position up to the front to be at least 95% correct.
        front = something(findfirst(<(0.95), by_position), T + 1) - 1
        first_token_flip_rate = vec(mean(predicted .!= ps_flipped[k]; dims = 2))
        window = hasproperty(opts, :attention_span) && opts.attention_span == :adjacent ?
            (; window_accuracy = accuracy(predicted, windowed_parity(x, k))) : (;)
        (; pass = k, accuracy = mean(by_position), by_position, front, first_token_flip_rate,
           window...)
    end
    (; length = T, examples = size(x, 2), exhaustive = T ≤ 8, passes)
end

function run_prefix(name, c, K; seed, recipe = RECIPE, tied = true, injection = :after,
                    supervision = :last, horizon = 2c.T, lengths = (c.T, 2c.T, 4c.T),
                    transition = (;), window_targets = false, θ_initial = nothing)
    opts = (; K, tied, injection, transition...)
    window_targets && supervision != :deep && throw(ArgumentError("window targets need deep supervision"))
    make_batch = prefix_batch(c.T, K, window_targets)
    fitted = fit(make_batch, c, seed, opts; recipe, supervision, θ_initial)
    curves = [evaluate_prefix(fitted.θ, c, opts, T, tied ? max(K, horizon) : K) for T in lengths]
    reference = reshape([isodd(t) ? 2 : 1 for t in 1:last(lengths)], :, 1)
    endpoint = endpoint_losses(make_batch, c, seed, opts, fitted; supervision, θ_initial)
    train_accuracy = first(curves).passes[K].accuracy
    @printf("%s seed %d P=%d loss %.4f → %.4f acc %.4f %.2fs\n",
            name, seed, nparams(c), endpoint.initial, endpoint.final, train_accuracy, fitted.elapsed)
    println("  fronts ", getproperty.(first(curves).passes, :front))
    target_info = window_targets ? (; window_targets) : (;)
    (; name, seed, config = config(c), nparams = nparams(c), options = opts, supervision, recipe,
       target_info...,
       update_loss = fitted.losses, endpoint, train_seconds = fitted.elapsed, evaluations = curves,
       θ = fitted.θ, trace_input = vec(reference), trace_target = vec(prefix(reference)),
       trace = example_trace(fitted.θ, reference, c, (; opts..., K = tied ? max(K, horizon) : K)))
end

function prefix_experiment()
    c = Config(V = 2, D = 8, H = 1, L = 1, T = 5, F = 12)
    K = c.T
    (; untied, wide) = control_configs(c, K)
    runs = [
        run_prefix(name, shape, passes; seed, tied, injection, supervision)
        for (name, shape, passes, tied, injection, supervision) in (
            ("tied", c, K, true, :after, :last),
            ("tied_deep", c, K, true, :after, :deep),
            ("one_pass", c, 1, true, :after, :last),
            ("untied", untied, K, false, :after, :last),
            ("wide_one_pass", wide, 1, true, :after, :last),
        ) for seed in SEEDS
    ]
    save_experiment("prefix-parity", (; task = "Prefix XOR", runs))
end

abspath(PROGRAM_FILE) == (@__FILE__) && prefix_experiment()
