# Run from the repository root: julia --project -O1 -t 4 loop/scripts/window_parity.jl
include(joinpath(@__DIR__, "../../scripts/common.jl"))
include(joinpath(@__DIR__, "../src/window.jl"))
using .WindowLoop, Statistics

const TRAIN_LENGTH = 8

binary_words(T) = [1 + ((word >> (t - 1)) & 1) for t in 1:T, word in 0:(1 << T) - 1]
prefix(x) = accumulate((a, b) -> 1 + xor(a - 1, b - 1), x; dims = 1)
per_pass_floor(T, k) = max(T - k - 1, 0) * log(2) / T
floor_loss(T, K, supervision) = supervision == :last ? per_pass_floor(T, K) :
    mean(per_pass_floor(T, k) for k in 1:K)

"""Group exhaustive labels by everything the readout at (k,t) can see."""
function verify_floor(T, K)
    words = binary_words(T)
    targets = prefix(words)
    checks = map(1:K) do k
        by_position = map(1:T) do t
            bins = zeros(Int, 2, 1 << min(t, k + 1))
            for b in axes(words, 2)
                code = foldl((key, id) -> 2key + id - 1, view(words, max(1,t-k):t, b); init = 0)
                bins[targets[t, b], code + 1] += 1
            end
            entropy = 0.0
            ambiguous = 0
            for counts in eachcol(bins)
                n = sum(counts)
                @assert n > 0
                @assert if t ≤ k + 1
                    count(iszero, counts) == 1
                else
                    first(counts) == last(counts)
                end
                ambiguous += all(>(0), counts)
                entropy -= sum(c * log(c / n) for c in counts if c > 0; init = 0.0)
            end
            (; t, ambiguous, contexts = size(bins, 2), entropy = entropy / size(words, 2))
        end
        entropy = mean(cell.entropy for cell in by_position)
        @assert entropy ≈ per_pass_floor(T, k)
        (; k, entropy, by_position)
    end
    deep = mean(cell.entropy for cell in checks)
    @assert deep ≈ floor_loss(T, K, :deep)
    (; T, K, words = size(words, 2), deep, checks)
end

function fit_window(c, seed; K = c.T, tied = true, supervision = :deep,
                    steps = 600, batch = 32, η = 0.02, ηmin = η/10,
                    λ = 0.01, warmup = 20, conditional = true, log_every = 100)
    rng = Xoshiro(1000 + seed)
    θ = init(c; rng = Xoshiro(seed))
    opt = AdamW(θ; β1 = 0.9, β2 = 0.95, ε = 1e-8, λ = decay(c, λ))
    words = binary_words(c.T)
    truth = prefix(words)
    make_batch() = batch == size(words, 2) ? (words, truth) :
        ((x = rand(rng, 1:2, c.T, batch)); (x, prefix(x)))
    objective(x, y) = windowloss(θ, x, y, c; K, tied, supervision, conditional)
    exact_loss() = first(windowloss(θ, words, truth, c; K, tied, supervision))
    # Compile the update path without consuming a step or moving the RNG.
    warm_x = rand(copy(rng), 1:2, c.T, batch)
    _, warm_back = objective(warm_x, prefix(warm_x))
    warm_back()

    loss = zeros(steps)
    curve = [(step = 0, loss = exact_loss())]
    elapsed = 0.0
    for start in 1:log_every:steps
        stop = min(start + log_every - 1, steps)
        elapsed += @elapsed for step in start:stop
            x, y = make_batch()
            loss[step], back = objective(x, y)
            rate = warmup_cosine(step; η, ηmin, warmup, steps)
            update!(opt, θ, clip!(back(), 1.0), rate)
        end
        push!(curve, (step = stop, loss = exact_loss()))
    end
    (; θ, curve, minibatch_loss_before_update = loss, elapsed,
       recipe = (; steps, batch, η, ηmin, warmup, λ, β1 = 0.9, β2 = 0.95, ε = 1e-8,
                  clip = 1.0, conditional, log_every))
end

function eval_words(T)
    T ≤ 8 && return binary_words(T)
    rand(Xoshiro(3100 + T), 1:2, T, 256)
end

probability(gap) = gap ≥ 0 ? inv(1 + exp(-gap)) : ((z = exp(gap)); z / (1 + z))
center(X) = X .- mean(X; dims = 1)

function evaluate_window(θ, c, T, K; tied = true, details = false,
                         stream_stats = false, visitor = WindowLoop.visit_passes, opts...)
    x = eval_words(T)
    y = prefix(x)
    B = size(x, 2)
    passes = NamedTuple[]
    last_prediction = falses(T, B)
    visitor(θ, x, c; K, tied, opts...) do k, Z, X, previous, weights
        gap = reshape(Z[2, :] - Z[1, :], T, B)
        prediction = gap .> 0
        hit = prediction .== (y .== 2)
        accuracy_by_position = vec(mean(hit; dims = 2))
        t = min(k + 1, T)
        accuracy = mean(accuracy_by_position)
        worst_position = minimum(accuracy_by_position)
        accuracy_inside = mean(view(accuracy_by_position, 1:t))
        front = something(findfirst(<(0.99), accuracy_by_position), T + 1) - 1
        metrics = (; k, accuracy, worst_position, accuracy_inside, front)
        if details || stream_stats
            delta = X - previous
            centered_state = center(X)
            metrics = (; metrics...,
                       centered_state_rms = norm(centered_state) / sqrt(B * T),
                       centered_state_max = maximum(norm, eachcol(centered_state)),
                       centered_step_rms = norm(center(delta)) / sqrt(B * T),
                       raw_step_max = maximum(abs, delta),
                       centered_step_max = maximum(abs, center(delta)),
                       column_mean_step_max = maximum(abs, mean(delta; dims = 1)))
        end
        if !details
            push!(passes, metrics)
            return
        end
        prob = probability.(gap)
        mean_probability_of_one = vec(mean(prob; dims = 2))
        mean_error_from_posterior = [mean(i ≤ t ? abs.(prob[i,:] .- (y[i,:] .== 2)) :
                                                abs.(prob[i,:] .- 0.5)) for i in 1:T]
        prev = reshape(view(weights, 2, :, :), c.H, T, B)
        previous_weight = vec(mean(prev; dims = (1,3)))
        previous_weight_min = vec(minimum(prev; dims = (1,3)))
        previous_weight_max = vec(maximum(prev; dims = (1,3)))
        strongly_previous = vec(mean(prev .≥ 0.95; dims = (1,3)))
        strongly_self = vec(mean(prev .≤ 0.05; dims = (1,3)))
        class_flip_rate = k == 1 ? nothing : mean(prediction .!= last_prediction)
        last_prediction .= prediction
        push!(passes, (; metrics..., accuracy_by_position, mean_probability_of_one,
                       mean_error_from_posterior, previous_weight, previous_weight_min,
                       previous_weight_max, strongly_previous, strongly_self, class_flip_rate,
                       cross_entropy = first(crossentropy(Z, y))))
    end
    (; T, K, words = B, exhaustive = T ≤ 8, passes)
end

function transport(θ, c; T = 16, K = 16, forward = windowgpt, opts...)
    x = rand(Xoshiro(723), 1:2, T, 1)
    flipped = copy(x)
    flip_at = T - K ÷ 2
    flipped[flip_at] = 3 - flipped[flip_at]
    Z, back = forward(θ, x, c; K, opts...)
    Δ = zero(Z)
    Δ[2,T] = 1
    Δ[1,T] = -1
    adjoints = NamedTuple[]
    back(Δ; adjoints)
    @assert all(adjoints) do frame
        left = max(1, T - (K - frame.pass))
        all(iszero, view(frame.X̄, :, 1:left-1))
    end
    A, _ = forward(θ, x, c; K, readouts = :all, opts...)
    F, _ = forward(θ, flipped, c; K, readouts = :all, opts...)
    flip = map(1:K) do k
        a, f = A[k], F[k]
        (; pass = k, logit_difference_change = vec((f[2,:] - f[1,:]) - (a[2,:] - a[1,:])),
           prediction_changed = vec((a[2,:] .> a[1,:]) .!= (f[2,:] .> f[1,:])))
    end
    (; input = vec(x), flip_at, seed_site = T, terminal_pass = K,
       state_adjoints = [(; frame.pass, X̄ = tensor(frame.X̄),
                          norm_by_position = [norm(col) for col in eachcol(frame.X̄)])
                         for frame in reverse(adjoints)], flip)
end

function run_window(spec)
    (; name, D, F, seed, K, tied, supervision, options) = spec
    c = Config(; V = 2, D, H = 1, F, T = TRAIN_LENGTH, L = tied ? 1 : K)
    fitted = fit_window(c, seed; K, tied, supervision, options...)
    evaluations = [evaluate_window(fitted.θ, c, T, tied && K > 1 ? 2T : K; tied)
                   for T in (c.T, 32, 64)]
    final_loss = last(fitted.curve).loss
    floor = floor_loss(c.T, K, supervision)
    @printf("%-18s D=%d F=%d seed=%d P=%d loss=%.6f gap=%.6f T=8 %.3f T=32 %.3f/%.3f T=64 %.3f/%.3f %.2fs\n",
            name, D, F, seed, nparams(c), final_loss, final_loss - floor,
            evaluations[1].passes[K].accuracy,
            evaluations[2].passes[tied && K > 1 ? 32 : K].accuracy, last(evaluations[2].passes).accuracy,
            evaluations[3].passes[tied && K > 1 ? 64 : K].accuracy, last(evaluations[3].passes).accuracy,
            fitted.elapsed)
    config = (; c.V, c.D, c.H, c.F, c.L, c.T)
    (; name, seed, K, tied, injection = :after, supervision, config, parameters = nparams(c),
       floor, gap = final_loss - floor, fitted..., evaluations)
end

function spec(name, D, F, seed; K = TRAIN_LENGTH, tied = true, supervision = :deep, options...)
    (; name, D, F, seed, K, tied, supervision, options = (; options...))
end

function write_runs(name, specs)
    BLAS.set_num_threads(1)
    anchor = verify_floor(TRAIN_LENGTH, TRAIN_LENGTH)
    runs = Vector{Any}(undef, length(specs))
    Threads.@threads for i in eachindex(specs)
        runs[i] = run_window(specs[i])
    end
    path = joinpath(@__DIR__, "../artifacts", "window-$name.json")
    writejson(path, (; julia = string(VERSION), anchor, runs))
    println("Wrote ", normpath(path))
end

function write_probe(selected; name = "probe")
    BLAS.set_num_threads(1)
    run = run_window(selected)
    c = Config(; run.config...)
    evaluations = [evaluate_window(run.θ, c, T, 2T; details = true) for T in (8, 32, 64)]
    path = joinpath(@__DIR__, "../artifacts", "window-$name.json")
    writejson(path, (; julia = string(VERSION), run, evaluations, transport = transport(run.θ, c)))
    println("Wrote ", normpath(path))
end

function main(args)
    mode = isempty(args) ? "screen" : only(args)
    mode == "probe" && return write_probe(spec("deep", 8, 12, 2))
    mode == "probe-tuned" && return write_probe(
        spec("deep-long", 8, 12, 1; steps = 1600, batch = 16); name = mode)
    specs = if mode == "screen"
        [spec("deep", D, F, seed) for (D,F) in ((8,12), (6,8), (4,4)) for seed in 1:5]
    elseif mode == "tuning"
        [
            [spec("deep-long", 8, 12, seed; steps = 1600, batch = 16) for seed in 1:3];
            [spec("deep-low", 8, 12, seed; steps = 1200, η = 0.005, batch = 24) for seed in 1:2];
            [spec("deep-hard", 8, 12, seed; steps = 800, batch = 24, conditional = false) for seed in 1:2];
            spec("deep-full", 8, 12, 1; steps = 400, η = 0.01, batch = 256, conditional = false)
        ]
    elseif mode == "long-extra"
        [spec("deep-long", 8, 12, seed; steps = 1600, batch = 16) for seed in 4:5]
    elseif mode == "controls"
        [spec(name, 8, F, seed; K, tied, supervision) for (name, F, K, tied, supervision) in (
            ("last", 12, 8, true, :last),
            ("one", 12, 1, true, :last),
            ("wide-one", 231, 1, true, :last),
            ("untied", 12, 8, false, :deep),
        ) for seed in 1:5]
    else
        error("expected screen, tuning, long-extra, controls, probe or probe-tuned")
    end
    write_runs(mode, specs)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
