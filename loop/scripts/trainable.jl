# Run from the root: julia --project -O1 loop/scripts/trainable.jl screen|prefix|rule60
include(joinpath(@__DIR__, "prefix_parity.jl"))

const SEARCH_RECIPE = (; RECIPE..., steps = 600, batch = 16, η = 0.004,
                        ηmin = 0.0004, warmup = 20, λ = 0.001)

function setting(name; task = :prefix, D = 8, F = 16, T = 12, H = 1,
                 mode = :tied, supervision = :last, curriculum = false,
                 extra = 0, injection = :after, attention_span = :full,
                 position_code = true, stabilize = false, previous_bias = 0.0,
                 scale_by_depth = false, architecture = :stock, bounded = true,
                 recipe = SEARCH_RECIPE)
    c = Config(; V = 2, D, F, T, H, L = 1)
    options = architecture == :register ? (; input = task == :prefix ? :source : :state, bounded) :
              (; injection, attention_span, position_code, stabilize, previous_bias)
    (; name, task, c, mode, supervision, curriculum, extra, scale_by_depth,
       architecture = Val(architecture), recipe,
       options)
end

uses_register(s) = s.architecture isa Val{:register}

function shape(s)
    if uses_register(s)
        (; V, D, F, T, H) = s.c
        depth = T + s.extra
        s.mode == :untied && return Config(; V, D, F, T, H, L = depth)
        s.mode == :wide && return Config(; V, D, F = round(Int, (depth*(4F+1) - 1)/4), T, H, L = 1)
        return s.c
    end
    s.mode == :untied && return control_configs(s.c, s.c.T + s.extra).untied
    s.mode == :wide && return control_configs(s.c, s.c.T + s.extra).wide
    s.c
end

function loop_options(s, K)
    uses_register(s) && return (; s.options..., K, tied = s.mode != :untied)
    (; s.options..., K, tied = s.mode != :untied,
       residual_scale = s.scale_by_depth ? inv(K) : 1)
end

forward(s) = uses_register(s) ? registergpt : loopgpt
objective_op(s) = uses_register(s) ? registerloss : looploss
parameter_count(s, c) = uses_register(s) ? register_nparams(c) : nparams(c)

function targets(s, x, horizon)
    if s.task == :prefix
        s.supervision != :window && return prefix(x)
        return [windowed_parity(x, k) for k in 1:horizon]
    end
    s.supervision == :last && return lucas(x, horizon)
    expected = x
    map(1:horizon) do _
        expected = rule60_step(expected)
    end
end

function mixed_batch(s, rng, step; B = s.recipe.batch)
    # Two equally sized sub-batches, each with one independently chosen length.
    # The split avoids padding and lets each length use its own number of passes.
    longest = s.curriculum ?
        min(s.c.T, 3 + fld((s.c.T - 3) * (step - 1), max(1, fld(3s.recipe.steps, 5)))) : s.c.T
    Ns = randperm(rng, longest - 1)[1:2] .+ 1
    map(eachindex(Ns)) do j
        N = Ns[j]
        K = s.task == :prefix ? N + rand(rng, 0:s.extra) : rand(rng, 1:s.c.T)
        x = rand(rng, BIT_IDS, N, B ÷ 2 + (j == 1 ? B % 2 : 0))
        (; x, K, y = targets(s, x, K))
    end
end

function mixed_loss(s, c, θ, batch; gradient = true)
    objective(g) = begin
        single = s.mode in (:one, :wide)
        K = single ? 1 : g.K
        y = single && g.y isa AbstractVector ? last(g.y) : g.y
        supervision = single ? :last : s.supervision == :window ? :deep : s.supervision
        objective_op(s)(θ, g.x, y, c; supervision, loop_options(s, K)...)
    end
    B = sum(g -> size(g.x, 2), batch)
    ℓ = 0.0
    g = zero(θ)
    for group in batch
        w = size(group.x, 2) / B
        ℓgroup, back = objective(group)
        ℓ += w * ℓgroup
        gradient && (g .+= back(w))
    end
    ℓ, g
end

function train(s, seed)
    BLAS.set_num_threads(1)
    c = shape(s)
    recipe = s.recipe
    (; steps, η, ηmin, warmup, λ, clip, β1, β2, ε) = recipe
    rng_init = Xoshiro(seed)
    if uses_register(s)
        θ = register_init(c, Float64; rng = rng_init)
        λ = repeat(vcat(fill(λ, 2c.F), zeros(c.F), fill(λ, c.F), 0.0), c.L)
    else
        θ = init(c, Float64; rng = rng_init)
        λ = decay(c, λ)
    end
    opt = AdamW(θ; β1, β2, ε, λ)
    rng = Xoshiro(1000 + seed)
    fixed = [group for j in 1:8 for group in mixed_batch(s, Xoshiro(8000+j), steps)]
    mixed_loss(s, c, θ, mixed_batch(s, copy(rng), 1))
    initial = first(mixed_loss(s, c, θ, fixed; gradient = false))
    losses = zeros(steps)
    elapsed = @elapsed for t in 1:steps
        losses[t], g = mixed_loss(s, c, θ, mixed_batch(s, rng, t))
        update!(opt, θ, clip!(g, clip), warmup_cosine(t; η, ηmin, warmup, steps))
    end
    final = first(mixed_loss(s, c, θ, fixed; gradient = false))
    (; θ, c, train_seconds = elapsed, losses, initial, final)
end

function attention_stats(A, boundary)
    T, _, H, B = size(A)
    T == 1 && return (;)
    left = [mean(A[t,t-1,h,b] for t in 2:T, b in 1:B) for h in 1:H]
    self = [mean(A[t,t,h,b] for t in 2:T, b in 1:B) for h in 1:H]
    previous_argmax = [mean(argmax(view(A,t,:,h,b)) == t-1 for t in 2:T, b in 1:B) for h in 1:H]
    (; left, self, previous_argmax, boundary)
end

function state_diagnostics(s, fitted, x, K)
    trace = NamedTuple[]
    forward(s)(fitted.θ, x, fitted.c; trace, loop_options(s, K)...)
    map(1:K) do k
        prev, next = trace[k], trace[k+1]
        d = next.X - prev.X
        boundary = hasproperty(next, :boundary_attention) ?
            vec(mean(next.boundary_attention; dims = 2)) : nothing
        (; pass = k, stream_rms = sqrt(mean(abs2, next.X)), delta_rms = sqrt(mean(abs2, d)),
           delta_max = maximum(abs, d), attention = attention_stats(next.A, boundary))
    end
end

function evaluation(s, fitted, T, K; B = 128, diagnostics = true)
    rng = Xoshiro(5200 + T)
    actual_K = s.mode == :tied ? K : s.mode == :untied ? min(K, fitted.c.L) : 1
    hits = zeros(Int, T, actual_K)
    step_hits = zeros(Int, actual_K)
    exact = zeros(Int, actual_K)
    requested_hits = zeros(Int, s.task == :rule60 && actual_K == 1 ? K : 0)
    for batch in Iterators.partition(1:B, 16)
        x = rand(rng, BIT_IDS, T, length(batch))
        Zs, _ = forward(s)(fitted.θ, x, fitted.c; readouts = :all, loop_options(s, actual_K)...)
        y = s.task == :prefix ? prefix(x) : x
        fixed_target = s.task == :rule60 ? lucas(x, K) : nothing
        for (k, Z) in enumerate(Zs)
            if s.task == :rule60
                y = rule60_step(y)
                @assert y == lucas(x, k)
            end
            predicted = predictions(Z, T)
            hits[:, k] .+= vec(sum(predicted .== y; dims = 2))
            exact[k] += count(vec(all(predicted .== y; dims = 1)))
            matched = s.task == :prefix ? windowed_parity(x, k) : fixed_target
            step_hits[k] += count(predicted .== matched)
            for horizon in eachindex(requested_hits)
                requested_hits[horizon] += count(predicted .== lucas(x, horizon))
            end
        end
    end
    passes = map(1:actual_K) do k
        by_position = hits[:, k] / B
        front = something(findfirst(<(0.99), by_position), T+1) - 1
        (; pass = k, accuracy = mean(by_position), by_position, front,
           exact_sequences = exact[k] / B, matched_accuracy = step_hits[k] / (T*B))
    end
    trace_bits = rand(Xoshiro(1200+T), BIT_IDS, T, 4)
    trace = diagnostics ? state_diagnostics(s, fitted, trace_bits, actual_K) : nothing
    (; length = T, examples = B, horizon = K, passes, trace,
       single_pass_by_requested_horizon = requested_hits / (T*B))
end

function impulse_evaluation(s, fitted, K)
    x = ones(Int, 2K + 1, 1)
    x[1] = 2
    Zs, _ = forward(s)(fitted.θ, x, fitted.c; readouts = :all, loop_options(s, K)...)
    map(1:K) do k
        expected, predicted = vec(lucas(x, k)), vec(predictions(Zs[k], size(x, 1)))
        (; pass = k, exact = predicted == expected, expected, predicted)
    end
end

function binary_gate(s, fitted)
    uses_register(s) && s.mode == :tied || return nothing
    x = [1 1 2 2; 1 2 1 2]
    Z, _ = registergpt(fitted.θ, x, fitted.c; loop_options(s, 1)...)
    logits = vec(reshape(Z[2, :], 2, :)[2, :])
    expected = bit_xor.(x[1, :], x[2, :])
    margins = logits .* (2 .* expected .- 3)
    (; left = x[1, :], current = x[2, :], expected, logits, margins,
       exact_clipped_gate = s.options.bounded && minimum(margins) ≥ 1)
end

function run(s, seed; screen = false)
    fitted = train(s, seed)
    T = s.c.T
    test_K = 8T
    train_metrics = [evaluation(s, fitted, n, s.task == :prefix ? n + s.extra : T;
                                B = screen ? 24 : 128, diagnostics = !screen && n == T)
                     for n in (screen ? (T,) : (2:T))]
    extrapolation = evaluation(s, fitted, 4T, test_K; B = screen ? 24 : 128, diagnostics = !screen)
    at_length = s.scale_by_depth ? evaluation(s, fitted, 4T, 4T; B = screen ? 24 : 128,
                                                diagnostics = !screen) : extrapolation
    single_rule60 = s.task == :rule60 && s.mode in (:one, :wide)
    horizon_accuracy(e) = mean(view(e.single_pass_by_requested_horizon, 1:T))
    trained = if single_rule60
        minimum(horizon_accuracy, train_metrics)
    else
        minimum(p.accuracy for e in train_metrics for p in e.passes
                if s.task == :rule60 || p.pass >= min(e.length, length(e.passes)))
    end
    k_reference = 4T
    beyond = [p.accuracy for p in extrapolation.passes if p.pass >= k_reference]
    long_accuracy = single_rule60 ? horizon_accuracy(at_length) :
                    at_length.passes[min(k_reference, end)].accuracy
    extra_accuracy = isempty(beyond) ? nothing : minimum(beyond)
    @printf("%-20s seed %d P=%d loss %.4f → %.4f train %.4f long %.4f extra %s %.2fs\n",
            s.name, seed, parameter_count(s, fitted.c), fitted.initial, fitted.final, trained,
            long_accuracy, isnothing(extra_accuracy) ? "—" : @sprintf("%.4f", extra_accuracy),
            fitted.train_seconds)
    flush(stdout)
    impulse = !screen && s.task == :rule60 && s.mode == :tied ? impulse_evaluation(s, fitted, test_K) : nothing
    qualifies = s.mode == :tied && trained ≥ 0.99 && !isempty(beyond) &&
                at_length.passes[min(k_reference, end)].accuracy ≥ 0.99 &&
                minimum(beyond) ≥ 0.99 && fitted.train_seconds ≤ 60 &&
                (s.task == :prefix || minimum(p.accuracy for p in extrapolation.passes) ≥ 0.99)
    architecture = uses_register(s) ? :register : :stock
    (; setting = merge(s, (; c = config(s.c), architecture)), seed, config = config(fitted.c),
       nparams = parameter_count(s, fitted.c), fitted.θ, fitted.train_seconds, fitted.losses, fitted.initial,
       fitted.final, train_metrics, extrapolation, train_accuracy = trained, long_accuracy, extra_accuracy,
       extrapolation_at_length = s.scale_by_depth ? at_length : nothing,
       impulse, gate = binary_gate(s, fitted), qualifies)
end

const LOCAL = (; attention_span = :adjacent, previous_bias = 3.0, position_code = false)
const HARD = (; attention_span = :previous, position_code = false)

function screen_settings()
    [
        setting("stock-after"; F = 12),
        setting("stock-before"; F = 12, injection = :before),
        setting("stock-normalized"; F = 12, stabilize = true),
        setting("stock-scaled"; F = 12, scale_by_depth = true),
        setting("stock-refresh"; F = 12, injection = :refresh),
        setting("stock-deep"; F = 12, supervision = :deep),
        setting("stock-D16"; D = 16, F = 24),
        setting("stock-nope"; F = 12, position_code = false),
        setting("stock-curriculum"; F = 12, curriculum = true),
        setting("stock-fast"; F = 12, recipe = (; SEARCH_RECIPE..., η = 0.012, ηmin = 0.0012)),
        setting("relative-full"; previous_bias = 4.0, position_code = false, injection = :refresh,
                supervision = :window),
        setting("adjacent-soft"; LOCAL..., previous_bias = 1e-12, injection = :refresh,
                supervision = :window),
        setting("adjacent-biased"; LOCAL..., injection = :refresh, supervision = :window),
        setting("adjacent-normal"; LOCAL..., stabilize = true, supervision = :window),
        setting("hard-refresh"; HARD..., injection = :refresh, supervision = :window),
        setting("hard-cone"; HARD..., injection = :refresh, supervision = :cone),
        setting("hard-final"; HARD..., injection = :refresh),
        setting("hard-curriculum"; HARD..., injection = :refresh, curriculum = true, supervision = :window),
        setting("hard-extra"; HARD..., injection = :refresh, extra = 4, supervision = :window),
        register_setting("register-window"; supervision = :window),
        register_setting("register-cone"; supervision = :cone),
        register_setting("register-final"),
        register_setting("register-four"; F = 4, supervision = :window),
        register_setting("register-four-final"; F = 4),
        register_setting("register-three-final"; F = 3),
        register_setting("register-two-final"; F = 2),
        register_setting("register-unbounded"; bounded = false),
        register_setting("register-curriculum"; curriculum = true, supervision = :window),
        register_setting("register-extra"; extra = 4, supervision = :window),
    ]
end

const REGISTER_RECIPE = (; SEARCH_RECIPE..., steps = 450, batch = 16,
                         η = 0.015, ηmin = 0.0015)

register_setting(name; recipe = REGISTER_RECIPE, kw...) =
    setting(name; architecture = :register, D = 2, F = 8, recipe, kw...)

function prefix_settings()
    vcat([register_setting("prefix-" * string(mode); mode, supervision = :window)
          for mode in (:tied, :one, :untied, :wide)],
         [register_setting("prefix-compact-" * string(mode); F = 4, mode)
          for mode in (:tied, :one, :untied, :wide)],
         [register_setting("prefix-three-" * string(mode); F = 3, mode)
          for mode in (:tied, :one, :untied, :wide)],
         [setting("adjacent-five"; LOCAL..., injection = :refresh, supervision = :window),
          register_setting("prefix-cone"; supervision = :cone),
          register_setting("prefix-cone-untied"; mode = :untied, supervision = :cone),
          register_setting("prefix-final"),
          register_setting("prefix-final-one"; mode = :one),
          register_setting("prefix-final-untied"; mode = :untied),
          register_setting("prefix-final-wide"; mode = :wide),
          register_setting("prefix-unbounded"; bounded = false),
          register_setting("prefix-unbounded-untied"; mode = :untied, bounded = false),
          setting("stock-diagnostics"; F = 12, recipe = SEARCH_RECIPE)])
end

function rule60_settings()
    vcat([register_setting("rule60-" * string(mode); mode, task = :rule60, supervision = :deep)
          for mode in (:tied, :one, :untied, :wide)],
         [register_setting("rule60-compact-" * string(mode); F = 4, mode, task = :rule60, supervision = :deep)
          for mode in (:tied, :one, :untied, :wide)],
         [register_setting("rule60-three-" * string(mode); F = 3, mode, task = :rule60, supervision = :deep)
          for mode in (:tied, :one, :untied, :wide)],
         [register_setting("rule60-final"; task = :rule60),
          register_setting("rule60-final-untied"; mode = :untied, task = :rule60),
          register_setting("rule60-unbounded"; task = :rule60, supervision = :deep, bounded = false),
          setting("rule60-soft"; task = :rule60, LOCAL..., injection = :none, stabilize = true,
                  supervision = :deep),
          setting("rule60-hard"; task = :rule60, HARD..., injection = :none, stabilize = true,
                  supervision = :deep)])
end

function main(args)
    group = isempty(args) ? "screen" : first(args)
    settings = group == "screen" ? screen_settings() : group == "prefix" ? prefix_settings() :
               group == "rule60" ? rule60_settings() : throw(ArgumentError("expected screen, prefix, or rule60"))
    names = length(args) > 1 ? Set(args[2:end]) : Set(s.name for s in settings)
    isempty(setdiff(names, Set(s.name for s in settings))) || throw(ArgumentError("unknown setting: $names"))
    seeds = get(ENV, "ROUND2_SEEDS", group == "screen" ? "1" : "1,2,3,4,5")
    for s in settings
        s.name in names || continue
        runs = [run(s, parse(Int, seed); screen = group == "screen") for seed in split(seeds, ',')]
        save_experiment("round2-" * (group == "screen" ? "screen-" : "") * s.name, (; runs))
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
