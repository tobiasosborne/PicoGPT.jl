# Run from the repository root with julia --project -O1 -t 1.
include(joinpath(@__DIR__, "window_round2.jl"))

const ROUND3_FLOOR = distribution_mean(floor_fraction, MIXED)
const ROUND3_NATS = ROUND3_FLOOR * log(2)

floor_record() = (; numerator = numerator(ROUND3_FLOOR), denominator = denominator(ROUND3_FLOOR),
                  factor = "ln(2)", nats = ROUND3_NATS)

function bos_floor()
    pass_floor(T, k) = if k ≥ T - 1
        0.0
    else
        m = T - k
        p = (m + 1) / (2m)
        -m / T * (p * log(p) + (1 - p) * log1p(-p))
    end
    distribution_mean(MIXED) do T, K
        mean(pass_floor(T, k) for k in 1:K)
    end
end

"""Enumerate both Rule 60 labels for every word and query, with no horizon input."""
function blind_rule60(T = 8)
    conflicts = zeros(Int, T)
    for word in 0:(1 << T)-1
        bits = [isodd(word >> t) for t in 0:T-1]
        tick(v) = xor.(v, [false; v[1:end-1]])
        a = tick(bits)
        b = tick(a)
        conflicts .+= a .!= b
    end
    examples = T * (1 << T)
    disagreements = sum(conflicts) // examples
    accuracy = 1 - disagreements / 2
    (; T, words = 1 << T, examples, conflicts,
       disagreement = (numerator(disagreements), denominator(disagreements)),
       ce_nats = disagreements * log(2),
       accuracy_ceiling = (numerator(accuracy), denominator(accuracy)))
end

function round3_spec(name; D = 4, F = 16, position = :boundary, injection = :normalized,
                     initialization = :stock, curriculum = :mixed, steps = 1250,
                     batch = 8, η = 0.02, warmup = 20, λ = 0.01, clip = 1.0, lag_bias = 0,
                     teacher_steps = 0, teacher_layout = :simplex, polish_steps = 0, start = :embedded)
    (; name, D, F, position, injection, initialization, curriculum, steps, batch, η, warmup, λ, clip,
       lag_bias, teacher_steps, teacher_layout, polish_steps, start)
end

function round3_init(c, seed, spec)
    rng = Xoshiro(seed)
    θ = init(c; rng)
    p = carve(θ, c)
    b = only(p.blocks)
    if spec.initialization in (:uniform, :near_identity)
        # Keep both factors nonzero so query/key gradients can move attention.
        b.Wq .*= 0.01
        b.Wk .*= 0.01
    end
    if spec.initialization == :near_identity
        b.Wo .*= 0.1
        b.W2 .*= 0.1
        p.Wout .*= 0.1
    end
    spec.position == :bos ? vcat(θ, randn(rng, c.D)) : θ
end

function round3_batch(rng, step, spec)
    T = if spec.curriculum == :grow && step ≤ 40
        2
    elseif spec.curriculum == :grow && step ≤ 150
        rand(rng, 2:4)
    elseif spec.curriculum == :grow && step ≤ 300
        rand(rng, 3:8)
    else
        rand(rng, MIXED_LENGTHS)
    end
    K = rand(rng, T-1:2T)
    rand(rng, 1:2, T, spec.batch), K
end

# Train a one-step block with supplied latent targets before unrolling it.
# The three codes are the vertices of a centred simplex for parity 0, 1,
# and unknown. They are supervision used during warmup only.
function teacher_codes(D, layout = :simplex)
    D ≥ 3 || throw(ArgumentError("the three warmup codes need at least three features"))
    codes = zeros(D, 3)
    if layout == :arc
        # Put the undecided code between the two known signs on the sphere.
        u = [1, 1, -2] / sqrt(6)
        v = [1, -1, 0] / sqrt(2)
        codes[1:3, :] .= sqrt(D) .* hcat((u - v) / sqrt(2), (u + v) / sqrt(2), u)
    else
        codes[1:3, :] .= (Matrix{Float64}(I, 3, 3) .- 1/3) .* sqrt(3D/2)
    end
    codes
end

function polish_round3!(θ, spec, seed, rates, objective)
    losses = zeros(spec.polish_steps)
    iszero(spec.polish_steps) && return losses, 0.0
    rng = Xoshiro(3000 + seed)
    opt = AdamW(θ; β1 = 0.9, β2 = 0.95, ε = 1e-8, λ = rates)
    elapsed = @elapsed for step in eachindex(losses)
        T = rand(rng, MIXED_LENGTHS)
        # Put two thirds of the updates on shallow rollouts, where the
        # newest known site and undecided exterior contribute most.
        K = step % 3 == 0 ? rand(rng, T-1:2T) : rand(rng, 1:3)
        x = rand(rng, 1:2, T, spec.batch)
        losses[step], pull = objective(θ, x, K)
        rate = warmup_cosine(step; η = spec.η / 5, warmup = 0, steps = spec.polish_steps)
        update!(opt, θ, clip!(pull(), spec.clip), rate)
    end
    losses, elapsed
end

function teacher_step(θ, c, spec, codes, rng, step)
    x = rand(rng, 1:2, 6, 24)
    y = prefix(x)
    k = step % 3 == 1 ? 1 : rand(rng, 2:6)
    p, source, back_embed, T, sites, _ = WindowLoop.window_start(θ, x, c, spec.position)
    ids = [t ≤ k + 1 ? y[t, b] : 3 for t in axes(y, 1), b in axes(y, 2)]
    target = codes[:, vec(ids)]
    X = if k == 1
        source
    else
        @views ids[k+1:end, :] .= 3
        latent = codes[:, vec(ids)]
        latent .+= 0.05 .* randn(rng, size(latent))
        latent
    end
    Y, pull_block, _, pull_norm = WindowLoop.advance(
        X, source, only(p.blocks), c.H, T, spec.injection, spec.lag_bias)
    Z, pull_readout = WindowLoop.readout(view(Y, :, sites), p)
    ce, pull_ce = WindowLoop.prefix_term(Z, y, k + 1)
    dY, d_readout = pull_readout(pull_ce(0.2))
    err = Y - target
    dY .+= (2 / length(err)) .* err
    Δ = pull_norm(dY)
    dX, d_block, d_anchor = pull_block(Δ)
    d_source = isnothing(d_anchor) ? (spec.injection == :none ? zero(source) : Δ) : d_anchor
    k == 1 && (d_source = d_source + dX)
    sum(abs2, err) / length(err) + 0.2ce, flatten(back_embed(d_source), d_block, d_readout)
end

function round3_rollout(θ, c, T, opts; diagnose)
    x = eval_words(T)
    B = size(x, 2)
    y = prefix(x) .== 2
    accuracy = zeros(2T)
    fronts = zeros(Int, 2T)
    diagnostics = NamedTuple[]
    WindowLoop.visit_passes(θ, x, c; K = 2T, opts...) do k, Z, X, previous, weights
        logits = reshape(view(Z, 2, :) - view(Z, 1, :), T, B)
        hit_by_site = vec(mean((logits .> 0) .== y; dims = 2))
        accuracy[k] = mean(hit_by_site)
        fronts[k] = something(findfirst(<(0.99), hit_by_site), T + 1) - 1
        if diagnose && k in (1, 4, 16, T, 2T)
            prev = view(reshape(weights, 2, c.H, T, B), 2, :, 2:T, :)
            posterior = [t ≤ k + 1 ? y[t, b] : 0.5 for t in 1:8, b in 1:B]
            errors = abs.(probability.(view(logits, 1:8, :)) .- posterior)
            push!(diagnostics, (; k, front = fronts[k], accuracy_inside = mean(hit_by_site[1:min(k+1,T)]),
                                centered_state_rms = norm(center(X)) / sqrt(B*T),
                                centered_step_rms = norm(center(X - previous)) / sqrt(B*T),
                                predecessor_mean = mean(prev), predecessor_min = minimum(prev),
                                predecessor_max = maximum(prev), hard_self = mean(prev .≤ 0.05),
                                hard_previous = mean(prev .≥ 0.95), first_eight_accuracy = hit_by_site[1:8],
                                first_eight_inside_cone_error = vec(mean(errors; dims = 2))))
        end
    end
    (; accuracy = accuracy[[T, 2T]], fronts, diagnostics)
end

function run_round3(spec, seed; log_every = spec.steps, diagnose = false, full_curve = false)
    c = config2(spec.D, spec.F, MIXED, true)
    opts = (; spec.position, spec.injection, spec.lag_bias, spec.start)
    visitor(f, θ, c_input, config; K) = WindowLoop.visit_passes(f, θ, c_input, config; K, opts...)
    objective(θ, x, K) = windowloss(θ, x, prefix(x), c; K, conditional = true, opts...)
    validate(θ) = validation_prefix(θ, c, MIXED, visitor)
    θ = round3_init(c, seed, spec)
    rates = decay(c, spec.λ)
    spec.position == :bos && append!(rates, fill(spec.λ, c.D))
    opt = AdamW(θ; β1 = 0.9, β2 = 0.95, ε = 1e-8, λ = rates)
    rng = Xoshiro(1000 + seed)
    x, K = round3_batch(copy(rng), 1, spec)
    update!(deepcopy(opt), copy(θ), clip!(last(objective(θ, x, K))(), spec.clip), spec.η)
    elapsed = 0.0
    curve = NamedTuple[]
    (iszero(spec.teacher_steps) || full_curve) && push!(curve, (; step = 0, loss = validate(θ)))
    codes = spec.teacher_steps > 0 ? teacher_codes(c.D, spec.teacher_layout) : nothing
    teacher_rng = Xoshiro(2000 + seed)
    teacher_loss_before_update = zeros(spec.teacher_steps)
    if spec.teacher_steps > 0
        spec.position != :bos || throw(ArgumentError("teacher warmup uses data-site states"))
        teacher_step(θ, c, spec, codes, copy(teacher_rng), 1)
        elapsed += @elapsed for step in 1:spec.teacher_steps
            teacher_loss_before_update[step], grad = teacher_step(θ, c, spec, codes, teacher_rng, step)
            update!(opt, θ, clip!(grad, spec.clip), spec.η)
        end
        opt = AdamW(θ; β1 = 0.9, β2 = 0.95, ε = 1e-8, λ = rates)
        push!(curve, (; step = spec.teacher_steps, loss = validate(θ)))
        @printf("%-22s seed=%d after teacher CE=%.7f\n", spec.name, seed, last(curve).loss)
        flush(stdout)
    end
    losses = zeros(spec.steps)
    for start in 1:log_every:spec.steps
        stop = min(start + log_every - 1, spec.steps)
        elapsed += @elapsed for step in start:stop
            x, K = round3_batch(rng, step, spec)
            losses[step], pull = objective(θ, x, K)
            rate = warmup_cosine(step; spec.η, spec.warmup, spec.steps)
            update!(opt, θ, clip!(pull(), spec.clip), rate)
        end
        push!(curve, (; step = stop + spec.teacher_steps, loss = validate(θ)))
    end
    polish_loss_before_update, polish_elapsed = polish_round3!(θ, spec, seed, rates, objective)
    elapsed += polish_elapsed
    if spec.polish_steps > 0
        push!(curve, (; step = spec.teacher_steps + spec.steps + spec.polish_steps, loss = validate(θ)))
    end
    evaluations = [round3_rollout(θ, c, T, opts; diagnose = diagnose && T == 64) for T in (32, 64)]
    diagnostics = last(evaluations).diagnostics
    gap = last(curve).loss - ROUND3_NATS
    rollout = vcat(getproperty.(evaluations, :accuracy)...)
    qualified = gap ≤ 0.01ROUND3_NATS && all(isone, rollout)
    parameters = length(θ)
    config = (; c.V, c.D, c.H, c.F, c.L, c.T)
    @printf("%-22s seed=%d P=%d CE=%.7f T32 %.5f/%.5f T64 %.5f/%.5f %6.2fs %s\n",
            spec.name, seed, parameters, last(curve).loss, rollout..., elapsed, qualified ? "PASS" : "")
    flush(stdout)
    (; spec, seed, config, θ, parameters, elapsed, curve, teacher_loss_before_update,
       minibatch_loss_before_update = losses, polish_loss_before_update, gap, rollout, qualified, diagnostics,
       fronts = getproperty.(evaluations, :fronts))
end

function round3_specs()
    [
        round3_spec("baseline"),
        round3_spec("bos"; position = :bos),
        round3_spec("uniform"; initialization = :uniform),
        round3_spec("near-identity"; initialization = :near_identity),
        round3_spec("grow"; curriculum = :grow),
        round3_spec("bos-grow"; position = :bos, curriculum = :grow),
        round3_spec("bos-identity"; position = :bos, initialization = :near_identity),
        round3_spec("bos-identity-grow"; position = :bos, initialization = :near_identity, curriculum = :grow),
        round3_spec("fresh-grow"; injection = :refresh, curriculum = :grow),
        round3_spec("anchor"; injection = :anchor_normalized),
        round3_spec("anchor-bias"; injection = :anchor_normalized, lag_bias = 2.0),
        round3_spec("anchor-polish"; injection = :anchor_normalized, lag_bias = 2.0, polish_steps = 300),
        round3_spec("anchor-bias-strong"; injection = :anchor_normalized, lag_bias = 4.0),
        round3_spec("anchor-boundary-start"; injection = :anchor_normalized, lag_bias = 4.0,
                    start = :boundary),
        round3_spec("anchor-bos"; injection = :anchor_normalized, lag_bias = 2.0, position = :bos),
        round3_spec("anchor-wide"; injection = :anchor_normalized, lag_bias = 4.0, D = 8, F = 32),
        round3_spec("teacher"; injection = :anchor_normalized, lag_bias = 4.0, D = 6, F = 16,
                    steps = 900, teacher_steps = 750),
        round3_spec("teacher-wide"; injection = :anchor_normalized, lag_bias = 4.0, D = 8, F = 32,
                    steps = 900, teacher_steps = 750),
        round3_spec("teacher-arc"; injection = :anchor_normalized, lag_bias = 4.0, D = 6, F = 16,
                    steps = 900, teacher_steps = 750, teacher_layout = :arc),
        round3_spec("teacher-control"; injection = :anchor_normalized, lag_bias = 4.0, D = 6, F = 16,
                    steps = 900),
    ]
end

function specimen_trace(run)
    c = Config(; run.config...)
    (; position, injection, lag_bias, start) = run.spec
    opts = (; position, injection, lag_bias, start)
    T, K = 16, 16
    x = rand(Xoshiro(723), 1:2, T, 1)
    trace, adjoints = NamedTuple[], NamedTuple[]
    Zs, back = windowgpt(run.θ, x, c; K, readouts = :all, trace, opts...)
    cotangents = [zero(z) for z in Zs]
    cotangents[K][2, T], cotangents[K][1, T] = 1, -1
    back(cotangents; adjoints)
    p = carve(view(run.θ, 1:nparams(c)), c)
    column_offset = position == :bos ? 1 : 0
    frames = map(zip(trace, reverse(adjoints))) do (frame, pull)
        k = frame.pass
        @assert pull.pass == k
        left = max(1, T + column_offset - (K - k))
        @assert all(iszero, view(pull.X̄, :, 1:left-1))
        Z = k == 0 ? first(WindowLoop.readout(frame.X[:, column_offset+1:end], p)) : Zs[k]
        (; pass = k, stream = tensor(frame.X),
           attention_weights = k == 0 ? nothing : tensor(frame.weights),
           probabilities = tensor(softmax(Z; dims = 1)),
           terminal_covector = tensor(pull.X̄))
    end
    (; T, K, input_bits = vec(x .- 1), target_bits = vec(prefix(x) .- 1), seed = 723,
       terminal = (; pass = K, data_site = T, observable = "class 2 logit minus class 1 logit",
                   differentiation = "stream cotangent with re-injected input held fixed"),
       attention_order = ["self", "predecessor"], pass_zero_readout = "shown for reference",
       column_offset, frames)
end

function round3_main(args)
    BLAS.set_num_threads(1)
    if args == ["rule60"]
        println(json(blind_rule60()))
        return
    end
    if first(args) == "trace"
        _, name = args
        spec = only(filter(s -> s.name == name, round3_specs()))
        c = config2(spec.D, spec.F, MIXED, true)
        config = (; c.V, c.D, c.H, c.F, c.L, c.T)
        θ = parse.(Float64, split(read(stdin, String)))
        println(json((; floor = floor_record(), rule60 = blind_rule60(),
                      trace = specimen_trace((; spec, config, θ)))))
        return
    end
    if first(args) == "specimen"
        _, name, seed = args
        spec = only(filter(s -> s.name == name, round3_specs()))
        run = run_round3(spec, parse(Int, seed); log_every = 300, diagnose = true, full_curve = true)
        path = joinpath(@__DIR__, "../artifacts/window-specimen.json")
        writejson(path, (; julia = string(VERSION), run..., floor = floor_record(),
                         rule60 = blind_rule60(), trace = specimen_trace(run)))
        return
    end
    names = split(first(args), ',')
    seeds = length(args) > 1 ? parse.(Int, args[2:end]) : collect(1:5)
    for name in names
        spec = only(filter(s -> s.name == name, round3_specs()))
        runs = run_batch(seed -> run_round3(spec, seed; diagnose = true), seeds)
        suffix = seeds == collect(1:5) ? "" : "-seeds-" * join(seeds, "-")
        path = joinpath(@__DIR__, "../artifacts/window-round3-$name$suffix.json")
        writejson(path, (; julia = string(VERSION), floor = floor_record(),
                         bos_floor_given_length_and_pass = bos_floor(), runs))
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && round3_main(ARGS)
