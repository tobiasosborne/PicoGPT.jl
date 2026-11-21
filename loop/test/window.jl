# Run from the repository root: julia --project -O1 loop/test/window.jl
using Test
include(joinpath(@__DIR__, "../scripts/window_round3.jl"))

BLAS.set_num_threads(1)
Random.seed!(123)

@testset "Two-token additive attention" begin
    T, H, B, D = 5, 2, 3, 6
    Q, K, V = (randn(D, T * B) for _ in 1:3)
    M = window_mask(T)
    @test M[1, :] == [0, -Inf, -Inf, -Inf, -Inf]
    A, _ = masked_attention(Q, K, H, M)
    @test all(isone, A[1, 1, :, :])
    @test sum(A; dims = 2) ≈ ones(T, 1, H, B)
    @test all(I -> all(iszero, A[I, :, :]), findall(==(-Inf), M))
    @test first(masked_attention(Q, K, H, window_mask(T, T))) ≈ first(attention(Q, K, H, T))
    errors = checkpullback((q, k) -> masked_attention(q, k, H, M), Q, K; wrt = (1, 2))
    @test maximum(errors) < 1e-7

    function dense_mix(q, k, v)
        a, back_a = masked_attention(q, k, H, M)
        o, back_o = mix(a, v)
        function back(ō)
            ā, v̄ = back_o(ō)
            q̄, k̄ = back_a(ā)
            q̄, k̄, v̄
        end
        o, back
    end
    dense, dense_back = dense_mix(Q, K, V)
    compact, compact_back, weights = WindowLoop.window_mix(Q, K, V, H, T)
    @test compact ≈ dense
    for b in 1:B, h in 1:H, t in 1:T
        n = (b - 1) * T + t
        @test weights[1, h, n] ≈ A[t, t, h, b]
        @test weights[2, h, n] ≈ (t == 1 ? 0 : A[t, t-1, h, b])
    end
    ō = randn(size(V))
    @test all(isapprox.(compact_back(ō), dense_back(ō)))
    function compact_mix(q, k, v)
        o, back, _ = WindowLoop.window_mix(q, k, v, H, T)
        o, back
    end
    @test maximum(checkpullback(compact_mix, Q, K, V; wrt = (1, 2, 3))) < 1e-7

    # At T=1, no keys or queries can influence the output, even across words.
    o, back, weights = WindowLoop.window_mix(Q, K, V, H, 1)
    @test o == V
    @test all(isone, weights[1, :, :])
    @test all(iszero, weights[2, :, :])
    q̄, k̄, v̄ = back(ō)
    @test all(iszero, q̄) && all(iszero, k̄) && v̄ == ō
end

@testset "Round 3 boundary, residual, and warmup" begin
    T, K = 4, 3
    x = binary_words(T)[:, 4:7]
    y = prefix(x)
    c = Config(; V = 2, D = 4, H = 1, F = 4, T, L = 1)
    for (position, injection, lag_bias, start) in ((:bos, :normalized, 0.0, :embedded),
                                                  (:bos, :anchor_normalized, 2.0, :embedded),
                                                  (:boundary, :anchor, 2.0, :embedded),
                                                  (:boundary, :anchor_normalized, 2.0, :embedded),
                                                  (:boundary, :anchor_normalized, 4.0, :boundary))
        spec = round3_spec("gradcheck"; c.D, c.F, position, injection, lag_bias, start)
        θ = round3_init(c, 36, spec)
        opts = (; position, injection, lag_bias, start)
        objective(p) = windowloss(p, x, y, c; K, conditional = true, opts...)
        @test only(checkpullback(objective, θ; wrt = (1,))) < 1e-6
        trace = NamedTuple[]
        Zs, _ = windowgpt(θ, x, c; K, readouts = :all, trace, opts...)
        WindowLoop.visit_passes(θ, x, c; K, opts...) do k, Z, X, _, weights
            @test Z == Zs[k]
            @test size(X) == (c.D, length(x))
            @test size(weights) == (2, c.H, length(x))
        end
        if position == :bos
            @test size(trace[end].X, 2) == (T + 1) * size(x, 2)
            @test all(isone, trace[end].weights[1, 1, 1:T+1:end])
            @test all(>(0), trace[end].weights[2, 1, 2:T+1:end])
        elseif start == :boundary
            @test all(iszero, reshape(trace[1].X, c.D, T, :)[:, 2:end, :])
            @test any(!iszero, trace[1].X[:, 1:T:end])
        end
    end

    spec = round3_spec("teacher-gradcheck"; c.D, c.F, injection = :anchor_normalized, lag_bias = 4.0)
    θ = round3_init(c, 36, spec)
    codes = teacher_codes(c.D)
    for step in (1, 2)
        function objective(p)
            loss, grad = teacher_step(p, c, spec, codes, Xoshiro(37), step)
            loss, Δ -> Δ .* grad
        end
        @test only(checkpullback(objective, θ; wrt = (1,))) < 1e-6
    end
    rule60 = blind_rule60()
    @test rule60.conflicts == [0; fill(128, 7)]
    @test rule60.disagreement == (7, 16)
    @test rule60.accuracy_ceiling == (25, 32)
    @test bos_floor() ≈ 0.20060266259274695
    @test bos_floor() > 1.01ROUND3_NATS
    config = (; c.V, c.D, c.H, c.F, c.L, c.T)
    trace = specimen_trace((; spec, config, θ))
    @test getproperty.(trace.frames, :pass) == collect(0:16)
    @test trace.frames[1].attention_weights === nothing
    @test all(frame -> frame.probabilities.shape == [2, 16], trace.frames)
    @test all(frame -> frame.terminal_covector.shape == [c.D, 16], trace.frames)
end

@testset "Round 2 floors and stream pullbacks" begin
    floors = round2_floors()
    @test floors.fixed.nats ≈ 21log(2) / 64
    @test floors.mixed.nats ≈ (384975233676617 // 1653814216454400) * log(2)
    @test floors.one.nats ≈ (89539 // 124740) * log(2)
    @test floors.nope_deep_floor > floors.mixed.nats
    @test all(row -> row.ceiling < 0.85, floors.nope_terminal)

    T, K = 4, 3
    x = binary_words(T)[:, 1:5]
    y = prefix(x)
    c = Config(; V = 2, D = 4, H = 1, F = 3, T, L = 1)
    θ = init(c; rng = Xoshiro(81))
    for (position, injection) in ((:none, :after), (:boundary, :after),
                                  (:boundary, :refresh), (:boundary, :normalized),
                                  (:none, :none))
        objective(p) = windowloss(p, x, y, c; K, position, injection, conditional = true)
        @test only(checkpullback(objective, θ; wrt = (1,))) < 1e-6
        Zs, _ = windowgpt(θ, x, c; K, position, injection, readouts = :all)
        WindowLoop.visit_passes(θ, x, c; K, position, injection) do k, Z, X, previous, weights
            @test Z == Zs[k]
            if injection == :normalized
                @test maximum(abs, vec(sum(X; dims = 1))) < 1e-12
                @test maximum(norm, eachcol(X)) ≤ sqrt(c.D)
            end
        end
    end

    homogeneous = fill(2, T, 1)
    Z, _ = windowgpt(θ, homogeneous, c; K, position = :none)
    @test Z ≈ repeat(Z[:, 1:1], 1, T)
    Z, _ = windowgpt(θ, homogeneous, c; K, position = :boundary)
    @test Z[:, 1] != Z[:, 2]
end

@testset "Round 2 predecessor register pullbacks" begin
    T, K = 5, 3
    x = binary_words(T)[:, 1:8]
    y = prefix(x)
    for (transition, tied) in ((:gain, true), (:mlp, true), (:gain, false))
        c = Config(; V = 2, D = 2, H = 1, F = 3, T, L = tied ? 1 : K)
        θ = init_register(c, transition; rng = Xoshiro(82))
        objective(p) = register_windowloss(p, x, y, c; K, tied, transition, conditional = true)
        @test only(checkpullback(objective, θ; wrt = (1,))) < 1e-6
    end
    c = Config(; V = 2, D = 2, H = 1, F = 1, T, L = 1)
    θ = [3.0, -3.0]
    Zs, _ = register_window(θ, x, c; K, readouts = :all)
    for k in 1:K
        @test reshape(Zs[k][2, :] .> 0, size(x))[1:k+1, :] == (y .== 2)[1:k+1, :]
        @test all(iszero, reshape(Zs[k][2, :], size(x))[k+2:end, :])
    end

    word = reshape([2, 1, 2, 1, 1], :, 1)
    z, back = register_window(θ, word, c; K)
    Δ = zero(z)
    Δ[2, T], Δ[1, T] = 1, -1
    adjoints = NamedTuple[]
    @test back(Δ; adjoints) == back(Δ)
    for frame in adjoints
        site = T - (K - frame.pass)
        @test findall(!iszero, vec(frame.X̄)) == [site]
    end

    targets = rule60_targets(x, K)
    @test targets == [lucas_window(x, k) for k in 1:K]
    rule_objective(p) = register_windowloss(p, x, targets, c; K, task = :rule60)
    @test only(checkpullback(rule_objective, [1.7, -1.9]; wrt = (1,))) < 1e-6
end

@testset "Prefix floor and loop pullbacks" begin
    T, K = 4, 3
    x = binary_words(T)
    y = prefix(x)
    verified = verify_floor(8, 8)
    @test verified.words == 256
    @test verified.deep ≈ 21log(2) / 64
    @test first(verified.checks).entropy ≈ 6log(2) / 8
    for (tied, supervision) in ((true, :deep), (false, :last))
        c = Config(; V = 2, D = 4, H = 1, F = 3, T, L = tied ? 1 : K)
        θ = init(c; rng = Xoshiro(232))
        objective(p) = windowloss(p, x, y, c; K, tied, supervision)
        @test only(checkpullback(objective, θ; wrt = (1,))) < 1e-6
    end

    c = Config(; V = 2, D = 4, H = 1, F = 3, T, L = 1)
    θ = init(c; rng = Xoshiro(12))
    a, back_a = windowloss(θ, x, y, c; K, supervision = :deep)
    b, back_b = windowloss(θ, x, y, c; K, supervision = :deep, conditional = true)
    @test a ≈ b
    @test back_a() ≈ back_b()

    x = reshape([1, 1, 2, 1], :, 1)
    Z, _ = windowgpt(θ, x, c; K, readouts = :all)
    changed = copy(x)
    changed[1] = 2
    F, _ = windowgpt(θ, changed, c; K, readouts = :all)
    for k in 1:K
        @test Z[k][:, k+2:end] == F[k][:, k+2:end]
    end
    @test Z[1][:, 2] != F[1][:, 2] # first pass already sees the predecessor

    z, back = windowgpt(θ, x, c; K)
    Δ = zero(z)
    Δ[2, T], Δ[1, T] = 1, -1
    adjoints = NamedTuple[]
    θ̄ = back(Δ; adjoints)
    @test θ̄ == back(Δ)
    @test getproperty.(adjoints, :pass) == collect(K:-1:0)
    for frame in adjoints
        left = T - (K - frame.pass)
        @test all(iszero, frame.X̄[:, 1:left-1])
        @test norm(frame.X̄[:, left]) > 0
    end
end
