using PicoGPT, LinearAlgebra, Random, Test
include(joinpath(@__DIR__, "../src/Loop.jl"))
using .Loop

BLAS.set_num_threads(1)
Random.seed!(8)

@testset "Loop" begin
    c = Config(V = 2, D = 4, H = 2, L = 1, T = 3, F = 5)
    c₃ = Config(; c.V, c.D, c.H, c.T, c.F, L = 3)
    θ = init(c) + 0.05randn(nparams(c))
    x = [1 2; 1 1; 2 1]
    y = [1 2; 1 2; 2 2]

    @testset "arbitrary logit covectors and losses" begin
        for injection in (:none, :after, :before)
            logits(θ, x) = loopgpt(θ, x, c; K = 3, injection)
            logit_error = only(checkpullback(logits, θ, x; wrt = (1,)))
            @test logit_error < 1e-6
            for supervision in (:last, :deep)
                targets = supervision == :deep ? [x, y, reverse(y; dims = 1)] : y
                objective(θ) = looploss(θ, x, targets, c; K = 3, injection, supervision)
                ℓ, back = objective(θ)
                g = back()
                err = gradcheck(θ -> first(objective(θ)), θ, g)
                @info "Finite differences" injection supervision logit_error loss_error = err
                @test err < 1e-6
                @test back(2) ≈ 2g
                @test back() == g
            end
        end
    end

    @testset "controls, weight sharing, and traces" begin
        Z, back = loopgpt(θ, x, c; K = 1)
        Z₁, back₁ = gpt(θ, x, c)
        G = randn(size(Z))
        @test Z == Z₁
        @test back(G) ≈ back₁(G)
        p = carve(θ, c)
        θ₃ = flatten(p.E, fill(only(p.blocks), 3), p.γf, p.βf, p.Wout, p.bout)
        Z, back = loopgpt(θ, x, c; K = 3)
        Z₃, back₃ = loopgpt(θ₃, x, c₃; tied = false)
        @test Z == Z₃ == first(gpt(θ₃, x, c₃))
        g = carve(back₃(G), c₃)
        @test back₃(G) ≈ last(gpt(θ₃, x, c₃))(G)
        @test back(G) ≈ flatten(g.E, sum(flatten, g.blocks), g.γf, g.βf, g.Wout, g.bout)

        xlong = vcat(x, x)
        trace = NamedTuple[]
        Zs, _ = loopgpt(θ, xlong, c; K = 3, injection = :after, readouts = :all, trace)
        @test length(trace) == 4 && length(Zs) == 3
        @test trace[1].X == first(embed(xlong, p.E))
        for k in 1:3
            frame = trace[k + 1]
            input = trace[k].X
            @test frame.X == first(block(input, only(p.blocks), c.H, size(xlong,1))) + trace[1].X
            @test frame.pass == k && frame.Z == Zs[k]
            @test frame.predictions == predictions(Zs[k], size(xlong,1))
            @test size(frame.A) == (6, 6, 2, 2)
            @test all(sum(frame.A, dims = 2) .≈ 1)
            @test Zs[k][:, 1:3] ≈ first(loopgpt(θ, x, c; K = k, injection = :after))[:, 1:3]
        end
    end
end

@testset "refreshed and scaled relative loops" begin
    c = Config(V = 2, D = 4, H = 2, L = 1, T = 3, F = 5)
    θ = init(c; rng = Xoshiro(22)) + 0.05randn(nparams(c))
    x = [2 1; 1 2; 2 1]
    y = 3 .- x
    for (span, bias, mode, stabilized, scale) in (
        (:full, 2.0, :after, false, 0.5),
        (:full, 2.0, :refresh, true, 1.0),
        (:adjacent, 3.0, :refresh, false, 0.5),
        (:previous, 0.0, :before, true, 0.5),
        (:previous, 0.0, :refresh, false, 1.0),
    )
        opts = (; K = 3, attention_span = span, previous_bias = bias, injection = mode,
                  position_code = false, stabilize = stabilized, residual_scale = scale)
        objective(θ) = looploss(θ, x, y, c; supervision = :cone, opts...)
        _, pull = objective(θ)
        err = gradcheck(θ -> first(objective(θ)), θ, pull())
        @info "Round 2 finite differences" span mode stabilized scale loss_error = err
        @test err < 1e-6
    end

    p = only(carve(θ, c).blocks)
    X, _ = embed(x, carve(θ, c).E)
    oldtrace, newtrace = NamedTuple[], NamedTuple[]
    Y, back_Y = Loop.adjacent_block(X, p, c.H, c.T; trace = oldtrace)
    Z, back_Z = Loop.relative_block(X, p, c.H, c.T, :adjacent, 0; trace = newtrace)
    G = randn(size(Y))
    @test Z ≈ Y
    @test only(newtrace).A ≈ only(oldtrace).A
    @test only(newtrace).boundary_attention ≈ only(oldtrace).boundary_attention
    @test flatten(back_Z(G)...) ≈ flatten(back_Y(G)...)
end

@testset "scalar predecessor register" begin
    c = Config(V = 2, D = 2, H = 1, L = 1, T = 3, F = 6)
    c₃ = Config(V = 2, D = 2, H = 1, L = 3, T = 3, F = 6)
    θ = register_init(c; rng = Xoshiro(57))
    x = [2 1; 1 2; 2 1]
    y = 3 .- x
    for input in (:source, :state), bounded in (false, true)
        # Moving the logits across the clipping threshold exercises both parts
        # of the pullback while keeping the finite differences off the kinks.
        weights = copy(θ)
        weights[end] += 0.8
        objective(w) = registerloss(w, x, [y, x, y], c; K = 3, input, bounded, supervision = :deep)
        _, back = objective(weights)
        err = gradcheck(w -> first(objective(w)), weights, back())
        @info "Register finite differences" input bounded loss_error = err
        @test err < 1e-6
    end
    trace = NamedTuple[]
    registergpt(θ, x, c; K = 5, trace)
    @test trace[1].X == reshape(2 .* x .- 3, 1, :)
    @test trace[4].X == trace[5].X == trace[6].X
    @test trace[2].A[2,1,1,2] == 1
    @test trace[2].boundary_attention == ones(1, 2)

    θ₃ = repeat(θ, 3)
    Z, back = registergpt(θ₃, x, c₃; K = 2, tied = false)
    Z₂, back₂ = registergpt(θ, x, c; K = 2)
    @test Z == Z₂
    G = randn(size(Z))
    untied_gradient = reshape(back(G), length(θ), 3)
    @test back₂(G) ≈ vec(sum(untied_gradient; dims = 2))
    @test iszero(untied_gradient[:, end])
end

@testset "adjacent causal windows and fixed normalization" begin
    c = Config(V = 2, D = 4, H = 1, L = 1, T = 3, F = 5)
    θ = init(c; rng = Xoshiro(19)) + 0.05randn(nparams(c))
    x = [2 1; 1 2; 2 1; 1 1; 2 2]
    y = 3 .- x
    opts = (; K = 2, injection = :after, attention_span = :adjacent,
              position_code = false, stabilize = true)
    trace = NamedTuple[]
    Zs, _ = loopgpt(θ, x, c; readouts = :all, trace, opts...)
    @test trace[1].X ≈ carve(θ, c).E[:, vec(x)]
    for frame in trace[2:end]
        A = frame.A
        @test all(A[t, s, h, b] == 0 for b in 1:2, h in 1:c.H, s in 1:5, t in 1:5
                  if s != t && s != t - 1)
        @test dropdims(sum(A[1:1, :, :, :]; dims = (1, 2)); dims = (1, 2)) +
              frame.boundary_attention ≈ ones(c.H, 2)
        @test all(sum(A[2:end, :, :, :]; dims = 2) .≈ 1)
        @test maximum(abs, sum(frame.X; dims = 1)) < 1e-12
    end
    changed = copy(x)
    changed[1, 1] = 3 - changed[1, 1]
    Z_changed, _ = loopgpt(θ, changed, c; readouts = :all, opts...)
    for k in 1:2
        @test Zs[k][:, k+2:end] == Z_changed[k][:, k+2:end]
    end
    objective(θ) = looploss(θ, x, [x, y], c; supervision = :deep, opts...)
    _, pull = objective(θ)
    err = gradcheck(θ -> first(objective(θ)), θ, pull())
    @info "Adjacent finite differences" loss_error = err
    @test err < 1e-6
end
