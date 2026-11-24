# Exactness of every lift, to rounding, on random parameters and random inputs.
#     julia --project=. -t 1 grow/lifts/test_lifts.jl
include(joinpath(@__DIR__, "Lifts.jl"))
using .Lifts, PicoGPT, Random, LinearAlgebra, Printf, Test

rng = Xoshiro(2026)
randθ(c) = init(c; rng) .+ 0.3 .* randn(rng, nparams(c))     # gains, offsets and biases all nontrivial
err(Z′, Z) = maximum(abs, Z′ .- Z)
report(name, e) = @printf("%-58s max |Δlogit| = %.2e\n", name, e)
const tol = 1e-11

c = Config(V = 7, D = 4, H = 2, L = 2, T = 8, F = 16)
net = Net(c)
θ = randθ(c)
x = rand(rng, 1:c.V, 8, 6)
Z, _ = logits(θ, x, net)

@testset "lifts" begin
    @testset "the model copy is PicoGPT; its pullback is right" begin
        @test logits(θ, x, net)[1] == gpt(θ, x, c)[1]
        y = rand(rng, 1:c.V, 8, 6)
        θd, netd = lift_width(θ, net; asym = 1.0, rng)                 # a net with U ≠ I
        e1 = gradcheck(t -> nll(t, x, y, netd)[1], θd, nll(θd, x, y, netd)[2]())
        Z★ = randn(rng, c.V, 48)
        e2 = gradcheck(t -> mse(t, x, Z★, netd)[1], θd, mse(θd, x, Z★, netd)[2]())
        @printf("finite differences: nll %.1e, mse %.1e (relative)\n", e1, e2)
        @test e1 < 1e-6 && e2 < 1e-6
    end

    @testset "width, coordinate duplication" begin
        for k in (2, 3), heads in (:same, :more, :fresh), asym in (0.0, 1.0)
            θ′, net′ = lift_width(θ, net; k, heads, asym, rng)
            e = err(logits(θ′, x, net′)[1], Z)
            report("width ×$k, heads $heads, asym $asym  (p $(nparams(c)) → $(nparams(net′.c)))", e)
            @test e < tol
        end
        # what each piece requires: break one thing at a time (k = 2, heads :same)
        θ′, net′ = lift_width(θ, net; rng)
        p′ = PicoGPT.carve(copy(θ′), net′.c)
        std = err(logits(θ′, x, Net(net′.c))[1], Z)                    # PicoGPT's own positions(2D, T)
        report("  … with PicoGPT's positions(2D, T) instead of U·P", std);  @test std > 1e-3
        θb = copy(θ′); pb = PicoGPT.carve(θb, net′.c)
        foreach(b -> (b.Wq .*= 2^(1/4); b.bq .*= 2^(1/4); b.Wk .*= 2^(1/4); b.bk .*= 2^(1/4)), pb.blocks)
        nos = err(logits(θb, x, net′)[1], Z)
        report("  … without the k^(−1/2) in the scores", nos);  @test nos > 1e-3
        # zero padding instead of duplication: the stream is (x, 0), and LayerNorm does not commute
        U0 = kron(Matrix{Float64}(I, c.D, c.D), [1.0; 0.0;;])
        θz = copy(θ′); pz = PicoGPT.carve(θz, net′.c)
        pz.E .= U0 * PicoGPT.carve(θ, c).E
        pad = err(logits(θz, x, Net(net′.c, U0))[1], Z)
        report("  … zero-padding the embedding instead of duplicating", pad);  @test pad > 1e-3
        # positions(2D, T) contains positions(D, T): rows 4i−3, 4i−2 are rows 2i−1, 2i
        @test positions(8, 16)[[1, 2, 5, 6], :] ≈ positions(4, 16) atol = 1e-15
    end

    @testset "heads at fixed D: exact only for low-rank score forms" begin
        θs, nets, tail = split_heads(θ, net; m = 2)
        e = err(logits(θs, x, nets)[1], Z)
        report("heads 2 → 4 at D = 4, generic parent (not exact)", e)
        @printf("   discarded fraction of each score form: %s\n", join(round.(tail, digits = 3), " "))
        @test e > 1e-3
        θr = copy(θ); pr = PicoGPT.carve(θr, c)                        # a parent whose forms have rank 1
        foreach(b -> (b.Wq[[2, 4], :] .= 0; b.bq[[2, 4]] .= 0), pr.blocks)
        θs, nets, tail = split_heads(θr, net; m = 2)
        e = err(logits(θs, x, nets)[1], logits(θr, x, net)[1])
        report("heads 2 → 4 at D = 4, rank-1 score forms", e);  @test e < tol
        # bk is a gauge direction: softmax ignores a constant added to a row of scores
        y = rand(rng, 1:c.V, 8, 6)
        ḡ = PicoGPT.carve(nll(θ, x, y, net)[2](), c)
        @test maximum(b -> maximum(abs, b.bk), ḡ.blocks) < 1e-14
    end

    @testset "MLP width" begin
        for mode in (:fresh, :dup, :dead)
            θ′, net′ = lift_mlp(θ, net; F′ = 32, mode, rng)
            e = err(logits(θ′, x, net′)[1], Z)
            report("MLP 16 → 32, $mode", e);  @test e < tol
        end
        θ′, net′ = lift_mlp(θ, net; F′ = 32, mode = :dead, rng)        # exact but stillborn
        y = rand(rng, 1:c.V, 8, 6)
        ḡ = PicoGPT.carve(nll(θ′, x, y, net′)[2](), net′.c)
        @test all(b -> iszero(b.W1[17:end, :]) && iszero(b.W2[:, 17:end]) && iszero(b.b1[17:end]), ḡ.blocks)
    end

    @testset "depth" begin
        for at in 1:3
            θ′, net′ = lift_depth(θ, net; at, rng)
            e = err(logits(θ′, x, net′)[1], Z)
            report("depth 2 → 3, new block at $at", e);  @test e < tol
        end
    end

    @testset "context" begin
        θ′, net′ = lift_context(θ, net; T′ = 16)
        e = err(logits(θ′, x, net′)[1], Z)
        report("context 8 → 16, sequences of length 8", e);  @test e == 0
        xl = rand(rng, 1:c.V, 16, 6)
        e = err(logits(θ′, xl, net′)[1][:, vec([t ≤ 8 for t in 1:16, b in 1:6])], logits(θ, xl[1:8, :], net)[1])
        report("context 8 → 16, first 8 positions of length-16 sequences", e);  @test e < tol
    end

    @testset "vocabulary" begin
        q = [0.5, 0.3, 0.2]
        θ′, net′, π = lift_vocab(θ, net; v = 3, m = 3, q)
        x′ = rand(rng, 1:net′.c.V, 8, 6)
        Z′ = logits(θ′, x′, net′)[1]
        Zp = logits(θ, π[x′], net)[1]
        lq = zeros(net′.c.V); lq[[3, 8, 9]] .= log.(q)
        e = err(Z′, Zp[π, :] .+ lq)
        report("vocabulary 7 → 9, token 3 split in three: z′ − (z∘π + log q)", e);  @test e < tol
        p′, p = softmax(Z′; dims = 1), softmax(Zp; dims = 1)
        coarse = [sum(p′[π .== u, n]) for u in 1:c.V, n in 1:size(p, 2)]
        e = err(coarse, p)
        report("   coarse-grained probabilities Σ_b p′ − p", e);  @test e < 1e-13
        y′ = rand(rng, 1:net′.c.V, 8, 6)                               # the loss splits
        new = -sum(lq[y′]) / length(y′)
        e = abs(nll(θ′, x′, y′, net′)[1] - nll(θ, π[x′], π[y′], net)[1] - new)
        report("   loss′ − (parent's loss + mean −log q)", e);  @test e < 1e-13
    end

    @testset "lifts compose" begin
        θ′, net′ = lift_width(θ, net; asym = 1.0, rng)
        θ′, net′ = lift_mlp(θ′, net′; F′ = 32, rng)
        θ′, net′ = lift_depth(θ′, net′; at = 2, rng)
        θ′, net′ = lift_context(θ′, net′; T′ = 16)
        θ′, net′, π = lift_vocab(θ′, net′; v = 1, m = 2)
        e = err(logits(θ′, x, net′)[1][1:c.V, :], Z .+ [log(0.5); zeros(c.V - 1)])
        report("all five composed (p $(nparams(c)) → $(nparams(net′.c)))", e);  @test e < tol
    end
end
