include("shakespeare_common.jl")
using Test

function check_shakespeare()
    BLAS.set_num_threads(1)
    rng = Xoshiro(303)
    c = Config(V = 5, D = 4, H = 2, T = 4, L = 1, F = 7)
    θ = init(c, Float64; rng) + 0.05randn(rng, nparams(c))
    x = [1 4; 2 2; 1 5; 3 1]
    y = [2 2; 1 5; 3 1; 4 3]
    K = 3

    @testset "Shakespeare tied forward and pullback" begin
        for name in (:looped, :stock)
            opts = model_options(MODELS[name], c)
            rollout(w, x = x; trace = nothing) = windowgpt(w, x, c; K, readouts = :all, trace, opts...)
            trace = NamedTuple[]
            Zs, back = rollout(θ; trace)
            cotangents = [randn(rng, size(Z)) for Z in Zs]
            f(w) = sum(dot(G, Z) for (G, Z) in zip(cotangents, first(rollout(w))))
            logit_error = gradcheck(f, θ, back(cotangents))
            @test logit_error < 1e-6

            objective(w) = windowloss(w, x, y, c; K, supervision = :deep, opts...)
            ℓ, pull = objective(θ)
            @test ℓ ≈ sum(first(crossentropy(Z, y)) for Z in Zs) / K
            loss_error = gradcheck(w -> first(objective(w)), θ, pull())
            @test loss_error < 1e-6
            @info "Float64 finite differences" model = name logit_error loss_error

            @test all(frame.weights[t, s, h, b] == 0 for frame in trace[2:end]
                      for h in 1:c.H, b in axes(x, 2), s in axes(x, 1) for t in 1:s-1)
            changed = copy(x)
            changed[end, 1] = 5
            @test all(isapprox(Z[:, 1:3], Z′[:, 1:3]) for
                      (Z, Z′) in zip(Zs, first(rollout(θ, changed))))
        end

        Z, pull = windowgpt(θ, x, c; K = 1, model_options(MODELS.one, c)...)
        Z₁, pull₁ = gpt(θ, x, c)
        G = randn(rng, size(Z))
        @test Z == Z₁
        @test pull(G) ≈ pull₁(G)

        cK = Config(; c.V, c.D, c.H, c.T, c.F, L = K)
        p = carve(θ, c)
        untied = flatten(p.E, fill(only(p.blocks), K), p.γf, p.βf, p.Wout, p.bout)
        ZK, pullK = gpt(untied, x, cK)
        Z, pull = windowgpt(θ, x, c; K, model_options(MODELS.stock, c)...)
        q = carve(pullK(G), cK)
        @test Z == ZK
        @test pull(G) ≈ flatten(q.E, sum(flatten, q.blocks), q.γf, q.βf, q.Wout, q.bout)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && check_shakespeare()
