# Checks of truncate.jl:    julia --project -t 3 grow/truncate/test.jl
#
#   1  with no cut, tgpt is gpt, and its gradient is gpt's            (to rounding)
#   2  the gradient of the cut loss, by central differences           (orthogonal and oblique cuts)
#   3  the stream adjoints adj[k], by central differences
#   4  the cut is idempotent: cutting the cut model again changes nothing
#   5  the cut model written in r coordinates (smallgpt) is the same function   (to rounding)
#   6  the direction 1 is invisible: M 1 = 0;  exact probes agree with Gaussian probes
#   7  a full-rank cut changes nothing; the intertwiner defect of a full-rank cut is zero

include("truncate.jl")
using Test

corpus = "to be or not to be"
abc = alphabet(corpus); data = encode(corpus, abc)

@testset "truncate" begin
    for (D, L) in ((8, 1), (4, 2))
        c = Config(V = length(abc), D = D, H = 2, L = L, T = 8, F = 4D)
        x, y = windows(data, c.T)
        rng = Xoshiro(7)
        θ = init(c; rng) .+ 0.3 .* randn(rng, nparams(c))      # off the special initial point
        K = 2L + 1

        # 1
        Z, back, = tgpt(θ, x, c)
        Zref, backref = gpt(θ, x, c)
        Z̄ = randn(rng, size(Z))
        @test Z == Zref
        @test first(back(Z̄)) == backref(Z̄)

        st = streamstats(θ, x, c)
        for rule in (:pca, :balanced, :random), r in (2, D - 1)
            q = choose(rule, r, st; rng)
            @test q.S * q.T ≈ I                                 # a projector
            # 2
            ℓ, backℓ = tloss(θ, x, y, c, q)
            θ̄, = backℓ()
            @test gradcheck(t -> first(tloss(t, x, y, c, q)), θ, θ̄) < 1e-6
            @test first(∇tloss(θ, x, y, c, q)) ≈ ℓ
            @test last(∇tloss(θ, x, y, c, q)) ≈ θ̄
            @test cutloss(θ, x, y, c, q) ≈ ℓ
            # 4
            Zq, _, sites, = tgpt(θ, x, c, q)
            @test all(first(site(sites[k], q, k)) ≈ sites[k] for k in 1:K)
            # 5
            Zs, zs = smallgpt(θ, x, c, q)
            @test maximum(abs, Zs - Zq) < 1e-10
            @test all(maximum(abs, zs[k] - q.S * (sites[k] .- q.μ[:, k])) < 1e-10 for k in 1:K)
        end

        # 3: adj[k] is the gradient of ⟨Z̄, logits⟩ with respect to a perturbation of site k
        q = choose(:balanced, D - 2, st)
        for cut in (nothing, q)
            _, backc, sites, = tgpt(θ, x, c, cut)
            _, adj = backc(Z̄)
            p = carve(θ, c)
            for k in 1:K
                function from(Δ)                                # rerun from site k, perturbed
                    X = sites[k] + Δ
                    for j in k+1:K
                        b = p.blocks[j ÷ 2]
                        X = first(site(X + (iseven(j) ? first(attend(X, b, c.H, c.T)) : first(think(X, b))), cut, j))
                    end
                    dot(Z̄, first(readout(X, p)))
                end
                @test gradcheck(from, zeros(size(sites[k])), adj[k]) < 1e-6
            end
        end

        # 6
        for metric in (:logit, :fisher)
            s = streamstats(θ, x, c; metric)
            @test all(norm(M * ones(D)) < 1e-10 * max(norm(M), 1) for M in s.M)
            g = streamstats(θ, x, c; metric, probes = 4000)
            @test all(norm(g.M[k] - s.M[k]) < 0.1 * norm(s.M[k]) for k in 1:K)
            @test g.C == s.C && g.μ == s.μ
        end
        # the exact Gram against explicit Jacobians, token by token (logit metric, last site)
        _, backc, sites, = tgpt(θ, x, c)
        N = length(x); Mk = zeros(D, D)
        for n in 1:N, a in 1:c.V
            Z̄1 = zeros(c.V, N); Z̄1[a, n] = 1; Z̄1[:, n] .-= 1 / c.V
            _, adj = backc(Z̄1)
            Mk += adj[1] * adj[1]' / N
        end
        @test Mk ≈ st.M[1]

        # 7
        full = choose(:balanced, D, st)
        @test maximum(abs, first(tgpt(θ, x, c, full)) - Z) < 1e-8
        d = defects(θ, θ, x, c, full)
        @test maximum(d.loc) < 1e-8 && maximum(d.drift) < 1e-8 && abs(d.kdrift) < 1e-10
        # and the direction 1 can be cut for free: rank D − 1 by :jac loses nothing
        @test maximum(abs, first(tgpt(θ, x, c, choose(:jac, D - 1, st))) - Z) < 1e-8
    end
end
