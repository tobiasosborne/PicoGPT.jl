using PicoGPT, Test, LinearAlgebra, Random

Random.seed!(1)

# The specimen of the book: small enough to see every number.
const corpus = "to be or not to be"
const abc = alphabet(corpus)
const specimen = Config(V = length(abc), D = 8, H = 2, L = 1, T = 8, F = 32)

@testset "PicoGPT" begin

@testset "text" begin
    @test String(abc) == " benort"
    @test decode(encode(corpus, abc), abc) == corpus
end

@testset "every pullback agrees with finite differences" begin
    D, H, T, B, V, F = 6, 2, 4, 3, 5, 10
    N = T * B
    ok(errors) = all(<(1e-6), errors)
    @test ok(checkpullback(embed, rand(1:V, T, B), randn(D, V); wrt = (2,)))
    @test ok(checkpullback(affine, randn(D, N), randn(F, D), randn(F); wrt = (1, 2, 3)))
    @test ok(checkpullback(relu, randn(F, N); wrt = (1,)))
    @test ok(checkpullback(layernorm, randn(D, N), randn(D), randn(D); wrt = (1, 2, 3)))
    @test ok(checkpullback(attention, randn(D, N), randn(D, N), H, T; wrt = (1, 2)))
    @test ok(checkpullback(mix, rand(T, T, H, B), randn(D, N); wrt = (1, 2)))
    @test ok(checkpullback(crossentropy, randn(V, N), rand(1:V, N); wrt = (1,)))
    c = Config(; V, D, H, L = 1, T, F)
    p = carve(randn(nparams(c)), c).blocks[1]
    @test ok(checkpullback(block, randn(D, N), p, H, T; wrt = (1,)))
end

@testset "the model's gradient ∇θ agrees with finite differences" begin
    for c in (specimen, Config(V = 5, D = 12, H = 3, L = 2, T = 6, F = 20))
        θ = init(c; rng = Xoshiro(2)) + 0.1randn(nparams(c))      # so that γ ≠ 1, b ≠ 0
        x, y = rand(1:c.V, c.T, 3), rand(1:c.V, c.T, 3)
        ℓ, back = loss(θ, x, y, c)
        θ̄ = back()
        @test gradcheck(θ -> first(loss(θ, x, y, c)), θ, θ̄) < 1e-6
        ℓ₂, θ̄₂ = ∇loss(θ, x, y, c)                                  # the parallel version
        @test ℓ₂ ≈ ℓ rtol = 1e-14
        @test θ̄₂ ≈ θ̄ rtol = 1e-14
    end
end

@testset "θ is one vector; the parameters are views onto it" begin
    c = Config(V = 5, D = 12, H = 3, L = 2, T = 6, F = 20)
    θ = init(c)
    @test length(θ) == nparams(c)
    @test flatten(carve(θ, c)) == θ
    p = carve(θ, c)
    p.blocks[2].Wq[3, 4] = 42
    @test 42 in θ
    @test nparams(specimen) == 1007
    # the fixed order of BRIEF §3
    @test keys(p) == (:E, :blocks, :γf, :βf, :Wout, :bout)
    @test keys(p.blocks[1]) == (:γ1, :β1, :Wq, :bq, :Wk, :bk, :Wv, :bv, :Wo, :bo, :γ2, :β2, :W1, :b1, :W2, :b2)
    @test size.(leaves(p.blocks[1])) == [(12,), (12,), (12, 12), (12,), (12, 12), (12,), (12, 12), (12,),
                                         (12, 12), (12,), (12,), (12,), (20, 12), (20,), (12, 20), (12,)]
    @test parentindices(p.blocks[1].γ1)[1] == 12 * 5 .+ (1:12)       # right after E
    λ = decay(c, 0.1)
    @test all(all(x .== (ndims(x) == 2 ? 0.1 : 0)) for x in leaves(carve(λ, c)))
end

@testset "causality: nothing flows from the future, or between sequences" begin
    c = Config(V = 11, D = 16, H = 4, L = 2, T = 10)
    θ = init(c; rng = Xoshiro(3))
    x = rand(1:c.V, c.T, 2)
    position = [t for t in 1:c.T, b in 1:2]
    Z = first(gpt(θ, x, c))
    for t₀ in (1, 4, 10)
        x′ = copy(x)
        x′[t₀, :] .= mod1.(x[t₀, :] .+ 1, c.V)                     # change token t₀
        Z′ = first(gpt(θ, x′, c))
        @test Z′[:, vec(position .< t₀)] == Z[:, vec(position .< t₀)]
        @test all(Z′[:, n] != Z[:, n] for n in findall(vec(position .≥ t₀)))
    end
    x′ = copy(x)
    x′[:, 2] = rand(1:c.V, c.T)                                     # change all of sequence 2
    @test first(gpt(θ, x′, c))[:, 1:c.T] == Z[:, 1:c.T]
end

@testset "attention matrices are row-stochastic and lower-triangular" begin
    H, T, B = 2, 6, 3
    A, _ = attention(randn(8, T * B), randn(8, T * B), H, T)
    @test size(A) == (T, T, H, B)
    @test all(≥(0), A)
    @test all(sum(A, dims = 2) .≈ 1)
    @test all(A[i, j, h, b] == 0 for i in 1:T, j in 1:T, h in 1:H, b in 1:B if j > i)
    @test all(A[1, 1, :, :] .== 1)                                 # the first token reads itself
end

@testset "softmax, logsumexp, cross-entropy" begin
    z = randn(7)
    @test sum(softmax(z)) ≈ 1
    @test softmax(z .+ 1000) ≈ softmax(z)                         # only differences matter
    @test logsumexp(z .+ 5) ≈ logsumexp(z) + 5
    @test logsumexp([1000.0, 1000.0]) ≈ 1000 + log(2)              # no overflow
    @test softmax(z ./ 1e-3) ≈ (1:7 .== argmax(z))                 # τ → 0: the ground state
    @test softmax(z ./ 1e6) ≈ fill(1 / 7, 7) rtol = 1e-4           # τ → ∞: uniform
    Z, y = randn(5, 9), rand(1:5, 9)
    @test first(crossentropy(Z, y)) ≈ -sum(log(softmax(Z[:, n])[y[n]]) for n in 1:9) / 9
end

@testset "at the start, the loss is ≈ ln V" begin
    c = Config(V = 30, D = 64, H = 4, L = 2, T = 16)
    x, y = rand(1:c.V, c.T, 8), rand(1:c.V, c.T, 8)
    @test first(loss(init(c; rng = Xoshiro(4)), x, y, c)) ≈ log(c.V) atol = 0.05
end

@testset "Float32 in, Float32 out" begin
    θ = init(specimen, Float32)
    x, y = windows(encode(corpus, abc), specimen.T)
    ℓ, θ̄ = ∇loss(θ, x, y, specimen)
    @test ℓ isa Float32 && eltype(θ̄) == Float32
    @test ℓ ≈ first(loss(Float64.(θ), x, y, specimen)) rtol = 1e-5
end

@testset "the specimen memorises its corpus" begin
    x, y = windows(encode(corpus, abc), specimen.T)
    θ = init(specimen; rng = Xoshiro(1))
    opt = AdamW(θ; λ = decay(specimen, 0.01))
    η(t) = warmup_cosine(t; η = 2e-2, warmup = 20, steps = 200, ηmin = 2e-3)
    losses = [step!(θ, opt, x, y, specimen; η = η(t), clip = 1) for t in 1:200]
    @test losses[1] ≈ log(specimen.V) atol = 0.1
    @test first(loss(θ, x, y, specimen)) < 0.06                    # the entropy floor is 0.0585
    @test decode(generate(θ, specimen, encode("to be or", abc), 10; τ = 0), abc) == corpus
end

end
