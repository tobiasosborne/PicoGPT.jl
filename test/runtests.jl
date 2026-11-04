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

@testset "shape boundaries and zero blocks" begin
    @test nparams(Config(V=65)) == 810049
    for dims in ((3,3,1,1,4,12), (3,4,0,1,4,16), (0,4,1,1,4,16),
                 (3,4,3,1,4,16), (3,4,1,-1,4,16), (3,4,1,1,0,16), (3,4,1,1,4,0))
        @test_throws ArgumentError Config(dims...)
        V,D,H,L,T,F = dims
        @test_throws ArgumentError Config(;V,D,H,L,T,F)
    end
    @test_throws ArgumentError positions(3,4)
    @test_throws ArgumentError positions(0,4)
    @test_throws ArgumentError positions(4,-1)
    @test size(positions(4,0)) == (4,0)
    @test positions(4,3)[:,1] == [0,1,0,1]
    for R in (Float32,Float64), L in (0,1), H in (1,2), B in (1,3), T in (1,3)
        c = Config(V=3,D=4,H=H,L=L,T=5,F=7)
        θ = init(c,R;rng=Xoshiro(19))
        @test flatten(@inferred carve(θ,c)) == θ
        x = rand(Xoshiro(10),1:c.V,T,B)
        trace = []
        Z,bZ = @inferred gpt(θ,x,c;trace)
        @test size(Z) == (c.V,T*B) && eltype(Z) == R
        @test length(trace) == L+2
        @test eltype(bZ(ones(R,size(Z)))) == R
        l,b = @inferred loss(θ,x,x,c)
        lp,gp = ∇loss(θ,x,x,c)
        tol = R == Float32 ? 2e-6 : 1e-13
        @test l isa R && lp isa R && eltype(b()) == R && eltype(gp) == R
        @test lp ≈ l rtol=tol
        @test gp ≈ b() rtol=tol
        g = b(); saved = copy(g)
        @test b(R(2)) ≈ R(2)*saved rtol=tol
        @test g == saved && b() == saved # later calls preserve earlier cotangents and caches
        @test first(gpt(θ,x[:,1:1],c)) ≈ Z[:,1:T] rtol=tol
        @test size(first(gpt(θ,zeros(Int,T,0),c))) == (c.V,0)
        @test_throws ArgumentError loss(θ,zeros(Int,T,0),zeros(Int,T,0),c)
        @test_throws ArgumentError ∇loss(θ,zeros(Int,T,0),zeros(Int,T,0),c)
        @test_throws ArgumentError gpt(θ,zeros(Int,0,1),c)
        @test_throws ArgumentError gpt(θ,ones(Int,c.T+1,1),c)
        @test_throws DimensionMismatch loss(θ,x,x[:,1:0],c)
    end
end

@testset "stable cross-entropy and independent pullback identities" begin
    for R in (Float32,Float64)
        shift = R == Float32 ? R(1e8) : R(1e16)
        Z = fill(shift,3,2)
        l,b = crossentropy(Z,[1,2])
        @test l isa R
        @test l ≈ log(R(3))
        expected = fill(R(1)/6,3,2)
        expected[1,1] -= R(1)/2; expected[2,2] -= R(1)/2
        @test b() ≈ expected
        @test eltype(b()) == R
        @test maximum(abs,sum(b();dims=1)) ≤ 2eps(R)
        # Compare extreme finite logits to high precision, independently of package helpers.
        Z = R[1000 -1000; -1000 1000; 999 -999]
        y = [2,3]
        reference = sum(log(sum(exp.(BigFloat.(Z[:,n]))))-BigFloat(Z[y[n],n]) for n in 1:2)/2
        @test first(crossentropy(Z,y)) ≈ R(reference)
        @test b(R(0)) == zeros(R,3,2)
        Q = randn(Xoshiro(12),R,4,1)
        A,back = attention(Q,Q,1,1)
        @test A == ones(R,1,1,1,1)
        @test all(iszero,first(back(ones(R,size(A)))))
        @test all(iszero,last(back(ones(R,size(A)))))
        Y,br = relu(R[-1,0,1])
        @test Y == R[0,0,1] && br(ones(R,3)) == R[0,0,1]
        x = [1 1; 1 2; 2 1]
        E = zeros(R,4,3)
        _,be = embed(x,E)
        @test be(ones(R,4,6)) == repeat(R[4 2 0],4,1)
    end
    @test_throws ArgumentError crossentropy(zeros(3,0),Int[])
    @test_throws DimensionMismatch crossentropy(zeros(3,2),[1])
    @test gradcheck(x -> 0.0,zeros(3),zeros(3)) == 0
    @test gradcheck(x -> 0.0,zeros(3),ones(3)) == 1
    # Constant columns resolve the epsilon scale; x̄ = Π(γ⊙ȳ)/√ε.
    X = fill(2.0,4,2); γ = [1.0,2,3,4]; β = [4.0,3,2,1]
    Y,b = layernorm(X,γ,β)
    G = randn(Xoshiro(2),4,2)
    xb,gb,bb = b(G)
    expected = γ .* G
    expected = (expected .- sum(expected;dims=1)/4)/sqrt(1e-5)
    @test Y == repeat(β,1,2)
    @test xb ≈ expected
    @test gb == zeros(4) && bb ≈ vec(sum(G;dims=2))
    @test gradcheck(x -> dot(G,first(layernorm(x,γ,β))),X,xb;h=1e-8) < 1e-7
    # Verify the nonzero radial response for epsilon > 0.
    X = reshape([-2.0,-1,1,2],4,1)
    _,b = layernorm(X,ones(4),zeros(4))
    σ = sqrt(sum(abs2,X)/4+1e-5)
    @test first(b(X)) ≈ 1e-5*X/σ^3 rtol=1e-9
    # An arbitrary logit covector checks the whole model independently of the loss.
    c = Config(V=3,D=4,H=2,L=2,T=4,F=7)
    θ = init(c;rng=Xoshiro(22)) + 0.1randn(Xoshiro(23),nparams(c))
    x = [1 2; 1 3; 2 1]
    @test all(<(1e-6),checkpullback((θ,x) -> gpt(θ,x,c),θ,x;wrt=(1,)))
    Z = first(gpt(θ,x,c)); θ′ = copy(θ)
    foreach(p -> p.bk .+= [1.0,2,3,4],carve(θ′,c).blocks)
    @test first(gpt(θ′,x,c)) ≈ Z rtol=1e-13
    g = last(loss(θ,x,x,c))()
    @test all(norm(p.bk)<1e-14 for p in carve(g,c).blocks)
end

@testset "AdamW, clipping, and schedule endpoints" begin
    for R in (Float32,Float64)
        θ = R[1,-2,3]; old=copy(θ); g=R[3,4,0]
        @test clip!(copy(g),0) == zeros(R,3)
        @test clip!(copy(g),2) ≈ R(2)/5*g
        @test clip!(zeros(R,3),0) == zeros(R,3)
        @test_throws ArgumentError clip!(g,-1)
        clipped = clip!(copy(g),1)
        o = AdamW(θ;λ=R[0.1,0,0.2])
        @test o.β1 isa R && o.β2 isa R && o.ε isa R && eltype(o.λ) == R
        @test eltype(decay(specimen,R(0.1))) == R
        # Explicit scalar reference over varying gradients catches bias-correction indexing,
        # epsilon placement, and decay applied after the adaptive update.
        mr=zeros(R,3); vr=zeros(R,3); ref=copy(θ)
        for t in 1:5
            gt = t==1 ? clipped : R[t,-2t,0]
            η=R(0.02)
            for i in eachindex(ref)
                mr[i]=o.β1*mr[i]+(1-o.β1)*gt[i]
                vr[i]=o.β2*vr[i]+(1-o.β2)*gt[i]^2
                ref[i]=(1-η*o.λ[i])*ref[i]-η*(mr[i]/(1-o.β1^t))/(sqrt(vr[i]/(1-o.β2^t))+o.ε)
            end
            update!(o,θ,gt,η)
            @test θ ≈ ref rtol=8eps(R)
            @test o.m ≈ mr && o.v ≈ vr && o.t == t
        end
        @test update!(Descent(),copy(old),g,0.02) ≈ old-R(0.02)*g
        η(t;warmup=2) = warmup_cosine(t;η=R(0.2),ηmin=R(0.02),warmup,steps=6)
        @test η(0) === zero(R)
        @test η(1) ≈ R(0.1) && η(2) ≈ R(0.2) && η(6) ≈ R(0.02)
        @test η(4) ≈ R(0.11)
        @test η(0;warmup=0) === R(0.2)
        @test η(6;warmup=0) ≈ R(0.02)
        @test all(η(t) isa R for t in 0:6)
        @test all(diff([η(t) for t in 2:6]) .≤ 0)
    end
    @test_throws ArgumentError warmup_cosine(1;η=0.1,warmup=2,steps=2)
    @test_throws ArgumentError warmup_cosine(7;η=0.1,warmup=2,steps=6)
    @test_throws ArgumentError AdamW(zeros(3);β1=1)
    c=Config(V=3,D=4,H=1,L=0,T=3)
    θ=init(c;rng=Xoshiro(24)); x=ones(Int,3,1)
    l,b=loss(θ,x,x,c); g=clip!(b(),0.01)
    o=AdamW(θ); θ′=copy(θ)
    @test step!(θ′,o,x,x,c;η=0.1,clip=0.01) == l
    @test o.m ≈ (1-o.β1)*g && o.v ≈ (1-o.β2)*g.^2
end

@testset "thread scopes overlap, nest, and unwind safely" begin
    original = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(3)
        entered=Channel{Nothing}(2); leave_a=Channel{Nothing}(1); leave_b=Channel{Nothing}(1)
        a=Threads.@spawn PicoGPT.without_blas_threads() do
            put!(entered,nothing); take!(leave_a)
        end
        take!(entered)
        b=Threads.@spawn PicoGPT.without_blas_threads() do
            put!(entered,nothing); take!(leave_b)
        end
        take!(entered)
        @test BLAS.get_num_threads() == 1
        put!(leave_a,nothing); wait(a)
        @test BLAS.get_num_threads() == 1
        put!(leave_b,nothing); wait(b)
        @test BLAS.get_num_threads() == 3
        @test_throws ErrorException PicoGPT.without_blas_threads() do
            PicoGPT.without_blas_threads() do
                error("nested failure")
            end
        end
        @test BLAS.get_num_threads() == 3
        x,y=windows(encode(corpus,abc),specimen.T)
        θ=init(specimen;rng=Xoshiro(7))
        expected=∇loss(θ,x,y,specimen)
        tasks=@sync [Threads.@spawn ∇loss(θ,x,y,specimen) for _ in 1:4]
        @test all(fetch(task)==expected for task in tasks)
        @test BLAS.get_num_threads() == 3
        bad=copy(x); bad[1,1]=0
        @test_throws Exception ∇loss(θ,bad,y,specimen)
        @test BLAS.get_num_threads() == 3
        @test_throws Exception PicoGPT.foreach_head(zeros(2,2,1,2)) do a
            error("head failure")
        end
        @test BLAS.get_num_threads() == 3
    finally
        BLAS.set_num_threads(original)
    end
end

# Exercise the actual script IO code; this never downloads data or writes artifacts.
module ScriptTools
include(joinpath(@__DIR__,"..","scripts","common.jl"))
end
@testset "checkpoint round trips and script evaluation" begin
    c=Config(V=3,D=4,H=1,L=1,T=3)
    abc=collect("aé𝄞"); θ=init(c,Float32;rng=Xoshiro(25))
    o=AdamW(θ;λ=decay(c,0.01))
    update!(o,θ,ones(Float32,length(θ)),0.01)
    mktempdir() do dir
        path=joinpath(dir,"state.bin")
        for opt in (nothing,o)
            ScriptTools.save_checkpoint(path;c,abc,step=1,θ,opt)
            ck=ScriptTools.load_checkpoint(path)
            @test ck.c == c && ck.abc == abc && ck.step == 1 && ck.θ == θ
            @test eltype(ck.θ) == Float32
            if opt===nothing
                @test isempty(ck.adam) && ck.t==0
            else
                @test ck.adam==[o.m,o.v] && ck.t==o.t
                restored=AdamW(ck.θ;λ=decay(c,0.01))
                restored.m .= ck.adam[1]; restored.v .= ck.adam[2]; restored.t=ck.t
                g=randn(Xoshiro(26),Float32,length(θ)); θr=copy(ck.θ); θs=copy(θ)
                update!(restored,θr,g,0.002)
                update!(deepcopy(o),θs,g,0.002)
                @test θr==θs
            end
        end
        bytes=read(path)
        @test bytes[9:16] == [0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00] # little-endian V
        write(path,bytes[1:end-1])
        @test_throws EOFError ScriptTools.load_checkpoint(path)
        write(path,"not a checkpoint")
        @test_throws ErrorException ScriptTools.load_checkpoint(path)
        @test_throws ArgumentError ScriptTools.save_checkpoint(path;c,abc,step=1,θ=Float64.(θ))
    end
    data=encode("aé𝄞aé𝄞a",abc)
    x,y=windows(data,3,[1,2,3])
    @test x == [data[i+t-1] for t in 1:3,i in 1:3]
    @test y == [data[i+t] for t in 1:3,i in 1:3]
    @test ScriptTools.meanloss(θ,data,[1,2,3],c) ≈ first(loss(θ,x,y,c))
    @test_throws ArgumentError ScriptTools.meanloss(θ,data,Int[],c)
    @test ScriptTools.json((s="a\n\t\"\\",x=[0.1,2.0])) == "{\"s\":\"a\\n\\t\\\"\\\\\",\"x\":[0.1,2.0]}"
    @test_throws ErrorException ScriptTools.json(Inf)
end

@testset "generation boundaries" begin
    c=Config(V=3,D=4,H=1,L=0,T=3); θ=init(c;rng=Xoshiro(27))
    @test generate(θ,c,[1],0) == [1]
    @test_throws ArgumentError generate(θ,c,Int[],1)
    @test_throws ArgumentError generate(θ,c,[1],1;τ=-1)
    @test_throws ArgumentError generate(θ,c,[1],-1)
    @test generate(θ,c,[1,2,3,1],2;τ=0)[1:4] == [1,2,3,1]
    @test generate(θ,c,[1],5;rng=Xoshiro(1)) == generate(θ,c,[1],5;rng=Xoshiro(1))
    for p in (Float64[],[0.0,0.0],[-1.0,2.0],[Inf,1.0])
        @test_throws ArgumentError draw(p)
    end
    @test draw([0.0,1.0,0.0]) == 2
end
