# From the repository root: julia -t1 --project grow/notes/blocking/check.jl
using PicoGPT, LinearAlgebra, Random, Printf, Statistics
include(joinpath(@__DIR__, "../../../jspace/src/JLens.jl"))
BLAS.set_num_threads(1)

diameter(V) = maximum(norm(a-b) for a in eachcol(V) for b in eachcol(V))

function block_terms(scores, V)
    m = length(scores)
    center = mean(scores)
    y = scores .- center
    w = vec(mean(V; dims=2))
    g = V .- w
    s1 = log(m) + center
    v1 = w + g*y/m
    s2 = s1 + mean(abs2, y)/2
    v2 = v1 + g*(y.^2)/(2m)
    s, v = logsumexp(scores), V*softmax(scores)
    width = maximum(y)-minimum(y)
    dv = diameter(V)
    tol = 1e-12
    @assert -tol ≤ s-s1 ≤ width^2/8 + tol
    @assert norm(v-w) ≤ dv*width/4 + tol
    @assert norm(v-v1) ≤ dv*width^2/8 + tol
    (; s, v, s1, v1, s2, v2, width, dv)
end

function inspect_head!(errors, Q, K, V, ref, blocksize)
    d, T = size(Q)
    for i in 1:T
        scores = K[:, 1:i]' * Q[:, i] / sqrt(d)
        blocks = [block_terms(scores[J], V[:, J]) for J in Iterators.partition(1:i, blocksize)]
        reads = map((:s, :s1, :s2), (:v, :v1, :v2)) do score, value
            hcat((getproperty(b,value) for b in blocks)...) *
                softmax([getproperty(b,score) for b in blocks])
        end
        exact, first, second = (norm(r-ref[:, i]) for r in reads)
        @assert exact < 1e-12
        maxwidth = maximum(b.width for b in blocks)
        @assert first ≤ diameter(V[:, 1:i])*maxwidth^2/32 +
            maximum(b.dv*b.width^2/8 for b in blocks) + 1e-12
        push!(errors, (; exact, first, second, maxwidth))
    end
end

function report(label, errors)
    @printf("%-18s %5d  %9.3g  %9.3g  %9.3g  %9.3g\n", label, length(errors),
        maximum(e.exact for e in errors), sqrt(mean(e.first^2 for e in errors)),
        sqrt(mean(e.second^2 for e in errors)), maximum(e.maxwidth for e in errors))
end

function random_heads()
    d, T, blocksize = 4, 12, 3
    for scale in (0.125, 0.25, 0.5, 1.0)
        rng = Xoshiro(1947)  # same heads at every scale
        errors = NamedTuple[]
        for _ in 1:64
            Q = randn(rng,d,T)
            centers = repeat(randn(rng,d,T÷blocksize); inner=(1,blocksize))
            K = centers + scale*randn(rng,d,T)
            V = randn(rng,d,T)
            A, _ = attention(Q,K,1,T)
            O, _ = mix(A,V)
            inspect_head!(errors,Q,K,V,O,blocksize)
        end
        report("random σ=$scale",errors)
    end
end

function specimen()
    data = JLens.specimen_data(joinpath(@__DIR__, "../../../artifacts/specimen.json"))
    c = data.c
    trace = []
    gpt(data.θ, data.x, c; trace)
    layer = trace[2]
    errors = NamedTuple[]
    d = c.D ÷ c.H
    for window in axes(data.x, 2), head in 1:c.H
        rows = (head-1)*d+1:head*d
        cols = (window-1)*c.T+1:window*c.T
        inspect_head!(errors, layer.Q[rows,cols], layer.K[rows,cols],
            layer.V[rows,cols], layer.O[rows,cols], 3)
    end
    report("trained step $(data.step)",errors)
end

function identities()
    rng = Xoshiro(19)
    K, V = randn(rng,4,9), randn(rng,4,9)
    x, shift = randn(rng,4), randn(rng,4)
    y = K'*x
    b = block_terms(y,V)
    children = [block_terms(y[J], V[:,J]) for J in (1:2,3:5,6:9)]
    scores = [c.s for c in children]
    @assert logsumexp(scores) ≈ b.s
    @assert hcat((c.v for c in children)...)*softmax(scores) ≈ b.v
    @assert logsumexp((K .+ shift)'*x) - b.s ≈ dot(x,shift)
    @assert logsumexp(fill(y[1],9)) ≈ log(9) + y[1]
    # Different values at identical keys with constant group mean.
    w = randn(rng,4)
    @assert hcat(w .+ V, w .- V)*softmax(vcat(y,y)) ≈ w
    μ = vec(mean(K; dims=2))
    covkeys(K,μ) = (K .- μ)*(K .- μ)'/size(K,2)
    pooled = zeros(4,4)
    for J in (1:2,3:5,6:9)
        μc = vec(mean(K[:,J]; dims=2))
        pooled .+= length(J)/9 * (covkeys(K[:,J],μc) + (μc-μ)*(μc-μ)')
    end
    @assert pooled ≈ covkeys(K,μ)
    # Directional finite differences at the expansion point.
    h = 2e-4
    score(t) = logsumexp(t*y)
    value(t) = V*softmax(t*y)
    value_source(t) = logsumexp(y + t*(V'*shift))
    @assert isapprox((value_source(h)-value_source(-h))/(2h), dot(b.v,shift); atol=1e-6)
    for (f,linear,quadratic) in ((score,b.s1-log(length(y)),b.s2-b.s1),
                                 (value,b.v1-value(0),b.v2-b.v1))
        @assert isapprox((f(h)-f(-h))/(2h),linear; atol=1e-6)
        @assert isapprox((f(h)+f(-h)-2f(0))/h^2,2quadratic; atol=1e-6)
    end
    # Two subkeys, the Gaussian kernel identity, and separable feature pooling.
    a = randn(rng,4)
    @assert logsumexp([dot(x,μ+a),dot(x,μ-a)]) - log(2) ≈ dot(x,μ)+log(cosh(dot(x,a)))
    @assert exp.(y) ≈ [exp(dot(x,x)/2 - sum(abs2,x-k)/2 + dot(k,k)/2) for k in eachcol(K)]
    features = exp.(K)
    query_features = exp.(x)
    weights = features'*query_features
    @assert V*weights/sum(weights) ≈ (V*features')*query_features/dot(vec(sum(features;dims=2)),query_features)
end

identities()
println("sample              rows   exact max      RMS 1      RMS 2      max W")
random_heads()
specimen()
