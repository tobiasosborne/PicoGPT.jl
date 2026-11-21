# Run from any directory with the root PicoGPT Julia project active.
using PicoGPT, LinearAlgebra, Statistics, Random
include(joinpath(@__DIR__, "../../scripts/window_round2.jl"))

# Prefixes suffice. Each t-bit prefix has 2^(T-t) completions. Keep the
# rightmost bit in the low bit of the key, repeatedly copying the first bit
# on the left. Compare labels both within a position and pooled over positions.
const Histogram = Dict{Int,Vector{Int}}
lossmass(a) = sum(n * log(sum(a) / n) for n in a if n > 0)
function context_histogram(T, k)
    by_context = Histogram()
    for t in 1:T
        by_site = Histogram()
        for w in 0:2^t-1
            padding = (w >> (t-1)) == 1 ? -1 : 0
            key = ((padding << t) | w) & (2^(k+1) - 1)
            label = 1 + isodd(count_ones(w))
            for bins in (by_site, by_context)
                get!(() -> [0,0], bins, key)[label] += 2^(T-t)
            end
        end
        @assert all(a -> t <= k+1 ? minimum(a) == 0 : a[1] == a[2], values(by_site))
    end
    by_context
end

function floors()
    fraction(T,K) = sum(max(T-k-1,0) for k in 1:K) // (T*K)
    weighted = sum(sum(fraction(T,K) for K in T-1:2T) / (T+2) for T in 4:12) / 9
    one = sum(fraction(T,1) for T in 4:12) / 9
    @assert fraction(8,8) == 21//64
    @assert weighted == 384975233676617//1653814216454400
    @assert one == 89539//124740
    println("Positioned floors ", log(2) .* (fraction(8,8), weighted, one))

    terminal_error(T) = sum((1 + (T-r)//2) / big(2)^r for r in 2:2:T) / T
    conditional = 0.0
    pooled = Dict{Tuple{Int,Int},Vector{Float64}}()
    for T in 4:12, k in 1:2T
        hist = context_histogram(T,k)
        population = T * 2^T
        if k >= T-1
            @assert sum(minimum, values(hist)) // population == terminal_error(T)
        end
        # Probability mass of one (T,k) term in a random training loss.
        weight = sum(1/K for K in max(T-1,k):2T) / (9 * (T+2))
        conditional += weight * sum(lossmass, values(hist)) / population
        for (key, counts) in hist
            get!(() -> zeros(2), pooled, (k,key)) .+= (weight/population) .* counts
        end
    end
    @assert isapprox(conditional,0.4323451798807894; atol=1e-13)
    @assert isapprox(sum(sum.(collect(values(pooled)))),1; atol=1e-13)
    println("NoPE deep CE, length revealed / hidden ", (conditional,sum(lossmass.(collect(values(pooled))))))
    println("NoPE terminal accuracy ", [Float64(1-terminal_error(T)) for T in (8,32,64)])
end

# Read only the saved parameters and reference losses using Python's stdlib
# JSON reader. All model evaluations and derivatives below run in Julia.
function checkpoint_rows(artifact, kind)
    folder = joinpath(@__DIR__, "../../artifacts")
    query = """
    import json, sys
    runs = json.load(open(sys.argv[1]))['runs']
    for r in runs:
        if sys.argv[2] == 'prefix' and r['name'] != 'gain': continue
        print(r['seed'], r['curve'][-1]['loss'], *r['θ'])
    """
    return [parse.(Float64, split(row)) for row in readlines(`python3 -c $query $(joinpath(folder,artifact*".json")) $kind`)]
end

function transport_path(g, x)
    T = length(x)
    c = Config(; V=2, D=2, H=1, F=1, L=1, T)
    Z, back = register_window(g, reshape(x,:,1), c; K=T)
    d = zero(Z)
    d[1,end], d[2,end] = -1, 1
    frames = NamedTuple[]
    back(d; adjoints=frames)
    return [vec(frame.X̄) for frame in frames]
end

function prefix_checks()
    c = Config(; V=2,D=2,H=1,F=1,L=1,T=12)
    for row in checkpoint_rows("window-round2-register","prefix")
        seed, published = row
        g = row[3:end]
        @assert g[1]>0 && g[2]<0 && tanh(0.99 * minimum(abs,g))>0.99
        objective = 0.0
        for T in 4:12
            x = binary_words(T)
            y = accumulate(xor,x .== 2;dims=1)
            cumulative = 0.0
            WindowLoop.visit_register(g,x,c; K=2T) do k,Z,X,previous,weights
                u = reshape(Z[2,:],size(x))
                t = min(k+1,T)
                @assert all((2 .* y[1:t,:] .- 1) .* u[1:t,:] .>= 0.99 * minimum(abs,g))
                @assert all(iszero,u[t+1:end,:])
                k>T && @assert isequal(X,previous)
                # Ordinary hard labels throughout. Stable binary logistic CE.
                cumulative += mean(log1p.(exp.(-(2 .* y .- 1) .* u)))
                k>=T-1 && (objective += cumulative / (9*(T+2)*k))
            end
        end
        @assert abs(objective - published) < 2e-13
        println("Prefix seed ",Int(seed)," mixed CE ",objective)
    end
    g = first(checkpoint_rows("window-round2-register","prefix"))[3:end]
    probe = transport_path(g,rand(Xoshiro(723),1:2,16))
    for (j,row) in enumerate(probe[1:16])
        @assert findall(!iszero,row) == [17-j]
    end
    @assert all(iszero,probe[end])
    println("16-site descending path ",[abs(probe[j][17-j]) for j in (1,2,3,16)])
    long = transport_path(g,ones(Int,64))
    @assert any(row -> all(iszero,row),long[2:end-1])
    println("Seed 1, 64 neutral bits, first all-zero cotangent at pass ",
            65-findfirst(row -> all(iszero,row),long))
    @assert tanh(20.0) == 1
end

function rule_checks()
    for k in 0:96, j in 0:k
        @assert ((j & k) == j) == isodd(binomial(big(k),j))
    end
    c = Config(; V=2,D=2,H=1,F=1,L=1,T=12)
    for row in checkpoint_rows("window-round2-rule60","rule")
        seed, published = row
        a, b = row[3:end]
        @assert a>0>b
        margin = 0.99*(0.995 * min(a,-b) - 0.005 * max(a,-b))
        @assert tanh(margin)>0.99
        for x in (eval_words(64),vcat(fill(2,1,1),ones(Int,96,1)))
            expected = x .== 2
            WindowLoop.visit_register([a,b],x,c; K=96,task=:rule60) do k,Z,X,previous,weights
                expected = xor.(expected,vcat(falses(1,size(x,2)),expected[1:end-1,:]))
                @assert (reshape(Z[2,:],size(x)) .> 0) == expected
                @assert (lucas_window(x,k) .== 2) == expected
            end
        end
        println("Rule 60 seed ",Int(seed)," minimum absolute logit bound ",margin)
    end
end

BLAS.set_num_threads(1)
floors()
prefix_checks()
rule_checks()
