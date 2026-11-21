using PicoGPT, LinearAlgebra, Random, Printf
include("handbuilt.jl")
using .Handbuilt
function ids(X,p)
    Z,_ = affine(first(layernorm(X,p.γf,p.βf)),p.Wout,p.bout)
    vec(Z[2,:] .> Z[1,:])
end
function checkstock()
    for T in 2:4
        c,θ,gap = Handbuilt.stock(T)
        p = carve(θ,c)
        maxerr = 0.
        for word in 0:(2^T-1), inject in (true,false)
            x = digits(word,base=2,pad=T)
            U,_ = embed(reshape(x.+1,:,1),p.E)
            X,s = U,copy(x)
            for k in 1:32
                s = xor.(inject ? x : s,Handbuilt.left(s))
                X = first(block(X,only(p.blocks),1,T))
                if inject
                    X = X+U
                end
                @assert ids(X,p)==s
                C = positions(4,T) + hcat([Handbuilt.bit_state(4,v) for v in s]...)
                maxerr = max(maxerr,maximum(abs,X-(inject ? U+C : C)))
            end
        end
        println((T=T,F=c.F,params=length(θ),gap=gap,largest=maximum(abs,θ),maxerr=maxerr))
    end
end
function checkcompact()
    c = Handbuilt.compact_config()
    θ = Handbuilt.compact()
    p = carve(θ,c)
    b = only(p.blocks)
    rng = MersenneTwister(7)
    for n in (6,64)
        data = n==6 ? [digits(w,base=2,pad=n) for w in 0:(2^n-1)] :
             vcat([zeros(Int,n), ones(Int,n), Int.(isodd.(1:n))], [rand(rng,0:1,n) for i in 1:5])
        maxerr = 0.
        for word in data
            x = [0;word]
            U,_ = embed(reshape(x.+1,:,1),p.E)
            X,s = U,zeros(Int,n+1)
            for k in 1:n+3
                s = xor.(x,Handbuilt.left(s))
                X = Handbuilt.register_step(X,U,b)
                @assert ids(X,p)==s
                maxerr = max(maxerr,maximum(abs,X[7,:]-U[7,:]-100s))
                if k==n+2
                    @assert maximum(abs,Handbuilt.register_step(X,U,b)-X)<1e-9
                end
            end
        end
        println((bits=n,words=length(data),passes=n+3,params=length(θ),maxerr=maxerr))
    end
    # Explicit minimum before gain perturbation. 256-bit MPFR at integer/rational angles.
    gap = setprecision(256) do
        w = BigFloat[1,1//10]
        v(t) = [f(a*t)+1-cos(BigFloat(t)/1000) for a in w for f in (sin,cos)]
        R = zeros(BigFloat,4,4)
        for (i,a) in zip((1,3),w)
            R[i:i+1,i:i+1]=[cos(a) -sin(a); sin(a) cos(a)]
        end
        V = hcat(v.(0:64)...)
        scores = transpose(R*V)*V
        [minimum(scores[i,i-1]-maximum(scores[i,j] for j in 1:i if j!=i-1) for i in 2:T) for T in (64,65)]
    end
    println("Fourier score gaps at 64 and 65 columns ",Float64.(gap))
end
checkstock()
checkcompact()
function checkaddress(task,n)
    c,θ,gap = Handbuilt.addressed(task,n)
    p = carve(θ,c)
    b = only(p.blocks)
    _,_,_,_,state = Handbuilt.address_parts(n,task,1)
    err = 0.
    if task==:scan
        for w in 0:(2^n-1)
            s = digits(w,base=2,pad=n)
            a = collect(0:n-1)
            X = first(embed(reshape(s.+1,:,1),p.E))
            for k in 0:5
                X = first(block(X,b,1,n))
                @assert ids(X,p)==s
                Y = hcat([state(a[t],t,a[t],s[t]) for t in 1:n]...)
                err = max(err,maximum(abs,X-Y))
                s = [a[t]==0 ? s[t] : xor(s[t],s[a[t]]) for t in 1:n]
                a = [a[t]==0 ? 0 : a[a[t]] for t in 1:n]
            end
        end
    else
        for tuple in Iterators.product(ntuple(_->1:n,n)...)
            π = collect(tuple)
            allunique(π) || continue
            edges = [(j-1)*n+π[j] for j in 1:n]
            M = hcat([state(0,j,π[j],0) for j in 1:n]...)
            for firstaddr in 1:n
                a = firstaddr
                X = first(embed(reshape([edges;n*n+a],:,1),p.E))
                for k in 0:8
                    X = first(block(X,b,1,n+1))
                    Z = first(affine(first(layernorm(X,p.γf,p.βf)),p.Wout,p.bout))
                    @assert argmax(Z[:,end])==n*n+a
                    err = max(err,maximum(abs,X-hcat(M,state(a,0,0,0))))
                    a = π[a]
                end
            end
        end
    end
    println((task=task,n=n,D=c.D,F=c.F,params=length(θ),gap=gap,maxerr=err))
end
checkaddress(:scan,4)
checkaddress(:chase,3)
function checkanchors()
    c,θ,_ = Handbuilt.stock(4)
    p = carve(θ,c)
    U = first(embed(reshape([2,1,2,2],:,1),p.E))
    X = U
    for _ in 1:5
        X = first(block(X,only(p.blocks),1,4))+U
    end
    Y,back = block(X,only(p.blocks),1,4)
    rng = MersenneTwister(47)
    seed = randn(rng,size(X))
    shifts = ones(4)*transpose([.1,-.2,.3,-.4])
    gauge = maximum(abs,first(block(X+shifts,only(p.blocks),1,4))-Y-shifts)
    gauge_vjp = abs(dot(first(back(seed)),shifts)-dot(seed,shifts))
    @assert gauge < 1e-8 && gauge_vjp < 1e-8
    c8 = Handbuilt.compact_config()
    p8 = carve(Handbuilt.compact(),c8)
    U8 = first(embed(reshape([1,2,1,2],:,1),p8.E))
    X8 = U8
    for _ in 1:4
        X8 = Handbuilt.register_step(X8,U8,only(p8.blocks))
    end
    _,back8 = block(X8,only(p8.blocks),1,4)
    regseed = zero(X8)
    regseed[7,:] .= randn(rng,4)
    reg_vjp = maximum(abs,first(back8(regseed))-regseed)
    @assert reg_vjp < 1e-8
    println((stock_translation_error=gauge,stock_translation_vjp_error=gauge_vjp,
             register_vjp_norm=reg_vjp))
    T = 8
    different = 0
    for w in 0:(2^T-1)
        x = digits(w,base=2,pad=T)
        s = copy(x)
        for k in 1:2^ceil(Int,log2(T))
            prev = s
            s = xor.(s,Handbuilt.left(s))
            if k==2
                different += count(s .!= prev)
            end
            ref = [mod(sum(binomial(k,j)*x[t-j] for j in 0:min(k,t-1)),2) for t in 1:T]
            @assert ref==s
        end
        @assert s==x
    end
    denominator = T*2^T
    @assert different//denominator==(T-1)//(2T)
    # Count labeled four-by-four completions without using any Sudoku theorem.
    digits4 = collect(1:4)
    rows = [collect(t) for t in Iterators.product(ntuple(_->digits4,4)...) if allunique(t)]
    boards = 0
    for rr in Iterators.product(ntuple(_->rows,4)...)
        g = hcat(rr...)
        if all(allunique,eachrow(g)) && all(allunique(g[i:i+1,j:j+1]) for i in (1,3),j in (1,3))
            boards += 1
        end
    end
    @assert boards==288
    println((T=T,mixed_horizon_disagreement=different//denominator,completed_boards=boards))
end
checkanchors()
