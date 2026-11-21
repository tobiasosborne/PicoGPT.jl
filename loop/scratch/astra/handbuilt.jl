module Handbuilt
using PicoGPT, LinearAlgebra

unit(D, i) = Float64.(1:D .== i)
function blank(c)
    θ = zeros(nparams(c))
    p = carve(θ,c)
    b = only(p.blocks)
    for γ in (p.γf,b.γ1,b.γ2)
        fill!(γ,1)
    end
    θ,p,b
end

"""D=8 linear-front parity. Returns a flat θ; use register_step, BOS=even, then bit ids 1,2."""
compact_config() = Config(V=2,D=8,H=1,F=4,L=1,T=65)
function compact(c=compact_config())
    θ,p,b = blank(c)
    e = [unit(8,i) for i in 1:8]
    carrier = 1e8
    σ₀ = carrier*sqrt(7)/8
    p.E[5,:] .= carrier
    p.E[6,2] = 100
    C = reduce(vcat,[transpose(e[i]-e[8]) for i in 1:4])
    R = zeros(4,4)
    for (i,w) in ((1,1.),(3,.1))
        R[i:i+1,i:i+1] = [cos(w) -sin(w); sin(w) cos(w)]
    end
    b.Wk[1:4,:] .= σ₀*C
    b.bk[1:4] .= 1
    b.Wq[1:4,:] .= 1024sqrt(8)*σ₀*R*C
    b.bq[1:4] .= 1024sqrt(8)*R*ones(4)
    b.Wv[1,:] .= σ₀*(e[7]-e[8])
    b.Wo[1,1] = 1
    # The normalized contrast reads (local input + received state) in units of 100.
    for i in 1:4
        b.W1[i,:] .= (σ₀/100)*(e[6]+e[1]-2e[8])
    end
    b.b1 .= -[.25,.75,1.25,1.75]
    b.W2[7,:] .= [200,-200,-200,200]
    p.Wout[2,:] .= (σ₀/100)*(e[7]-e[8])
    p.bout[2] = -.5
    θ
end

"""U + e₇e₇ᵀ(B(X)−X), including the unchanged sinusoidal code from U."""
function register_step(X,U,b)
    Y,_ = block(X,b,1,size(X,2))
    next = copy(U)
    next[7,:] .+= Y[7,:] .- X[7,:]
    next
end

# The following finite table uses the unmodified Julia block for both ordinary loops.
bit_input(D,x) = 3x*unit(D,3) + 3(2x-1)*unit(D,4)
bit_state(D,s) = 12(2s-1)*unit(D,4)
left(s) = [0; s[1:end-1]]
function stock_rows(T)
    rows = Tuple{Matrix{Float64}, Vector{Float64}}[]
    for t in 1:T, word in 0:(2^t-1)
        x = digits(word,base=2,pad=t)
        P = positions(4,t)
        U = P + reduce(hcat,(bit_input(4,v) for v in x);init=zeros(4,0))
        C(s) = P + reduce(hcat,(bit_state(4,v) for v in s);init=zeros(4,0))
        s = xor.(x,left(x))
        push!(rows,(U,C(s)[:,end]),(C(x),C(s)[:,end]))
        for k in 1:t
            snew = xor.(x,left(s))
            push!(rows,(C(s)+U,C(snew)[:,end]))
            s = snew
        end
    end
    rows
end

"""Finite-point ReLU interpolation. Writes the residual correction in PicoGPT layout."""
function install_table!(b,points,outputs)
    # All points include the real LN₂. Scalar projection leaves distinct table entries separated.
    v = [sin(i*i+1) for i in axes(b.W1,2)]
    u = [dot(v,z) for z in points]
    order = sortperm(u)
    lo = first(order)
    b.b2 .= outputs[lo]
    for (i,(a,j)) in enumerate(zip(order,order[2:end]))
        Δ = u[j]-u[a]
        Δ > 1e-9 || error("indistinguishable interpolation keys: gap $Δ")
        slope = 2(outputs[j]-outputs[a])/Δ
        for (h,knot,sign) in ((2i-1,u[a]+Δ/4,1),(2i,u[j]-Δ/4,-1))
            b.W1[h,:] .= v
            b.b1[h] = -knot
            b.W2[:,h] .= sign*slope
        end
    end
    minimum(diff(sort(u)))
end

"""Default T=4. Returns (c,θ,min_gap). Same θ solves parity (B(X)+U) and Rule 60 (B(X))."""
function stock(T=4)
    1 <= T <= 8 || throw(ArgumentError("finite-table constructor supports 1 to 8 positions"))
    e = [unit(4,i) for i in 1:4]
    value = [1.,-.7,.3,.6]
    v = [sin(i*i+1) for i in 1:4]
    points,outputs = Vector{Float64}[],Vector{Float64}[]
    seen = Dict{Float64,Int}()
    for (X,y) in stock_rows(T)
        z,_ = layernorm(X,ones(4),zeros(4))
        # Uniform attention at the final column. Use the same matrix operations as the block.
        Y = X[:,end] + e[1]*(sum(transpose(value)*z)/size(X,2))
        Z = vec(first(layernorm(reshape(Y,4,1),ones(4),zeros(4))))
        key = dot(v,Z)
        correction = y-Y
        j = get(seen,key,0)
        if j == 0
            push!(points,Z)
            push!(outputs,correction)
            seen[key] = length(points)
        else
            isapprox(correction,outputs[j];atol=1e-12) || error("inconsistent table")
        end
    end
    c = Config(V=2,D=4,H=1,L=1,T=T,F=2(length(points)-1))
    θ,p,b = blank(c)
    p.E .= hcat(bit_input(4,0),bit_input(4,1))
    b.Wv[1,:] .= value
    b.Wo[1,1] = 1
    p.Wout[2,:] .= e[4]-e[2]
    gap = install_table!(b,points,outputs)
    c,θ,gap
end
stock_theta(T=4) = stock(T)[2]

# Compiled bounded-size examples. One preparation pass converts the stock embeddings
# to centred address registers. The same ordinary block then makes each lookup.
function address_parts(n,task,F)
    m = n+1
    D = 2(4m+2)+4
    c = Config(V=task==:scan ? 2 : n*n+n,D=D,H=1,L=1,T=task==:scan ? n : n+1,F=F)
    θ,p,b = blank(c)
    u(j) = (unit(D,2j-1)-unit(D,2j))/sqrt(2)
    G(i) = hcat([u(i*m+j) for j in 1:m]...)
    Q,K,V,S = G.(0:3)
    bit,scratch = u(4m+1),u(4m+2)
    σ = sqrt(4/D+1e-5)
    b.Wq[1:m,:] .= 4sqrt(D)*σ*transpose(Q)
    b.Wk[1:m,:] .= σ*transpose(K)
    b.Wv[1:m,:] .= σ*transpose(V)
    b.Wv[m+1,:] .= σ*bit
    b.Wo[:,1:m] .= S
    b.Wo[:,m+1] .= scratch
    # Raw tokens are identified by their id and the stock sinusoidal positions.
    for j in 1:c.V
        p.E[D-2,j] = j
    end
    if task==:scan
        p.Wout[2,:] .= bit
    else
        for j in 1:n
            p.Wout[n*n+j,:] .= Q[:,j+1]
        end
    end
    state(q,k,v,s) = Q[:,q+1]+K[:,k+1]+V[:,v+1]+(2s-1)*bit
    c,θ,p,b,state
end

"""Bounded stock scan (:scan) or permutation chase (:chase). Returns c,θ,min_gap."""
function addressed(task,n)
    task in (:scan,:chase) || throw(ArgumentError("expected :scan or :chase"))
    2<=n<=4 || throw(ArgumentError("constructors enumerate tasks with 2 to 4 labels"))
    c,_,p,b,state = address_parts(n,task,1)
    points,outputs = Vector{Float64}[],Vector{Float64}[]
    function add(X,target)
        A,_ = layernorm(X,b.γ1,b.β1)
        score = transpose(b.Wk*A)*(b.Wq*A[:,end])/sqrt(c.D)
        Y = X[:,end]+b.Wo*((b.Wv*A)*softmax(score))
        Z = vec(first(layernorm(reshape(Y,:,1),b.γ2,b.β2)))
        push!(points,Z)
        push!(outputs,target-Y)
    end
    function addall(X,Y)
        for j in axes(X,2)
            add(X[:,1:j],Y[:,j])
        end
    end
    if task==:scan
        for w in 0:(2^n-1)
            s = digits(w,base=2,pad=n)
            a = collect(0:n-1)
            canonical() = hcat([state(a[t],t,a[t],s[t]) for t in 1:n]...)
            X = first(embed(reshape(s.+1,:,1),p.E))
            Y = canonical()
            addall(X,Y)
            for k in 1:ceil(Int,log2(n))+1
                X = Y
                s = [a[t]==0 ? s[t] : xor(s[t],s[a[t]]) for t in 1:n]
                a = [a[t]==0 ? 0 : a[a[t]] for t in 1:n]
                Y = canonical()
                addall(X,Y)
            end
        end
    else
        for tuple in Iterators.product(ntuple(_->1:n,n)...)
            π = collect(tuple)
            allunique(π) || continue
            edge = [(j-1)*n+π[j] for j in 1:n]
            M = hcat([state(0,j,π[j],0) for j in 1:n]...)
            for a in 1:n
                X = first(embed(reshape([edge;n*n+a],:,1),p.E))
                Y = hcat(M,state(a,0,0,0))
                addall(X,Y)
                addall(Y,hcat(M,state(π[a],0,0,0)))
            end
        end
    end
    v = [sin(i*i+1) for i in 1:c.D]
    order = sortperm([dot(v,z) for z in points])
    distinct = Int[]
    for i in order
        if !isempty(distinct) && norm(points[i]-points[last(distinct)])<1e-11
            norm(outputs[i]-outputs[last(distinct)])<1e-10 || error("inconsistent output")
        else
            push!(distinct,i)
        end
    end
    c,θ,_,b,_ = address_parts(n,task,2(length(distinct)-1))
    gap = install_table!(b,points[distinct],outputs[distinct])
    c,θ,gap
end
addressed_theta(task,n) = addressed(task,n)[2]

end
