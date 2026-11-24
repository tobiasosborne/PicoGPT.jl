# From the repository root: julia -t1 -O0 --project=. grow/notes/check_definition.jl
using PicoGPT, LinearAlgebra, Random, Printf
BLAS.set_num_threads(1)
const rng = MersenneTwister(731)
const residuals = Float64[]
const fd_errors = Float64[]
function same(a, b)
    err = maximum(abs, a .- b)
    @assert err < 2e-13 "identity residual $err"
    push!(residuals, err)
end
function fd(f, x, g)
    # All coordinates, including dormant ones. Absolute errors near zero.
    probe = copy(x)
    for i in eachindex(x)
        h = 2e-6
        probe[i] = x[i] + h
        plus = f(probe)
        probe[i] = x[i] - h
        minus = f(probe)
        probe[i] = x[i]
        err = abs((plus-minus)/(2h) - g[i])
        @assert err < 2e-7 "finite difference at $i: $err"
        push!(fd_errors, err)
    end
end
function copyblock!(b, a)
    for name in keys(a)
        b[name] .= a[name]
    end
end
logits(θ,x,c) = first(gpt(θ,x,c))
probs(z) = softmax(z; dims=1)
const c = Config(V=3,D=4,H=2,F=5,L=1,T=3)
const θ = init(c; rng)
const p = carve(θ,c)
# Include nonzero biases and nontrivial LayerNorm gain and offset.
θ .+= 0.13 .* randn(rng, length(θ))
const x = reshape([1,2,1],:,1)
const y = reshape([2,1,3],:,1)
const z = logits(θ,x,c)
const X = first(embed(x,p.E))
const G = randn(rng,size(X))
const b = only(p.blocks)
const parent_loss, parent_back = loss(θ,x,y,c)
const parent_grad = parent_back()

function vocab()
    fibre = [1,1,2,3]
    q = [0.3,0.7,1,1]
    child = Config(V=4,D=c.D,H=c.H,F=c.F,L=1,T=c.T)
    t = init(child; rng)
    u = carve(t,child)
    u.E .= p.E[:,fibre]
    copyblock!(only(u.blocks),b)
    u.γf .= p.γf
    u.βf .= p.βf
    u.Wout .= p.Wout[fibre,:]
    u.bout .= p.bout[fibre] .+ log.(q)
    xx, yy = reshape([1,3,2],:,1), reshape([3,2,4],:,1)
    zz = logits(t,xx,child)
    same(probs(zz), probs(z)[fibre,:] .* q)
    l, back = loss(t,xx,yy,child)
    same(l,parent_loss - sum(log,q[yy])/length(yy))
    g = back()
    fd(v->first(loss(v,xx,yy,child)), t,g)
    gu, gp = carve(g,child), carve(parent_grad,c)
    for v in 1:c.V
        ids = findall(==(v),fibre)
        same(sum(gu.E[:,ids];dims=2)[:],gp.E[:,v])
        same(vec(sum(gu.Wout[ids,:];dims=1)),gp.Wout[v,:])
        same(sum(gu.bout[ids]),gp.bout[v])
    end
    same(flatten(gu.blocks,gu.γf,gu.βf),flatten(gp.blocks,gp.γf,gp.βf))
    @assert norm(gu.E[:,1]-gu.E[:,2]) > 1e-5
    # Coarse-labelled loss, differentiated w.r.t. all fine logits.
    mass = [sum(probs(zz)[fibre.==v,n]) for v in 1:c.V, n in 1:length(y)]
    delta = copy(mass)
    for (n,v) in enumerate(y)
        delta[v,n] -= 1
    end
    function coarse_loss(scores)
        pp = probs(scores)
        -sum(log(sum(pp[fibre.==v,n])) for (n,v) in enumerate(y))/length(y)
    end
    fd(coarse_loss, zz, delta[fibre,:] .* q ./ length(y))
end

function passes()
    # The same block can be the old pass and the optional next pass.
    Y = first(block(X,b,c.H,c.T))
    inc = first(block(Y,b,c.H,c.T)) - Y
    gate(a) = dot(G, Y + a[1]*inc)
    same(gate([0.0]), dot(G,Y))
    fd(gate, [0.0], [dot(G,inc)])
    function shared(v, a)
        bv = only(carve(v,c).blocks)
        y1 = first(block(X,bv,c.H,c.T))
        dot(G, y1 + a*(first(block(y1,bv,c.H,c.T)) - y1))
    end
    gb = last(block(X,b,c.H,c.T))(G)[2]
    g = zeros(length(θ))
    for (out,grad) in zip(values(only(carve(g,c).blocks)),gb)
        out .= grad
    end
    fd(v->shared(v,0.0),θ,g)
    t = copy(θ)
    u = only(carve(t,c).blocks)
    for key in (:Wo,:bo,:W2,:b2)
        fill!(u[key],0)
    end
    trace = []
    Y0, back = block(X,u,c.H,c.T;trace)
    same(Y0,X)
    gX, gb0 = back(G)
    same(gX,G)
    gv = zero(t)
    dest = only(carve(gv,c).blocks)
    dest.Wo .= G * only(trace).O'
    dest.W2 .= G * only(trace).R'
    dest.bo .= dest.b2 .= vec(sum(G;dims=2))
    same(flatten(gb0),flatten(dest))
    fd(v->dot(G,first(block(X,only(carve(v,c).blocks),c.H,c.T))),t,gv)
end

function mlp_heads()
    cw = Config(V=c.V,D=c.D,H=c.H,F=c.F+2,L=1,T=c.T)
    t = init(cw; rng)
    bw = only(carve(t,cw).blocks)
    for key in keys(b)
        if key == :W1
            bw.W1[1:c.F,:] .= b.W1
        elseif key == :b1
            bw.b1[1:c.F] .= b.b1
        elseif key == :W2
            bw.W2[:,1:c.F] .= b.W2
            bw.W2[:,c.F+1:end] .= 0
        else
            bw[key] .= b[key]
        end
    end
    trace = []
    yw, back = block(X,bw,c.H,c.T;trace)
    same(yw,first(block(X,b,c.H,c.T)))
    gw = zero(t)
    dest = only(carve(gw,cw).blocks)
    for (out,grad) in zip(values(dest),last(back(G)))
        out .= grad
    end
    same(dest.W2[:,c.F+1:end],G*only(trace).R[c.F+1:end,:]')
    same(dest.W1[c.F+1:end,:],0)
    same(dest.b1[c.F+1:end],0)
    fd(v->dot(G,first(block(X,only(carve(v,cw).blocks),c.H,c.T))),t,gw)

    # Independent extra head of original head width in an expanded attention module.
    A = first(layernorm(X,b.γ1,b.β1))
    op(W,bias) = W*A .+ bias
    oldQ,oldK,oldV = op(b.Wq,b.bq),op(b.Wk,b.bk),op(b.Wv,b.bv)
    oldO = first(mix(first(attention(oldQ,oldK,c.H,c.T)),oldV))
    d = c.D ÷ c.H
    function head(v)
        W = reshape(v[1:end-d*c.D],3d,c.D+1)
        C = reshape(v[end-d*c.D+1:end],c.D,d)
        q,k,val = (W[rows,:]*vcat(A,ones(1,c.T)) for rows in (1:d,d+1:2d,2d+1:3d))
        on = first(mix(first(attention(q,k,1,c.T)),val))
        b.Wo*oldO .+ b.bo + C*on, on
    end
    v = vcat(randn(rng,3d*(c.D+1)),zeros(c.D*d))
    out,on = head(v)
    same(out,b.Wo*oldO .+ b.bo)
    fd(vv->dot(G,first(head(vv))),v,vcat(zeros(3d*(c.D+1)),vec(G*on')))
    basis = Matrix{Float64}(I,4,4) .- 1/4
    inc = first(layernorm(basis,ones(4),zeros(4)))
    @assert rank(inc[:,2:end] .- inc[:,1]) == 3
end

function narrow_head()
    # Rank-two scores and value-output map, factored into a narrower active head.
    U = vcat(X,ones(1,c.T))
    d,small = 4,2
    A,B,V = (randn(rng,small,c.D+1) for _ in 1:3)
    C = randn(rng,c.D,small)
    pad = Matrix{Float64}(I,d,small)
    # A change of basis ensures the old head can use all of its rows.
    basis = Matrix(qr(randn(rng,d,d)).Q)
    embed = basis*pad
    q,k,v = (embed*W*U for W in (A,B,V))
    out = C*embed'*first(mix(first(attention(q,k,1,c.T)),v))
    factor = (small/d)^(1/4)
    qnew = vcat(factor*A*U,randn(rng,small,c.T))
    knew = vcat(factor*B*U,randn(rng,small,c.T))
    vnew = vcat(V*U,randn(rng,small,c.T))
    outnew = hcat(C,zeros(c.D,small))*first(mix(first(attention(qnew,knew,2,c.T)),vnew))
    same(out,outnew)
end

function repeatmap(n,r)
    repeat(Matrix{Float64}(I,n,n);inner=(r,1))
end
function widen(v,cw,J,R,Jf,Rf)
    a = carve(v,c)
    u = only(a.blocks)
    t = zeros(nparams(cw))
    w = carve(t,cw)
    bw = only(w.blocks)
    scale = (cw.D/c.D)^(-1/4)
    w.E .= J*a.E
    for name in (:γ1,:β1,:γ2,:β2,:bv,:bo,:b2)
        bw[name] .= J*u[name]
    end
    for name in (:Wq,:Wk)
        bw[name] .= scale*J*u[name]*R
    end
    for name in (:bq,:bk)
        bw[name] .= scale*J*u[name]
    end
    for name in (:Wv,:Wo)
        bw[name] .= J*u[name]*R
    end
    bw.W1 .= Jf*u.W1*R
    bw.b1 .= Jf*u.b1
    bw.W2 .= J*u.W2*Rf
    w.γf .= J*a.γf
    w.βf .= J*a.βf
    w.Wout .= a.Wout*R
    w.bout .= a.bout
    t
end
function width(s)
    r = 2
    cw = Config(V=c.V,D=r*c.D,H=c.H,F=s*c.F,L=1,T=c.T)
    J,Jf = repeatmap(c.D,r),repeatmap(c.F,s)
    R,Rf = J'/r,Jf'/s
    lift(v) = widen(v,cw,J,R,Jf,Rf)
    t = lift(θ)
    injectedP = J*positions(c.D,c.T)
    function copied(v; gradient=false)
        w = carve(v,cw)
        # The only departure from stock gpt is its fixed input position table.
        xx = w.E[:,vec(x)] + injectedP
        out,back = block(xx,only(w.blocks),cw.H,cw.T)
        ff,backln = layernorm(out,w.γf,w.βf)
        zz,backz = affine(ff,w.Wout,w.bout)
        l,dl = crossentropy(zz,y)
        if !gradient
            return l
        end
        df,dw,db = backz(dl())
        dout,dγ,dβ = backln(df)
        dx,gb = back(dout)
        de = zero(w.E)
        for (n,v) in enumerate(x)
            de[:,v] .+= dx[:,n]
        end
        l,zz,flatten(de,[gb],dγ,dβ,dw,db)
    end
    l,zz,g = copied(t;gradient=true)
    same(zz,z)
    same(l,parent_loss)
    w = carve(t,cw)
    tr,trw = [],[]
    out = first(block(X,b,c.H,c.T;trace=tr))
    outw = first(block(J*X,only(w.blocks),cw.H,cw.T;trace=trw))
    same(only(tr).A,only(trw).A)
    same(J*out,outw)
    same(R*outw,out)
    fd(copied,t,g)
    # The table is linear in all parent parameters. Pull back the child gradient.
    basis = zero(θ)
    for i in eachindex(θ)
        basis[i] = 1
        same(dot(g,lift(basis)),parent_grad[i])
        basis[i] = 0
    end
    gw = carve(g,cw)
    same(J*R*gw.E,gw.E)
    same(gw.Wout*J*R,gw.Wout)
    for name in (:Wq,:Wk,:Wv,:Wo)
        M = only(gw.blocks)[name]
        same(J*R*M*J*R,M)
    end
    # Off-section LN and block fail, even at T=1.
    x0 = reshape([1.,-1.,0.,2.],:,1)
    delta = zeros(cw.D,1)
    delta[1:2] .= (1.,-1.)
    xw = J*x0 + delta
    same(R*xw,x0)
    bw = only(w.blocks)
    u = only(carve(copy(θ),c).blocks)
    Iold = Matrix{Float64}(I,c.D,c.D)
    for (bb,J0,R0) in ((u,Iold,Iold),(bw,J,R))
        bb.γ1 .= 1
        bb.β1 .= bb.bv .= bb.bo .= bb.b2 .= 0
        bb.Wv .= bb.Wo .= J0*R0
        bb.W2 .= 0
    end
    @assert norm(R*first(block(xw,bw,cw.H,1))-first(block(x0,u,c.H,1))) > 1e-4
end

function contexts()
    cc = Config(V=c.V,D=c.D,H=c.H,F=c.F,L=1,T=5)
    xx = reshape([1,2,1,3,2],:,1)
    zl,back = gpt(θ,xx,cc)
    same(zl[:,1:c.T],z)
    same(logits(θ,x,cc),z)
    same(positions(c.D,cc.T)[:,1:c.T],positions(c.D,c.T))
    cov = randn(rng,size(z))
    covlong = hcat(cov,zeros(c.V,2))
    grad = back(covlong)
    same(grad,last(gpt(θ,x,c))(cov))
    fd(v->dot(logits(v,xx,cc),covlong),θ,grad)
    function witness(D)
        small = Config(V=2,D=D,H=1,L=0,T=2)
        t = zeros(nparams(small))
        w = carve(t,small)
        w.γf .= 1
        w.Wout[1,1:2:end] .= 2/D
        logits(t,ones(Int,2,1),small)[1,:]
    end
    a,f = witness(2),witness(4)
    @assert a[1] < 0 < a[2] && f[2] < 0
    same(a[1],f[1])
    @printf("Stock width 2→4, first-row logit at position two: %.8f → %.8f\n",a[2],f[2])
end

function blocked()
    q = [-1.,0.,1.]
    keys = [-1.,1.,0.,0.]
    vals = [1.,1.,0.,0.]
    for query in q
        prob = softmax(query*keys)
        same(log(sum(prob[1:2])/sum(prob[3:4])),log(cosh(query)))
        same(dot(vals,prob),sum(prob[1:2]))
    end
    @assert 2log(cosh(1.)) > 0 # discrete second difference of the log-odds
    function row(v)
        k,val = eachcol(reshape(v,:,2))
        s = softmax(0.7 .* k)
        out = dot(s,val)
        out,vcat(0.7 .* s .* (val .- out),s)
    end
    orig = [-0.2,0.9,-1.,2.]
    dup = repeat(orig;inner=2)
    coarse,g = row(orig)
    fine,gf = row(dup)
    same(fine,coarse)
    same(gf,repeat(g;inner=2)/2)
    fd(v->first(row(v)),orig,g)
    fd(v->first(row(v)),dup,gf)
end

function blocked_stock()
    # Three independent feature channels in the centered four-coordinate stream.
    axes = [1. 1. 1.;-1. 1. 1.;0. -2. 1.;0. 0. -3.]
    axes ./= sqrt.(sum(abs2,axes;dims=1))
    cc = Config(V=7,D=4,H=4,L=1,F=1,T=5)
    t = zeros(nparams(cc))
    w = carve(t,cc)
    b = only(w.blocks)
    b.γ1 .= 1
    b.Wq[1,:] .= axes[:,1]
    b.Wk[1,:] .= axes[:,2]
    b.Wv[1,:] .= axes[:,3]
    P = positions(4,5)
    for (token,(q,k,v)) in enumerate(((0,-1,1),(0,1,1),(0,0,0),(0,0,0),
                                       (-1,0,0),(0,0,0),(1,0,0)))
        feature = axes*[q,k,v]
        # Invert the epsilon-regularized LN on the interior of its centered ball.
        w.E[:,token] .= sqrt(1e-5/(1-dot(feature,feature)/4))*feature - P[:,min(token,5)]
    end
    for token in 5:7
        trace = []
        gpt(t,reshape([1,2,3,4,token],:,1),cc;trace)
        A = trace[2].A
        mass = A[5,:,1,1]
        same(log(sum(mass[1:2])/sum(mass[3:4])),log(cosh(token-6)))
        same(trace[2].O[1,5],sum(mass[1:2]))
    end
end

function external_only()
    # Equal output kernels coexist with incompatible input fibres.
    fine = copy(θ)
    parent = copy(θ)
    fc,pc = carve(fine,c),carve(parent,c)
    fc.Wout .= pc.Wout .= 0
    fc.E .= 0
    xx = reshape([1,2],1,2)
    same(logits(fine,xx,c),logits(parent,xx,c))
    xf = first(embed(xx,fc.E))
    xp = first(embed(xx,pc.E))
    same(xf[:,1],xf[:,2])
    @assert norm(xp[:,1]-xp[:,2]) > 1e-3
end

function worlds()
    # Rows are fine states. Check both intertwining and path marginals.
    C = [1 0;1 0;0 1]
    K = [0.1 0.6 0.3;0.65 0.05 0.3;0.2 0.2 0.6]
    Q = [0.7 0.3;0.4 0.6]
    same(K*C,C*Q)
    same(K^4*C,C*Q^4)
    kl(a,b) = sum(a .* log.(a ./ b))
    tv(a,b) = sum(abs,a-b)/2
    bern(p) = [p,1-p]
    a,b,d = bern(0.8),bern(0.5),bern(0.2)
    @assert kl(a,d) > kl(a,b)+kl(b,d)
    @assert tv(a,d) <= tv(a,b)+tv(b,d)+eps()
    @assert tv(a,d) <= sqrt(kl(a,b)/2)+sqrt(kl(b,d)/2)
    # Explicit conditional mutual information and the best coarse predictor.
    P = [0.7 0.3;0.2 0.8;0.5 0.5]
    μ = [0.25,0.25,0.5]
    m = C'*μ
    star = (C'*(μ .* P)) ./ m
    label = [1,1,2]
    err = sum(μ[i]*kl(P[i,:],Q[label[i],:]) for i in eachindex(μ))
    info = sum(μ[i]*kl(P[i,:],star[label[i],:]) for i in eachindex(μ))
    same(err,info+sum(m[i]*kl(star[i,:],Q[i,:]) for i in eachindex(m)))
    @assert info > 0
    ref = [0.6 0.4;0.25 0.75]
    same(kl(vec(a.*Q),vec(b.*ref)),kl(a,b)+sum(a[i]*kl(Q[i,:],ref[i,:]) for i in 1:2))
    # Strong versus initial-law-only lumpability, and the long-memory example.
    bad = [0 0 1;0 1 0;0 0 1]
    @assert (bad*C)[1,:] != (bad*C)[2,:]
    same([0 1 0]*bad^3,[0 1 0])
    aStay,bStay = 0.8,0.3
    posterior = [0.5,0.5]
    predictions = Float64[]
    for n in 1:8
        pred = dot(posterior,[aStay,bStay])
        same(pred,(aStay^n+bStay^n)/(aStay^(n-1)+bStay^(n-1)))
        push!(predictions,pred)
        posterior = posterior .* [aStay,bStay] ./ pred
    end
    @assert all(diff(predictions) .> 0)
    # Action-dependent stochastic rewards: two-stage discounted return laws.
    # Two selectors below choose distinct representatives of each coarse action.
    maps = [Q,ref]
    function returns(i,coarse,rep)
        fine = [K,[0.1 0.5 0.4;0.4 0.2 0.4;0.1 0.15 0.75]]
        # Swap how mass is distributed between hidden states of the same class.
        kernels = coarse ? maps : [P[:,rep == 1 ? [1,2,3] : [2,1,3]] for P in fine]
        class(j) = coarse ? j : label[j]
        dist = zeros(4)
        for j in axes(first(kernels),1), k in axes(first(kernels),1)
            # First action = 1, next action = observed coarse state.
            reward1 = class(j)-1
            reward2 = class(k)-1
            dist[1+2reward1+reward2] += kernels[1][i,j]*kernels[class(j)][j,k]
        end
        dist
    end
    for i in eachindex(label), rep in (1,2)
        same(returns(i,false,rep),returns(label[i],true,rep))
    end
    # Score-based local metric and iid sign fixed-point derivative.
    u,v = [0.7,0.3],[0.1,-0.1]
    h = 1e-3
    curvature = (kl(u,u+h*v)+kl(u,u-h*v))/h^2
    @assert abs(curvature-sum(v.^2 ./ u)) < 1e-7
    product_mean(m) = sum(i*j*(1+i*m)*(1+j*m)/4 for i in (-1,1), j in (-1,1))
    for m0 in (0.,0.3,1.)
        same(product_mean(m0),m0^2)
    end
    fd(v->product_mean(v[1]),[0.0],[0.0])
    # At the probability boundary use the feasible one-sided quadratic difference.
    same((3product_mean(1.)-4product_mean(0.5)+product_mean(0.))/(2*0.5),2.)
end

function checks()
    vocab()
    passes()
    mlp_heads()
    narrow_head()
    width(1)
    width(2)
    contexts()
    blocked()
    blocked_stock()
    external_only()
    worlds()
    @printf("Largest absolute identity residual %.3g\n",maximum(residuals))
    @printf("Largest absolute gradient error %.3g over %d coordinate differences\n",
            maximum(fd_errors),length(fd_errors))
end
if abspath(PROGRAM_FILE) == @__FILE__
    checks()
end
