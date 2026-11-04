#!/usr/bin/env julia
# Standalone, stdlib-only companion to derivations.md.
# Run from any directory: julia /path/to/book/notes/verify_derivations.jl
using LinearAlgebra
using Random
using Statistics
using Test
using Printf

const RNG = MersenneTwister(20261104)
const STEP = eps(Float64)^(1 / 3)
const ERRORS = Pair{String,Float64}[]
const COUNTS = Int[]
rnd(dims...) = randn(RNG, dims...)
inner(x, y) = sum(x .* y)

function finite_gradient(f, x; step=STEP)
    work = copy(x)
    result = similar(x)
    for i in eachindex(x)
        h = step * max(1.0, abs(x[i]))
        work[i] = x[i] + h
        plus = f(work)
        work[i] = x[i] - h
        minus = f(work)
        work[i] = x[i]
        result[i] = (plus - minus) / (2h)
    end
    result
end

# Test every coordinate, rather than only one randomly chosen direction.
# The denominator floors the comparison scale at one: near-zero gradients
# are checked in absolute error, rather than through an unstable ratio.
function check_gradient(name, f, x, analytic; tol=2e-7, step=STEP)
    numeric = finite_gradient(f, x; step)
    err = maximum(abs.(numeric - analytic)) /
          max(1.0, maximum(abs.(numeric)), maximum(abs.(analytic)))
    push!(ERRORS, name => err)
    push!(COUNTS, length(x))
    @test isfinite(err) && err < tol
end

function check_pullback(name, op, args...; step=STEP)
    y, back = op(args...)
    ybar = rnd(size(y)...)
    bars = back(ybar)
    @assert length(bars) == length(args)
    for j in eachindex(args)
        objective = z -> inner(first(op(ntuple(k -> k == j ? z : args[k], length(args))...)), ybar)
        check_gradient("$name/input$j", objective, args[j], bars[j]; step)
    end
end

function affine(W, X, b)
    Y = W * X .+ b
    Y, Ybar -> (Ybar * X', W' * Ybar, vec(sum(Ybar; dims=2)))
end

function gather(E, ids)
    E[:, ids], Ybar -> begin
        Ebar = zeros(size(E))
        for n in eachindex(ids)
            Ebar[:, ids[n]] .+= Ybar[:, n]
        end
        (Ebar,)
    end
end

relu(X) = (max.(X, 0.0), Ybar -> (Ybar .* (X .> 0),))

function softmax(X; dims=1)
    ex = exp.(X .- maximum(X; dims))
    ex ./ sum(ex; dims)
end

function softmax_op(X; dims=1)
    p = softmax(X; dims)
    p, pbar -> (p .* (pbar .- sum(p .* pbar; dims)),)
end

function masked_softmax(S)
    # A fixed mask is a restriction to allowed coordinates, not an input
    # with respect to which one differentiates an infinite number.
    T = size(S, 1)
    mask = [j > i ? -Inf : 0.0 for i in 1:T, j in 1:T]
    softmax_op(S + mask; dims=2)
end

function ce(Z, targets)
    N = size(Z, 2)
    maxima = maximum(Z; dims=1)
    lse = maxima .+ log.(sum(exp.(Z .- maxima); dims=1))
    loss = (sum(lse) - sum(Z[targets[n], n] for n in 1:N)) / N
    p = softmax(Z)
    grad = copy(p)
    for n in 1:N
        grad[targets[n], n] -= 1
    end
    grad ./= N
    loss, lossbar -> (lossbar .* grad,)
end

function centre(X)
    X .- mean(X; dims=1), Ybar -> (Ybar .- mean(Ybar; dims=1),)
end

function sphere(u)
    D = length(u)
    radius = norm(u)
    unit = u / radius
    sqrt(D) .* unit, ybar -> (sqrt(D) / radius .* (ybar - unit * dot(unit, ybar)),)
end

function ln(X, gamma, beta; epsilon=1e-5)
    u = X .- mean(X; dims=1)
    sigma = sqrt.(mean(u .^ 2; dims=1) .+ epsilon)
    xhat = u ./ sigma
    Y = gamma .* xhat .+ beta
    Y, Ybar -> begin
        g = gamma .* Ybar
        Xbar = (g .- mean(g; dims=1) .- xhat .* mean(g .* xhat; dims=1)) ./ sigma
        (Xbar, vec(sum(Ybar .* xhat; dims=2)), vec(sum(Ybar; dims=2)))
    end
end

function head(Q, K, Vh)
    d = size(Q, 1)
    A, aback = masked_softmax(Q' * K / sqrt(d))
    O = Vh * A'
    O, Obar -> begin
        Vbar = Obar * A
        Abar = Obar' * Vh
        Sbar = only(aback(Abar))
        (K * Sbar' / sqrt(d), Q * Sbar / sqrt(d), Vbar)
    end
end

function multihead(Q, K, Vh; H=2, T=3)
    D, N = size(Q)
    @assert D % H == 0 && N % T == 0
    d = D ÷ H
    O = zeros(D, N)
    caches = []
    for b in 0:(N ÷ T - 1), h in 0:(H - 1)
        rows, cols = (h*d+1):(h+1)*d, (b*T+1):(b+1)*T
        Oh, back = head(Q[rows, cols], K[rows, cols], Vh[rows, cols])
        O[rows, cols] = Oh
        push!(caches, (rows, cols, back))
    end
    O, Obar -> begin
        Qbar, Kbar, Vbar = zeros(D, N), zeros(D, N), zeros(D, N)
        for (rows, cols, back) in caches
            qb, kb, vb = back(Obar[rows, cols])
            Qbar[rows, cols] = qb
            Kbar[rows, cols] = kb
            Vbar[rows, cols] = vb
        end
        (Qbar, Kbar, Vbar)
    end
end

function residual(W, X, b)
    f, back = affine(W, X, b)
    X + f, Ybar -> begin
        Wbar, branchbar, bbar = back(Ybar)
        (Wbar, Ybar + branchbar, bbar)
    end
end

parameter_count(V, D, F, L) = 2D*V + V + 2D + L*(4D^2 + 2D*F + 9D + F)

function layout(V, D, F, L)
    entries = Tuple{String,Tuple}[("E", (D, V))]
    for l in 1:L
        prefix = "block$l/"
        append!(entries, [(prefix*"gamma1", (D,)), (prefix*"beta1", (D,))])
        for name in ("q", "k", "v", "o")
            append!(entries, [(prefix*"W$name", (D,D)), (prefix*"b$name", (D,))])
        end
        append!(entries, [(prefix*"gamma2", (D,)), (prefix*"beta2", (D,)),
                          (prefix*"W1", (F,D)), (prefix*"b1", (F,)),
                          (prefix*"W2", (D,F)), (prefix*"b2", (D,))])
    end
    append!(entries, [("gammaf", (D,)), ("betaf", (D,)),
                      ("Wout", (V,D)), ("bout", (V,))])
    entries
end

# A tiny local reverse-mode tape composes only the hand-written pullbacks
# above. It is not a dependency on autodiff or on the concurrently rewritten src/.
mutable struct Node
    value::Array{Float64}
    bar::Array{Float64}
end
Node(x::Array{Float64}) = Node(x, zeros(size(x)))
function apply!(tape, op, args::Node...)
    y, back = op(map(a -> a.value, args)...)
    output = Node(y)
    push!(tape, (output, () -> begin
        for (arg, bar) in zip(args, back(output.bar))
            arg.bar .+= bar
        end
    end))
    output
end
add(X, Y) = (X + Y, Zbar -> (Zbar, Zbar))
add_fixed(X, P) = (X + P, Zbar -> (Zbar,))

function positions(D, values)
    @assert iseven(D)
    P = zeros(D, length(values))
    for (j, t) in enumerate(values), r in 0:(D ÷ 2 - 1)
        omega = 10000.0^(-2r / D)
        P[2r+1, j], P[2r+2, j] = sin(omega*t), cos(omega*t)
    end
    P
end

function rotations(D, k)
    R = zeros(D, D)
    for r in 0:(D ÷ 2 - 1)
        angle = 10000.0^(-2r / D) * k
        R[(2r+1):(2r+2), (2r+1):(2r+2)] =
            [cos(angle) sin(angle); -sin(angle) cos(angle)]
    end
    R
end

function model(theta, cfg, ids; continuous=nothing)
    (; V, D, F, L, H, T) = cfg
    tape = Tuple{Node,Function}[]
    pars = Dict{String,Node}()
    cursor = 1
    for (name, shape) in layout(V,D,F,L)
        count = prod(shape)
        pars[name] = Node(copy(reshape(theta[cursor:cursor+count-1], shape)))
        cursor += count
    end
    @assert cursor - 1 == length(theta)
    X = if continuous === nothing
        apply!(tape, E -> gather(E, ids), pars["E"])
    else
        Node(copy(continuous))
    end
    initial = X
    pos = repeat(positions(D, 0:T-1), 1, length(ids) ÷ T)
    X = apply!(tape, x -> add_fixed(x, pos), X)
    for l in 1:L
        p = name -> pars["block$l/$name"]
        U = apply!(tape, ln, X, p("gamma1"), p("beta1"))
        Q = apply!(tape, affine, p("Wq"), U, p("bq"))
        K = apply!(tape, affine, p("Wk"), U, p("bk"))
        Vh = apply!(tape, affine, p("Wv"), U, p("bv"))
        O = apply!(tape, (q,k,v) -> multihead(q,k,v; H, T), Q,K,Vh)
        branch = apply!(tape, affine, p("Wo"), O, p("bo"))
        X = apply!(tape, add, X, branch)
        U = apply!(tape, ln, X, p("gamma2"), p("beta2"))
        U = apply!(tape, affine, p("W1"), U, p("b1"))
        U = apply!(tape, relu, U)
        branch = apply!(tape, affine, p("W2"), U, p("b2"))
        X = apply!(tape, add, X, branch)
    end
    U = apply!(tape, ln, X, pars["gammaf"], pars["betaf"])
    Z = apply!(tape, affine, pars["Wout"], U, pars["bout"])
    Z.value, Zbar -> begin
        for p in values(pars)
            fill!(p.bar, 0)
        end
        for (output, _) in tape
            fill!(output.bar, 0)
        end
        fill!(initial.bar, 0)
        Z.bar .= Zbar
        for (_, back) in reverse(tape)
            back()
        end
        flat = reduce(vcat, [vec(pars[name].bar) for (name, _) in layout(V,D,F,L)])
        flat, initial.bar
    end
end

function init_theta(cfg)
    theta = Float64[]
    for (name, shape) in layout(cfg.V,cfg.D,cfg.F,cfg.L)
        a = 0.18 .* rnd(shape...)
        if occursin("gamma", name)
            a .+= 1
        elseif endswith(name, "/b1")
            # Keep end-to-end finite differences well away from ReLU kinks.
            a .= [isodd(j) ? 1.5 : -1.5 for j in eachindex(a)]
        end
        append!(theta, vec(a))
    end
    theta
end

@testset "Primitive and composed pullbacks" begin
    check_pullback("affine", affine, rnd(3,4), rnd(4,5), rnd(3))
    check_pullback("gather repeated ids", E -> gather(E, [2,1,2,4,2]), rnd(3,4))
    check_pullback("ReLU away from zero", relu, [-2.0 1.5; 0.7 -1.2])
    @test only(last(relu(zeros(1,1)))(ones(1,1))) == zeros(1,1)
    check_pullback("column softmax", softmax_op, rnd(4,3))
    check_pullback("row softmax", X -> softmax_op(X; dims=2), rnd(3,4))
    check_pullback("masked row softmax", masked_softmax, rnd(4,4))
    targets = [1,4,2]
    Z = rnd(4,3)
    loss, back = ce(Z, targets)
    check_gradient("mean cross entropy", z -> first(ce(z, targets)), Z, only(back(1.0)))
    @test all(abs.(sum(only(back(1.0)); dims=1)) .< 1e-14)
    check_pullback("centring projection", centre, rnd(4,3))
    check_pullback("sphere normalization", sphere, rnd(5))
    for epsilon in (0.0, 1e-5, 0.3)
        check_pullback("LayerNorm epsilon=$epsilon", (x,g,b) -> ln(x,g,b; epsilon),
                       rnd(5,3), rnd(5), rnd(5))
    end
    # Near u=0 the curvature scale is sqrt(epsilon); shrink the FD step
    # accordingly rather than relaxing the comparison tolerance.
    check_pullback("constant-column LayerNorm", ln, fill(2.0, 4,2), rnd(4), rnd(4);
                   step=STEP*sqrt(1e-5))
    check_pullback("D=1 LayerNorm", ln, rnd(1,3), rnd(1), rnd(1))
    check_pullback("attention head", head, rnd(3,4), rnd(3,4), rnd(3,4))
    check_pullback("multihead two sequences", multihead, rnd(4,6), rnd(4,6), rnd(4,6))
    check_pullback("residual fan-out", residual, rnd(4,4), rnd(4,3), rnd(4))
end

@testset "Geometric and probabilistic identities" begin
    D = 5
    Pi = Matrix{Float64}(I,D,D) - ones(D,D) / D
    @test Pi' ≈ Pi
    @test Pi * Pi ≈ Pi
    for epsilon in (0.0, 1e-5, 0.3)
        x, gamma, beta, ybar = rnd(D), rnd(D), rnd(D), rnd(D)
        u = Pi*x
        sigma = sqrt(dot(u,u)/D + epsilon)
        xhat = u / sigma
        J = Pi / sigma - (u * u') / (D * sigma^3)
        _, back = ln(reshape(x,D,1), gamma, beta; epsilon)
        compact = vec(first(back(reshape(ybar,D,1))))
        @test compact ≈ J' * (gamma .* ybar) atol=2e-13
        @test norm(J * ones(D)) < 1e-12
        @test J*u ≈ (epsilon / sigma^3) .* u atol=2e-13
        @test dot(xhat,xhat) ≈ D * (1 - epsilon/sigma^2)
        if epsilon == 0
            uhat = u / norm(u)
            sphere_J = sqrt(D)/norm(u) * (I - uhat*uhat')
            @test J ≈ sphere_J * Pi
        end
    end
    z = [-0.5,1.2,0.3,1.2]
    p = vec(softmax(reshape(z,:,1)))
    J = Diagonal(p) - p*p'
    _, back = softmax_op(reshape(z,:,1))
    v = rnd(4)
    @test vec(only(back(reshape(v,:,1)))) ≈ J'*v
    @test vec(softmax(reshape(z .+ 1000,:,1))) ≈ p
    @test vec(softmax(reshape(1e-8 .* z,:,1))) ≈ fill(0.25,4) atol=1e-8
    @test vec(softmax(reshape(1e4 .* z,:,1))) ≈ [0,0.5,0,0.5]
    beta = 1.7
    lsebeta = zz -> log(sum(exp.(beta .* zz))) / beta
    check_gradient("temperature log partition", lsebeta, z,
                   vec(softmax(reshape(beta .* z,:,1))))
    energy = -z
    freeenergy = ee -> -log(sum(exp.(-beta .* ee))) / beta
    check_gradient("free energy versus energies", freeenergy, energy,
                   vec(softmax(reshape(-beta .* energy,:,1))))
    q = [0.2,0.3,0.1,0.4]
    entropy = -sum(p .* log.(p))
    kl = sum(p .* log.(p ./ q))
    @test -sum(p .* log.(q)) ≈ entropy + kl
    @test kl >= 0
    @test first(ce(zeros(7,3), [1,2,3])) ≈ log(7)
    @test isfinite(first(ce([1000.0 -1000; -1000 1000], [1,2])))
    # Chain-factor normalisation for a binary length-three autoregression.
    first_token = [0.3,0.7]
    second = [0.2 0.8; 0.6 0.4] # rows = previous token
    third = [0.9 0.1; 0.35 0.65]
    @test sum(first_token[a]*second[a,b]*third[b,c]
              for a in 1:2,b in 1:2,c in 1:2) ≈ 1
    # Explicit fan-out and chain-rule test, including a shared parameter.
    x = rnd(4)
    W, U, b = rnd(4,4), rnd(3,4), rnd(4)
    f = xx -> sum(abs2, U*(W*xx + b) + U*xx)
    r = U*(W*x+b) + U*x
    check_gradient("chain rule and two branches", f, x, (W' + I)*U'*(2r))
end

@testset "Sinusoidal group action" begin
    for D in (2,8,128), k in (-11,0,7)
        t = [-3,0,1,23]
        R = rotations(D,k)
        @test positions(D,t .+ k) ≈ R*positions(D,t) atol=2e-14
        @test R'*R ≈ Matrix{Float64}(I,D,D) atol=2e-14
        @test rotations(D,k+3) ≈ R*rotations(D,3) atol=2e-14
    end
end

@testset "Layout and full exact model" begin
    for (V,D,F,L) in ((7,8,32,1),(65,128,512,4),(3,4,7,0),(9,6,13,2))
        explicit = sum(prod(shape) for (_,shape) in layout(V,D,F,L))
        @test explicit == parameter_count(V,D,F,L)
    end
    @test parameter_count(7,8,32,1) == 1007
    @test parameter_count(65,128,512,4) == 810049
    # Two residual blocks, two heads, two sequences, repeated embedding ids.
    cfg = (V=4,D=4,F=7,L=2,H=2,T=3)
    ids, targets = [1,2,1,3,1,4], [2,1,3,1,4,2]
    theta = init_theta(cfg)
    Z, back = model(theta,cfg,ids)
    loss, lossback = ce(Z,targets)
    theta_bar, _ = back(only(lossback(1.0)))
    check_gradient("whole model all parameters", t -> first(ce(first(model(t,cfg,ids)),targets)),
                   theta,theta_bar)
    # Key biases add only row-constant shifts to scores: an exact gauge.
    shifted_theta = copy(theta)
    cursor = 1
    for (name, shape) in layout(cfg.V,cfg.D,cfg.F,cfg.L)
        count = prod(shape)
        if endswith(name, "/bk")
            @test maximum(abs.(theta_bar[cursor:cursor+count-1])) < 1e-13
            shifted_theta[cursor:cursor+count-1] .+= rnd(count)
        end
        cursor += count
    end
    @test first(model(shifted_theta,cfg,ids)) ≈ Z atol=1e-13
    X = rnd(cfg.D, length(ids))
    Z, back = model(theta,cfg,ids; continuous=X)
    Zbar = rnd(size(Z)...)
    _, Xbar = back(Zbar)
    check_gradient("whole model continuous inputs", x -> inner(first(model(theta,cfg,ids; continuous=x)),Zbar),
                   X,Xbar)
    # Causality and sequence isolation at the full-model level.
    changed = copy(X)
    changed[:,3] .+= rnd(cfg.D)
    changed[:,4:6] .+= rnd(cfg.D,3)
    Zchanged = first(model(theta,cfg,ids; continuous=changed))
    @test Z[:,1:2] ≈ Zchanged[:,1:2] atol=1e-13
    # At a single head: mask, stochasticity, and an exact one-source first row.
    A, aback = masked_softmax(rnd(4,4))
    @test triu(A,1) == zeros(4,4)
    @test vec(sum(A; dims=2)) ≈ ones(4)
    @test A[1,:] == [1,0,0,0]
    Sbar = only(aback(rnd(4,4)))
    @test triu(Sbar,1) == zeros(4,4)
    @test Sbar[1,:] == zeros(4)
end

@testset "Optimisation and variance identities" begin
    # Exact finite ensemble: independent Rademacher entries have variance one.
    for d in (1,2,3,4)
        dots = Float64[]
        for bits in 0:(2^(2d)-1)
            q = [((bits >> j)&1) == 1 ? 1.0 : -1.0 for j in 0:d-1]
            k = [((bits >> (d+j))&1) == 1 ? 1.0 : -1.0 for j in 0:d-1]
            push!(dots, dot(q,k))
        end
        @test mean(dots) == 0
        @test mean(dots.^2) == d
        @test mean((dots ./ sqrt(d)).^2) ≈ 1
    end
    # Population expectation of per-example gradients equals gradient of mean.
    A, y, x = rnd(5,3), rnd(5), rnd(3)
    per_example = [(dot(A[i,:],x)-y[i]) .* A[i,:] for i in 1:5]
    full_grad = A'*(A*x-y)/5
    @test sum(per_example)/5 ≈ full_grad
    B = Matrix(Diagonal([0.3,1.0,2.0]))
    x, eta = rnd(3), 0.4
    g = B*x
    quadratic(xx) = dot(xx,B*xx)/2
    @test quadratic(x-eta*g) <= quadratic(x) - eta*(1-opnorm(B)*eta/2)*dot(g,g) + 1e-14
    # Exact fan-in variance by enumerating zero-mean independent weights.
    n, s, input = 3,0.4,[0.7,-1.1,2.0]
    outputs = [sum((isodd(bits >> j) ? s : -s)*input[j+1] for j in 0:n-1)
               for bits in 0:(2^n-1)]
    @test mean(outputs) ≈ 0 atol=1e-15
    @test mean(outputs.^2) ≈ s^2 * dot(input,input)
    @test mean(max.(outputs,0).^2) ≈ mean(outputs.^2)/2
    # Exact random-walk residual variance: K independent unit-variance steps.
    Ksteps = 6
    scaled_sums = [sum(isodd(bits >> j) ? 1.0 : -1.0 for j in 0:Ksteps-1)/sqrt(Ksteps)
                   for bits in 0:(2^Ksteps-1)]
    @test mean(scaled_sums.^2) ≈ 1
    for radius in (0.0,0.5,10.0)
        g = rnd(4)
        clipped = norm(g) > radius ? radius*g/norm(g) : copy(g)
        @test norm(clipped) <= radius + 1e-14
        if norm(g) > radius && radius > 0
            @test clipped/radius ≈ g/norm(g)
            # Projection variational inequality, checked at an interior point.
            v = rnd(4); v .*= radius/(2norm(v))
            @test dot(g-clipped,v-clipped) <= 1e-14
        end
    end
    g = [2.0,-3.0,0.2]
    diagonal_step = Diagonal(1 ./ sqrt.(g.^2))*g
    @test diagonal_step ≈ sign.(g)
    m, v, beta1, beta2 = zeros(3),zeros(3),0.9,0.999
    for k in 1:4
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*(g.^2)
        @test m/(1-beta1^k) ≈ g
        @test v/(1-beta2^k) ≈ g.^2
    end
    theta, preconditioner, lambda, eta = rnd(3), [0.5,2.0,4.0],0.1,0.01
    adamw = (1-eta*lambda)*theta - eta*(preconditioner.*g)
    coupled = theta - eta*(preconditioner.*(g+lambda*theta))
    @test adamw-coupled ≈ eta*lambda.*(preconditioner.-1).*theta
    # Central-difference truncation is quadratic before roundoff dominates.
    errors = [abs((exp(0.3+h)-exp(0.3-h))/(2h)-exp(0.3)) for h in (0.04,0.02,0.01)]
    @test errors[1]/errors[2] ≈ 4 rtol=1e-3
    @test errors[2]/errors[3] ≈ 4 rtol=1e-3
end

println()
println("Coordinate finite-difference comparisons (relative scale floored at 1):")
for (name, err) in ERRORS
    @printf("  %-43s %.3e\n", name, err)
end
@printf("Total: %d comparisons, %d coordinates; maximum scaled error %.3e\n",
        length(ERRORS), sum(COUNTS), maximum(last,ERRORS))
@printf("Float64 central-difference base step: %.8e\n", STEP)
println("Parameter counts: specimen = ",parameter_count(7,8,32,1),
        "; real model = ",parameter_count(65,128,512,4))
println("All verification testsets passed.")
