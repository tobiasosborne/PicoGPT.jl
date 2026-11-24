# From the repository root: julia -t1 --project=. grow/notes/check_dmrg.jl
using LinearAlgebra, Statistics, Test, Random
import PicoGPT as PG
BLAS.set_num_threads(1)

@testset "Discarded weight and Fisher metric" begin
    # Columns are equally weighted centered samples.
    X = [1. 0 -1 0; 1 2 -1 -2; 0 1 0 -1]
    M = [2. 1 0; 1 2 0; 0 0 3]
    R = sqrt(Symmetric(M))
    A = R * X / sqrt(size(X, 2))
    U, s, _ = svd(A)
    Q = R \ (U[:, 1:1] * U[:, 1:1]' * R)
    @test Q * Q ≈ Q
    @test Q' * M ≈ M * Q
    @test norm(R * (I-Q) * X)^2 / size(X, 2) ≈ sum(abs2, s[2:end])
    @test norm(Q-Q') > 1e-3
    coord = Diagonal([1., 0])
    mixed = fill(0.5, 2, 2)
    costs(q) = (sum(abs2, diag(I-q))/2, (q[1,2]^2 + q[2,1]^2)/2)
    @test costs(coord) == (0.5, 0.)
    @test costs(mixed) == (0.25, 0.25)

    x = [1.1, -0.9, 0.6, 0.2]
    v = [0.8, 0.7, -0.3, 0.2]
    g = [1.3, 0.8, 0.9, 1.1]
    w = [1. 0 2 1; 2 1 0 1; 0 2 1 1]
    ln(x) = vec(first(PG.layernorm(reshape(x, :, 1), g, zero(g))))
    z(x) = w * ln(x)
    c = x .- mean(x)
    sig = sqrt(mean(abs2, c) + 1e-5)
    cent = I - ones(4, 4)/4
    J = w * Diagonal(g) * (cent/sig - c*c'/(4sig^3))
    h = 1e-4
    @test (z(x+h*v)-z(x-h*v))/(2h) ≈ J*v rtol=1e-8
    @test norm(J*ones(4)) < 2e-15
    @test J*c ≈ w*Diagonal(g)*c*(1e-5/sig^3) atol=3e-15
    p = PG.softmax(z(x))
    kl(t) = sum(p .* log.(p ./ PG.softmax(z(x+t*v))))
    @test (kl(h)+kl(-h))/h^2 ≈ dot(J*v, (Diagonal(p)-p*p')*J*v) rtol=2e-7
end

@testset "Information per refinement and exact readout rank" begin
    for (coarse, fine) in [(2,3), (3,6)]
        # Uniform fine histories/futures; prediction from the coarse prefix.
        mass = 2.0^(-fine)
        coarse_mass = 2.0^(-coarse)
        gain = sum(mass * log(coarse_mass/mass) for _ in 1:2^fine)
        @test gain ≈ (fine-coarse)*log(2)
    end
    tau = 0.125
    for K in [4, 8, 16]
        c = PG.Config(V=K, D=K, H=1, T=1, L=0)
        theta = zeros(PG.nparams(c))
        par = PG.carve(theta, c)
        unit = Matrix{Float64}(I, K, K)
        par.E .= unit .- PG.positions(K, 1)
        par.γf .= 1
        sigma = sqrt((K-1)/K^2 + 1e-5)
        par.Wout .= (tau*sigma)*unit
        Z, _ = PG.gpt(theta, reshape(1:K, 1, :), c)
        probs = PG.softmax(Z; dims=1)
        @test probs ≈ PG.softmax(tau*unit; dims=1)
        B = I - ones(K, K)/K
        @test B*log.(probs)*B ≈ tau*B
        @test rank(B*Z*B) == K-1
        @test 0 < sum(probs .* log.(K .* probs))/K <= tau
    end
end

@testset "Exact product and minimal coupled split" begin
    w = [3/8, 1/8, 1/2]
    bern(a) = 1/(1+exp(-a))
    for a in [-2., 0., 1.], u in [-1., 0., 0.7]
        q = [bern(a)*bern(u), bern(a)*(1-bern(u)), 1-bern(a)]
        coarse = -dot([1/2, 1/2], log.([bern(a), 1-bern(a)]))
        fine = -dot(w[1:2], log.([bern(u), 1-bern(u)]))
        @test sum(q[1:2]) ≈ bern(a)
        @test -dot(w, log.(q)) ≈ coarse+fine
    end
    child(a,u) = PG.softmax([a+u-log(2), a-u-log(2), 0])
    for a in [-2., 0., 1.]
        @test sum(child(a,0)[1:2]) ≈ bern(a)
    end
    joint_u = log(3)/2
    @test child(-log(cosh(joint_u)), joint_u) ≈ w
    loss(u) = -dot(w, log.(child(0,u)))
    h = 1e-5
    frozen_u = log(5/3)
    for u in [-1., 0., frozen_u, 2.]
        @test (loss(u+h)-loss(u-h))/(2h) ≈ tanh(u/2)-1/4 atol=3e-11
    end
    @test child(0, frozen_u) ≈ [25, 9, 30]/64
    gap = loss(frozen_u) + dot(w, log.(w))
    @test gap ≈ sum(w .* log.(w ./ child(0, frozen_u))) atol=4e-16
    @test gap > 0
    println("Frozen excess (nats): ", gap)
end

@testset "Whole-lineage work for a separable dense baseline" begin
    q = [0.9, 0.75]
    eta = 4log(3)
    ce(z,r) = log1p(exp(z)) - r*z
    slope(z,r) = 1/(1+exp(-z)) - r
    step(z,r) = z - eta*slope(z,r)
    h = 1e-5
    for r in q, z in [0., 0.7, 1.4]
        @test (ce(z+h,r)-ce(z-h,r))/(2h) ≈ slope(z,r) atol=2e-11
    end
    first_step = step.(zeros(2), q)
    joint = step.(first_step, q)
    staged = [step(first_step[1], q[1]), step(0., q[2])]
    @test first_step[2] ≈ log(3)
    @test first_step[1] < log(9)
    @test staged ≈ joint
    @test sum(ce.(staged,q)) < sum(ce.(first_step,q))
end

@testset "Global stream change with fixed positional code" begin
    rng = MersenneTwister(17)
    c = PG.Config(V=3, D=4, H=2, T=2, F=5, L=1)
    theta = randn(rng, PG.nparams(c))
    p = PG.carve(theta,c)
    newtheta = copy(theta)
    s = PG.carve(newtheta,c)
    P = PG.positions(c.D,c.T)
    delta = P[:,2]-P[:,1]
    direction = nullspace(hcat(ones(c.D),delta)')[:,1]
    O = I - 2direction*direction'
    A = O + ones(c.D)*direction'
    @test O'*O ≈ I
    @test A*delta ≈ delta
    @test norm(A'*A-I) > 1
    s.E .= A*p.E .+ (A-I)*P[:,1]
    function change_reader!(newW,newb,W,b,gamma,beta)
        newW .= W*Diagonal(gamma)*O'
        newb .= b+W*beta
    end
    for (orig, new) in zip(p.blocks,s.blocks)
        for (weight,bias) in [(:Wq,:bq),(:Wk,:bk),(:Wv,:bv)]
            change_reader!(new[weight], new[bias], orig[weight], orig[bias], orig.γ1, orig.β1)
        end
        change_reader!(new.W1,new.b1,orig.W1,orig.b1,orig.γ2,orig.β2)
        new.γ1 .= new.γ2 .= 1
        new.β1 .= new.β2 .= 0
        new.Wo .= A*orig.Wo
        new.bo .= A*orig.bo
        new.W2 .= A*orig.W2
        new.b2 .= A*orig.b2
    end
    change_reader!(s.Wout,s.bout,p.Wout,p.bout,p.γf,p.βf)
    s.γf .= 1
    s.βf .= 0
    tokens = [1 2 3; 2 3 1]
    origtrace, newtrace = [], []
    Z, _ = PG.gpt(theta, tokens, c; trace=origtrace)
    Znew, _ = PG.gpt(newtheta, tokens, c; trace=newtrace)
    @test newtrace[1].X0 ≈ A*origtrace[1].X0
    @test newtrace[2].Xmid ≈ A*origtrace[2].Xmid
    @test newtrace[2].Xout ≈ A*origtrace[2].Xout
    @test Znew ≈ Z
    s.E .+= randn(rng, 1, c.V)
    s.blocks[1].Wo .+= randn(rng, 1, c.D)
    @test first(PG.gpt(newtheta, tokens, c)) ≈ Z
end

@testset "Joint moment, oblique normal equation and coordinate invariance" begin
    us = [[1., -2.], [2., 1.], [-1., 1.]]
    gs = [[2. 1; 1 3], [1. 0; 0 2], [3. -1; -1 2]]
    objective(q, xs, metrics) = mean(dot((I-q)*x, g*((I-q)*x)) for (x,g) in zip(xs,metrics))
    # Plane e1. K has one coefficient and its normal equation is scalar.
    k = sum(g[1,2]*x[2]^2 for (x,g) in zip(us,gs))/sum(g[1,1]*x[2]^2 for (x,g) in zip(us,gs))
    proj(a) = [1. a; 0. 0.]
    q = proj(k)
    h = 1e-5
    @test abs((objective(proj(k+h),us,gs)-objective(proj(k-h),us,gs))/(2h)) < 1e-9
    K = mean(kron(x*x', g) for (x,g) in zip(us,gs))
    residual = vec(I-q)
    @test dot(residual,K*residual) ≈ objective(q,us,gs)
    a = [2. 1; 0. 3.]
    @test objective(a*q/a, [a*x for x in us], [a'\g/a for g in gs]) ≈ objective(q,us,gs)
end
