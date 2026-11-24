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
