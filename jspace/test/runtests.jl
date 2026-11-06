using Test, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "../../scripts/common.jl"))
include(joinpath(@__DIR__, "../src/JLens.jl"))
using .JLens
BLAS.set_num_threads(1)
verification=Dict{String,Any}("source"=>"jspace/test/runtests.jl", "finite_difference_step"=>1e-5, "nnls_random_trials"=>30)
s = specimen_data(artifact("specimen.json"))
tr = residual_trace(s.θ,s.x[:,1:1],s.c)

@testset "Residual semantics and hand-written Jacobians" begin
    logits,_ = gpt(s.θ,s.x[:,1:1],s.c)
    @test lens_logits(tr.p,tr.states[end]) ≈ logits atol=1e-12
    @test continue_from(tr,tr.states[2],2) ≈ tr.states[end] atol=1e-12
    largest_error = 0.0
    for j in 1:3, (target,source) in ((1,1),(5,2),(8,8))
        K = jacobian_block(tr,j,target,source); finite = zeros(s.c.D,s.c.D)
        h = 1e-5
        for q in 1:s.c.D
            plus=copy(tr.states[j]);minus=copy(plus)
            plus[q,source]+=h;minus[q,source]-=h
            finite[:,q]=(continue_from(tr,plus,j)[:,target]-continue_from(tr,minus,j)[:,target])/(2h)
        end
        error=norm(K-finite)/max(norm(K),1e-12)
        largest_error=max(largest_error,error)
        @test error < 1e-6
    end
    verification["finite_difference_max_relative_error"]=largest_error
    verification["causality_blocks_checked"]=3*s.c.T*(s.c.T-1)÷2
    println("Maximum finite-difference relative block error: ",largest_error)
    for j in 1:3, target in 1:s.c.T, source in target+1:s.c.T
        @test iszero(jacobian_block(tr,j,target,source))
    end
    a=averaged_jacobians(tr;self_targets=collect(1:s.c.T))
    @test a.J[:,:,end] == Matrix{Float64}(I,s.c.D,s.c.D)
    @test a.self[:,:,end] ≈ I atol=1e-14
    @test norm(a.future[:,:,end]) < 1e-14
    for j in 1:3
        explicit=sum(jacobian_block(tr,j,target,source) for target in 1:s.c.T for source in 1:target)/s.c.T
        @test a.J[:,:,j] ≈ explicit atol=1e-11
        strict=sum(jacobian_block(tr,j,target,source) for target in 2:s.c.T for source in 1:target-1)/s.c.T
        @test a.future[:,:,j] ≈ strict atol=1e-11
    end
    @test a.J ≈ a.self + a.future atol=1e-14
    batch=averaged_jacobians(residual_trace(s.θ,s.x,s.c);self_targets=collect(1:length(s.x)))
    @test batch.J ≈ batch.self + batch.future atol=1e-14
    @test batch.J[:,:,end] == Matrix{Float64}(I,s.c.D,s.c.D)
end

@testset "Final-depth lens and LayerNorm epsilon" begin
    full=residual_trace(s.θ,s.x,s.c)
    a=averaged_jacobians(full); X=full.states[end]; scale=2/(s.c.T+1)
    model=lens_probs(full.p,X); readout=lens_probs(full.p,X,a.J[:,:,end])
    @test maximum(abs,readout-model) ≤ 1e-9
    @test lens_frame(full.p,a.J[:,:,end])' == full.p.Wout
    literal=lens_probs(full.p,X,scale*a.J[:,:,end])
    # LN_epsilon(scale*x) = LN_(epsilon/scale²)(x), not LN_epsilon(x).
    Y=scale*X; centred=Y.-mean(Y;dims=1)
    F=full.p.γf.*centred./sqrt.(mean(centred.^2;dims=1).+1e-5*scale^2).+full.p.βf
    adjusted=softmax(full.p.Wout*F.+full.p.bout;dims=1)
    @test maximum(abs,adjusted-model)<1e-9
    literal_error=maximum(abs,literal-model)
    @test literal_error ≈ 1.72324e-5 rtol=1e-5
    @test literal_error > 1e-9
    println("Scale-free final lens probability discrepancy: ",maximum(abs,readout-model))
    println("Literal paper-weighted final lens epsilon discrepancy (80 columns): ",literal_error)
    verification["final_lens_max_probability_error"]=maximum(abs,readout-model)
    verification["literal_final_lens_max_probability_error"]=literal_error
end

@testset "NNLS versus exhaustive active supports" begin
    rng=Xoshiro(17)
    for trial in 1:30
        A=randn(rng,6,4);x=randn(rng,6)
        best=sum(abs2,x)
        for mask in 1:15
            selected=findall(i->(mask>>(i-1))&1==1,1:4)
            z=A[:,selected]\x
            if all(z.≥0);best=min(best,sum(abs2,x-A[:,selected]*z));end
        end
        c=nnls(A,x);w=A'*(x-A*c)
        @test minimum(c)≥0
        @test sum(abs2,x-A*c) ≈ best atol=1e-9
        @test maximum(w)<1e-8
        @test maximum(abs,c.*w)<1e-8
    end
    @test nnls([1.0 1;0 0],[1.0,0]) ≈ [1,0]
    @test nnls(Matrix{Float64}(I,3,3),[-1.0,2,0]) ≈ [0,2,0]
    A=unit_columns(randn(rng,8,7));x=randn(rng,8)
    distances=[norm(x-last(exact_decomposition(A,x,k))) for k in 1:7]
    @test all(diff(distances).≤1e-10)
    curve,_=nnomp(A,x)
    @test all(diff(curve).≤1e-10)
    @test curve[end]≈norm(x-A*nnls(A,x))^2/norm(x)^2 atol=1e-8
end

@testset "Gram pursuit agrees with direct NNOMP" begin
    rng=Xoshiro(177)
    A=unit_columns(randn(rng,12,9));X=randn(rng,12,20)
    curves,counts=JLens.pursuit_curves(A,X)
    for n in axes(X,2)
        direct,nonzero=nnomp(A,X[:,n])
        @test curves[:,n] ≈ direct atol=1e-9
        @test counts[:,n] == nonzero
    end
end

@testset "Two-block continuation" begin
    c=Config(V=7,D=8,H=2,L=2,T=8,F=16)
    θ=init(c;rng=Xoshiro(71));tokens=s.x[:,1:1]
    trace=residual_trace(θ,tokens,c)
    Z,_=gpt(θ,tokens,c)
    @test lens_logits(trace.p,trace.states[end]) ≈ Z atol=1e-12
    for j in 1:5
        @test continue_from(trace,trace.states[j],j) ≈ trace.states[end] atol=1e-12
    end
end

@testset "Pseudoinverse coordinate swap" begin
    rng=Xoshiro(19);V=randn(rng,8,2);x=randn(rng,8)
    edited,c,gram=swap_coordinates(x,V)
    @test gram ≈ V'V
    @test pinv(V)*edited ≈ reverse(c) atol=1e-12
    projector=I-V*pinv(V)
    @test projector*edited ≈ projector*x atol=1e-12
    @test first(swap_coordinates(edited,V)) ≈ x atol=1e-12
end

@testset "Linear CKA" begin
    rng=Xoshiro(23);X=randn(rng,7,8);Q=Matrix(qr(randn(rng,8,8)).Q)
    @test linear_cka(X,X) ≈ 1 atol=1e-12
    @test linear_cka(X,X*Q) ≈ 1 atol=1e-12
    @test linear_cka(X,3X.+randn(rng,1,8)) ≈ 1 atol=1e-12
end

@testset "Export schema and numeric invariants (when artifacts exist)" begin
    unpack(a)=reshape(Float64.(a["data"]),Int.(a["shape"])...)
    for name in ("specimen","shakespeare")
        path=joinpath(@__DIR__,"../artifacts/$(name)-jlens.json")
        isfile(path) || continue
        data=read_json(path); J=unpack(data["jacobians"]["averaged"])
        config=data["model"]["config"];D=config["D"];T=config["T"];V=config["V"]
        @test size(J)==(D,D,2config["L"]+1)
        @test J[:,:,end]≈I atol=1e-12
        @test data["sanity"]["final_j_error"]≤1e-14
        @test data["sanity"]["final_lens_error"]≤1e-9
        @test data["conventions"]["jacobian_factor"]==(T+1)/2
        cka=unpack(data["cka"])
        @test cka≈cka' atol=1e-6
        @test diag(cka)≈ones(size(cka,1)) atol=1e-6
        frames=unpack(data["lenses"]["vectors"])
        @test size(frames)==(V,D,size(J,3))
        @test frames[:,:,end]≈unpack(data["lenses"]["logit_vectors"])[:,:,end] atol=1e-12
        literal=data["lenses"]["literal"]
        @test literal["final_max_probability_error"]≈data["sanity"]["final_literal_lens_error"]
        if name=="specimen"
            @test J≈unpack(data["jacobians"]["self"])+unpack(data["jacobians"]["future"]) rtol=2e-6
            blocks=unpack(data["jacobians"]["blocks"])
            @test blocks[:,:,end,end,end]≈((T+1)/2)*I atol=1e-12
            window_average=dropdims(sum(blocks[:,:,:,:,2];dims=(3,4));dims=(3,4))/(T*(T+1)/2)
            @test window_average≈unpack(data["jacobians"]["per_window_attention"])[:,:,1] rtol=3e-6
            @test maximum(abs,unpack(data["lenses"]["j_probs"])[:,:,end]-unpack(data["lenses"]["model_probs"]))≤1e-12
            @test size(unpack(literal["j_probs"]))==(V,80,size(J,3))
            @test filesize(path)<3*1024^2
            table=unpack(data["interventions"]["probability_table"])
            @test minimum(table)≥0
            @test maximum(abs,vec(sum(table;dims=1)).-1)<3e-6
            for depth in data["interventions"]["by_depth"]
                for kind in ("steer","steer_later_window1","ablate","swap")
                    ids=depth[kind]["data"]
                    @test minimum(ids)≥0
                    @test maximum(ids)≤size(table,2)
                end
            end
            for fit in data["j_space"]
                distance=unpack(fit["distance"])
                @test maximum(diff(distance;dims=1))≤2e-5
                @test minimum(fit["coefficients"]["data"])≥0
            end
        else
            @test length(literal["metrics"])==size(J,3)
            @test literal["final_max_probability_error"]≈0.18907 rtol=1e-4
            @test data["lenses"]["metrics"][end]["j_lens"]["model_agreement"]==1
            @test filesize(path)<6*1024^2
            @test data["sampling"]["windows"]≥64
            @test data["sampling"]["activation_positions"]≥2000
            for fit in data["j_space"]
                @test maximum(diff(Float64.(fit["residual_fraction"])))≤1e-5
                @test maximum(diff(Float64.(fit["random_residual_fraction"])))≤1e-5
            end
            # Check the fast solver on trained-model directions as well as
            # isotropic test matrices; their conditioning can be very different.
            ck=load_checkpoint(artifact("shakespeare","checkpoints","latest.bin"))
            _,validation=shakespeare_split(tiny_shakespeare(),ck.abc)
            tokens,_=windows(validation,ck.c.T,[data["sampling"]["window_starts"][1]])
            trace=residual_trace(Float64.(ck.θ),tokens,ck.c)
            largest_pursuit_error=0.0
            for j in axes(frames,3)
                A=unit_columns(Matrix(frames[:,:,j]'))
                X=trace.states[j][:,[1,64,128,256]]
                curves,_=JLens.pursuit_curves(A,X)
                for n in axes(X,2)
                    direct,_=nnomp(A,X[:,n])
                    error=maximum(abs,curves[:,n]-direct)
                    largest_pursuit_error=max(largest_pursuit_error,error)
                    @test error<1e-6
                end
            end
            println("Maximum trained-frame Gram/direct NNOMP discrepancy: ",largest_pursuit_error)
            verification["trained_frame_pursuit_max_error"]=largest_pursuit_error
        end
    end
end

@testset "Specimen training export endpoints and trajectory" begin
    path=joinpath(@__DIR__,"../artifacts/specimen-jlens-training.json")
    @test isfile(path)
    if isfile(path)
        data=read_json(path)
        unpack(a)=reshape(Float64.(a["data"]),Int.(a["shape"])...)
        vectors=unpack(data["vectors"]);probs=unpack(data["last_probs"])
        @test filesize(path)<1_000_000
        @test data["steps"]==collect(0:200)
        @test data["depth"]==[0,0.5,1]
        @test size(vectors)==(7,8,3,201)
        @test size(probs)==(7,3,201)
        @test length(data["loss"])==201
        @test minimum(probs)≥0
        @test maximum(abs,sum(probs;dims=1).-1)<3e-5
        @test data["sanity"]["theta_final_max_error"]≤1e-12
        @test data["sanity"]["loss_trajectory_max_error"]≤1e-12
        @test data["sanity"]["final_depth_max_probability_error"]≤1e-9
        source=read_json(artifact("specimen.json"))
        @test Float64.(data["loss"])≈Float64.(source["train"]["loss"]) rtol=1e-4
        for (i,θ) in ((1,Float64.(source["theta0"])),(201,s.θ))
            trace=residual_trace(θ,s.x,s.c);J=averaged_jacobians(trace).J
            for j in 1:3
                @test vectors[:,:,j,i]≈trace.p.Wout*J[:,:,j] rtol=1e-4
                expected=lens_probs(trace.p,trace.states[j][:,s.c.T:s.c.T],J[:,:,j])[:,1]
                @test probs[:,j,i]≈expected rtol=1e-4
            end
        end
    end
end

writejson(joinpath(@__DIR__,"../artifacts/verification.json"),verification;sig=12)
