# Run from the repository root: julia --project -t auto jspace/scripts/export_specimen_jlens_training.jl
include(joinpath(@__DIR__, "../../scripts/common.jl"))
include(joinpath(@__DIR__, "../src/JLens.jl"))
using .JLens, SHA

"""
Record v_a(k) = J_j(θ_k)ᵀ Wout(θ_k)ᵀ e_a after k=0,…,200 updates.

Every J averages all training windows and source positions, summing causal
targets. Training calls the same step!, schedule and matrix-only decay as
scripts/export_specimen.jl, starting from the stored full-precision theta0.
"""
function export_specimen_training()
    started = time(); BLAS.set_num_threads(1)
    path = artifact("specimen.json"); source = read_json(path)
    s = specimen_data(path); (; c,x,y) = s
    h = source["train"]["hyper"]; steps = s.step
    @assert steps == 200
    θ = Float64.(source["theta0"])
    opt = AdamW(θ; β1=h["beta1"],β2=h["beta2"],ε=h["eps"],λ=decay(c,h["weight_decay"]))
    η(t) = warmup_cosine(t; η=h["eta"],ηmin=h["eta_min"],warmup=h["warmup"],steps)
    S = 2c.L+1
    vectors = zeros(c.V,c.D,S,steps+1); last_probs = zeros(c.V,S,steps+1)
    losses = zeros(steps+1); final_lens_error = 0.0
    for k in 0:steps
        k > 0 && step!(θ,opt,x,y,c; η=η(k),clip=h["clip"])
        trace = residual_trace(θ,x,c); J = averaged_jacobians(trace).J
        @assert J[:,:,end] == Matrix{Float64}(I,c.D,c.D)
        for j in 1:S
            vectors[:,:,j,k+1] = trace.p.Wout * J[:,:,j]
            # Column T is the last position of window 1, not the batch's end.
            last_probs[:,j,k+1] = vec(lens_probs(trace.p,trace.states[j][:,c.T:c.T],J[:,:,j]))
        end
        Z = lens_logits(trace.p,trace.states[end])
        losses[k+1] = first(crossentropy(Z,y))
        final_lens_error = max(final_lens_error,maximum(abs,last_probs[:,end,k+1]-softmax(Z[:,c.T])))
    end
    theta_error = maximum(abs,θ-s.θ)
    loss_error = maximum(abs,losses-Float64.(source["train"]["loss"]))
    @assert theta_error ≤ 1e-12
    @assert loss_error ≤ 1e-12
    @assert final_lens_error ≤ 1e-9
    result = (description="Every specimen J-lens covector and window-1 final-position readout during the canonical training run.",
        model=(;config=(;c.V,c.D,c.H,c.L,c.T,c.F),parameter_count=nparams(c),
            source="artifacts/specimen.json:theta0,train.hyper,batch",step=steps,
            source_sha256=bytes2hex(sha256(read(path)))),
        conventions=(;ids="1-based character ids; time index k+1 is after k updates, including k=0",
            tensors="column-major {shape,data}; Float64 computation, five significant digits on export",
            jacobian="J = mean over windows and sources of SUM over causal targets K; J_L=I",
            jacobian_factor=(c.T+1)/2,
            vectors="vectors[character,coordinate,depth,k+1] = Wout(theta_k)*J_depth(theta_k)",
            last_probs="last_probs[character,depth,k+1]; last input position T of training window 1",
            loss="loss[k+1] is the full-batch cross-entropy after k updates"),
        vocab=string.(s.abc),depth_metadata(c)...,steps=collect(0:steps),
        vectors=tensor(vectors),last_probs=tensor(last_probs),loss=losses,
        sampling=(;windows=size(x,2),jacobian_positions=length(x),
            causal_pairs=size(x,2)*c.T*(c.T+1)÷2,readout_window=1,readout_position=c.T),
        provenance=(;exporter="jspace/scripts/export_specimen_jlens_training.jl",
            recipe="scripts/export_specimen.jl",hyper=h,theta0="stored Float64 theta0; no random reinitialisation",
            julia_version=string(VERSION),julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads()),
        sanity=(;theta_final_max_error=theta_error,loss_trajectory_max_error=loss_error,
            final_depth_max_probability_error=final_lens_error,assertion_tolerance=1e-12),
        wall_seconds=time()-started)
    out = joinpath(@__DIR__,"../artifacts/specimen-jlens-training.json")
    writejson(out,result;sig=5)
    @assert filesize(out) < 1_000_000
    @printf("Specimen training export: %.3f seconds; %d bytes; %d snapshots\n",result.wall_seconds,filesize(out),steps+1)
    @printf("Final theta max error: %.12g; loss trajectory max error: %.12g\n",theta_error,loss_error)
end
export_specimen_training()
