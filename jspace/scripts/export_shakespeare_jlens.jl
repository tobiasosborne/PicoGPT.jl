# Run from the repository root: julia --project -t auto jspace/scripts/export_shakespeare_jlens.jl
include(joinpath(@__DIR__, "../../scripts/common.jl"))
include(joinpath(@__DIR__, "../src/JLens.jl"))
using .JLens, Statistics, SHA

pack(x::AbstractArray{<:Number}) = ndims(x) > 1 ? tensor(x) : collect(x)
pack(x::AbstractArray) = pack.(x)
pack(x::NamedTuple) = (; (k => pack(v) for (k,v) in pairs(x))...)
pack(x::AbstractDict) = Dict(k => pack(v) for (k,v) in pairs(x))
pack(x) = x

function export_shakespeare()
    started = time(); BLAS.set_num_threads(1)
    checkpoint = artifact("shakespeare","checkpoints","latest.bin")
    ck = load_checkpoint(checkpoint); c = ck.c; θ = Float64.(ck.θ); p = carve(θ,c)
    text = tiny_shakespeare(); training,validation = shakespeare_split(text,ck.abc)
    B = 64; per_window = 32
    starts = round.(Int,range(1,length(validation)-c.T;length=B))
    positions = round.(Int,range(1,c.T;length=per_window))
    x,y = windows(validation,c.T,starts); S = 2c.L+1
    per_window_J = zeros(c.D,c.D,S,B)
    states = [zeros(c.D,B*per_window) for _ in 1:S]
    targets = zeros(Int,B*per_window); own_logits = zeros(c.V,B*per_window)
    maximum_column_norm = zeros(B,S)
    jacobian_started = time()
    Threads.@threads for b in 1:B
        tr = residual_trace(θ,x[:,b:b],c)
        a = averaged_jacobians(tr)
        per_window_J[:,:,:,b] = a.J
        columns = (b-1)*per_window+1:b*per_window
        for j in 1:S
            states[j][:,columns] = tr.states[j][:,positions]
            maximum_column_norm[b,j] = maximum(sqrt.(sum(abs2,tr.states[j];dims=1)))
        end
        targets[columns] = y[positions,b]
        own_logits[:,columns] = lens_logits(p,tr.states[end][:,positions])
        println("Jacobian window ",b,"/",B," complete at ",round(time()-started;digits=1)," s");flush(stdout)
    end
    J = dropdims(mean(per_window_J;dims=4);dims=4)
    factor = (c.T+1)/2
    @assert maximum(abs,J[:,:,end]-I) ≤ 1e-14
    jacobian_seconds = time()-jacobian_started
    frames = [lens_frame(p,J[:,:,j]) for j in 1:S]
    metrics = [(; j_lens=lens_metrics(p,states[j],J[:,:,j],own_logits,targets),
                   logit_lens=lens_metrics(p,states[j],nothing,own_logits,targets)) for j in 1:S]
    literal_metrics = [(; j_lens=lens_metrics(p,states[j],J[:,:,j]/factor,own_logits,targets),
                           logit_lens=metrics[j].logit_lens) for j in 1:S]
    model_probs = softmax(own_logits;dims=1)
    final_error = maximum(abs,lens_probs(p,states[end],J[:,:,end])-model_probs)
    @assert final_error ≤ 1e-9
    @assert frames[end]' == p.Wout
    literal_error = maximum(abs,lens_probs(p,states[end],J[:,:,end]/factor)-model_probs)
    j_logit_agreement = [mean([argmax(z) for z in eachcol(lens_logits(p,states[j],J[:,:,j]))] .==
                             [argmax(z) for z in eachcol(lens_logits(p,states[j]))]) for j in 1:S]
    space_started=time(); spaces = greedy_space_statistics(frames,states)
    space_seconds=time()-space_started
    prompt = "GREMIO:\nGood morrow, neighbour Baptista.\n\nBAPTISTA:\nGood morrow, neighbour Gremio.\n"
    tokens = reshape(encode(prompt,ck.abc),:,1); trace = residual_trace(θ,tokens,c)
    grid = prompt_grid(trace,J); swap_started=time()
    swap = prompt_swap(trace,frames[5]);swap_seconds=time()-swap_started
    structure = structure_statistics(p,c,frames)
    cka = [linear_cka(frames[i]',frames[j]') for i in 1:S,j in 1:S]
    result = (description = "Float64 J-lens analysis on held-out Tiny Shakespeare; exact all-pair Jacobian averages and sampled activation analyses.",
        model = (; config=(;c.V,c.D,c.H,c.L,c.T,c.F), parameter_count=nparams(c),
            source="artifacts/shakespeare/checkpoints/latest.bin",step=ck.step,
            source_sha256=bytes2hex(sha256(read(checkpoint)))),
        conventions = (;ids="1-based",tensors="column-major {shape,data}",
            jacobian="J[output coordinate,input coordinate,depth] = mean over windows and sources 1:T of SUM over causal targets K; (T+1)/2 times the literal uniform-pair average",
            jacobian_factor=factor,
            vectors="lens_vectors[character,coordinate,depth]; rows of Wout J",
            grid="[rank,position,depth,lens]; lens 1=J,2=logit; model_top1 covers every prompt position",
            nnomp="unit frame columns; positive residual correlation; exact active-set NNLS refit; stalled fits carry forward",
            occupancy="smallest k>=1 with J marginal improvement below mean random marginal improvement; null if none",
            self_future="omitted to reserve runtime for exact 64-window all-position J and six-frame NNOMP curves",
            lens="softmax(Wout LN(Jx;gamma_f,beta_f)+bout), epsilon=1e-5; J_L=I gives the exact model readout",
            literal_lens="lenses.literal uses J/((T+1)/2), the uniform causal-pair average, with fixed epsilon=1e-5"),
        vocab=string.(ck.abc),depth_metadata(c)...,
        sampling=(;split="validation; last 10 percent in corpus order",training_tokens=length(training),validation_tokens=length(validation),
            corpus_sha256=bytes2hex(sha256(text)),windows=B,window_starts=starts,window_length=c.T,
            jacobian_positions=B*c.T,causal_pairs=B*c.T*(c.T+1)÷2,
            activation_positions=B*per_window,activation_positions_per_window=positions,
            random_frame_draws=5,self_future_target_positions=0,
            reduction="Activation and lens metrics use 32 evenly spaced positions in each of all 64 windows; J uses every position."),
        seeds=(;greedy_random=937,structure=1229),
        jacobians=(;averaged=J,self=nothing,future=nothing,
            singular_values=hcat([svdvals(J[:,:,j]) for j in 1:S]...),
            readout_singular_values=hcat([svdvals(p.Wout*J[:,:,j]) for j in 1:S]...),
            per_window_frobenius=[norm(per_window_J[:,:,j,b]) for b in 1:B,j in 1:S],
            per_window_max_residual_column_norm=maximum_column_norm),
        lenses=(;vectors=permutedims(cat(frames...;dims=3),(2,1,3)),
            logit_vectors=cat([Matrix(p.Wout) for _ in 1:S]...;dims=3),
            cosine_gram=cat(cosine_gram.(frames)...;dims=3),metrics,j_logit_agreement,
            final_max_probability_error=final_error,
            literal=(;metrics=literal_metrics,final_max_probability_error=literal_error),
            prompt=(;text=prompt,tokens=vec(tokens),grid...)),
        j_space=spaces,interventions=(;swap),structure,cka,
        sanity=(;final_j_error=maximum(abs,J[:,:,end]-I), final_lens_error=final_error,
            final_literal_lens_error=literal_error),
        timings=(;jacobian_seconds,space_seconds,swap_seconds),wall_seconds=time()-started)
    path=joinpath(@__DIR__,"../artifacts/shakespeare-jlens.json")
    writejson(path,pack(result);sig=5)
    @printf("Shakespeare export: %.3f seconds; %d bytes\n",result.wall_seconds,filesize(path))
    @printf("Stages: Jacobian %.3f; NNOMP %.3f; swap %.3f seconds\n",jacobian_seconds,space_seconds,swap_seconds)
    filesize(path)<6*1024^2 || @warn "Shakespeare exceeds approximate 6 MiB budget"
end
export_shakespeare()
