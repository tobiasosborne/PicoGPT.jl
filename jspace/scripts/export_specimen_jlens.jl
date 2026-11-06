# Run from the repository root: julia --project -t auto jspace/scripts/export_specimen_jlens.jl
include(joinpath(@__DIR__, "../../scripts/common.jl"))
include(joinpath(@__DIR__, "../src/JLens.jl"))
using .JLens, Statistics, SHA

# Reuse common.jl's tensor convention and JSON writer for every numerical array.
pack(x::AbstractArray{<:Number}) = ndims(x) > 1 ? tensor(x) : collect(x)
pack(x::AbstractArray) = pack.(x)
pack(x::NamedTuple) = (; (k => pack(v) for (k,v) in pairs(x))...)
pack(x::AbstractDict) = Dict(k => pack(v) for (k,v) in pairs(x))
pack(x) = x

function export_specimen()
    started = time(); BLAS.set_num_threads(1)
    s = specimen_data(artifact("specimen.json")); (;c,θ,x,y) = s
    p = carve(θ,c); B = size(x,2); N = length(x)
    traces = [residual_trace(θ, x[:,b:b], c) for b in 1:B]
    averages = [averaged_jacobians(tr; self_targets = collect(1:c.T)) for tr in traces]
    J = mean([a.J for a in averages]); self = mean([a.self for a in averages])
    future = mean([a.future for a in averages]); S = size(J,3)
    factor = (c.T+1)/2
    @assert J ≈ self + future
    @assert maximum(abs,J[:,:,end]-I) ≤ 1e-14
    frames = [lens_frame(p,J[:,:,j]) for j in 1:S]
    states = [hcat([tr.states[j] for tr in traces]...) for j in 1:S]
    blocks = zeros(c.D,c.D,c.T,c.T,S)
    for j in 1:S, target in 1:c.T, source in 1:target
        blocks[:,:,target,source,j] = factor * jacobian_block(traces[1],j,target,source)
    end
    model = lens_probs(p,states[end])
    j_probs = cat([lens_probs(p,states[j],J[:,:,j]) for j in 1:S]...; dims=3)
    logit_probs = cat([lens_probs(p,X) for X in states]...;dims=3)
    final_error = maximum(abs,j_probs[:,:,end]-model)
    @assert final_error ≤ 1e-9
    @assert frames[end]' == p.Wout
    literal_probs = cat([lens_probs(p,states[j],J[:,:,j]/factor) for j in 1:S]...;dims=3)
    literal_error = maximum(abs,literal_probs[:,:,end]-model)
    spaces = exact_space_statistics(frames,states)
    interventions = specimen_interventions(traces,frames,y)
    structure = structure_statistics(p,c,frames)
    cka = [linear_cka(frames[i]',frames[j]') for i in 1:S,j in 1:S]
    result = (description = "Exact J-lens and J-space analysis of the trained PicoGPT specimen.",
        model = (; config = (;c.V,c.D,c.H,c.L,c.T,c.F), parameter_count = nparams(c),
            source = "artifacts/specimen.json:train.theta_final", step = s.step,
            source_sha256 = bytes2hex(sha256(read(artifact("specimen.json"))))),
        conventions = (; ids = "1-based; 0 only pads subsets and out-of-window readout ids",
            tensors = "column-major {shape,data}; token n=t+(b-1)T",
            jacobian = "J[output coordinate,input coordinate,depth] = mean over windows and sources of SUM over causal targets K; (T+1)/2 times the literal uniform-pair average; averaged=self+future",
            jacobian_factor = factor,
            blocks = "[output coordinate,input coordinate,target,source,depth] = ((T+1)/2)*K; window 1; divide by jacobian_factor for the raw derivative; anti-causal zeros are padding",
            vectors = "lens_vectors[character,coordinate,depth]; rows of Wout J",
            j_space = "selected[slot,k,column]; coefficients match selected, zero-padded after slot k",
            interventions = "readout ids index probability_table[:,id]; steer axes [column,character,alpha]; steer_later_window1 axes [offset,position,character,alpha]; ablate [offset,column,character]; swap [offset,column,pair]; offset=1 is edited position; 0 pads beyond window",
            lens = "softmax(Wout LN(Jx;gamma_f,beta_f)+bout), epsilon=1e-5; J_L=I gives the exact model readout",
            literal_lens = "lenses.literal uses J/((T+1)/2), the uniform causal-pair average, with the same fixed epsilon=1e-5",
            concept_vectors = "[coordinate,character,component] before equal-norm rescaling"),
        vocab = string.(s.abc), depth_metadata(c)...,
        sampling = (; windows=B,positions=N,causal_pairs=B*c.T*(c.T+1)÷2,
            block_window=1, self_positions=N, random_frame_draws=20,
            steer_later_windows=[1], reduction="Steering covers all 80 edited positions; later steering readouts use window 1 to meet the 3 MiB budget. Ablations and swaps retain all later positions."),
        seeds = (; exact_random=431,structure=1229),
        jacobians = (; averaged=J,self,future,blocks,
            per_window_attention = cat([a.J[:,:,2] for a in averages]...;dims=3),
            singular_values = hcat([svdvals(J[:,:,j]) for j in 1:S]...),
            readout_singular_values = hcat([svdvals(p.Wout*J[:,:,j]) for j in 1:S]...)),
        lenses = (; vectors = permutedims(cat(frames...;dims=3),(2,1,3)),
            logit_vectors = cat([Matrix(p.Wout) for _ in 1:S]...;dims=3),
            cosine_gram = cat(cosine_gram.(frames)...;dims=3),
            j_probs,logit_probs,model_probs=model,
            final_max_probability_error=final_error,
            literal=(;j_probs=literal_probs,final_max_probability_error=literal_error),
            j_logit_agreement=[mean([argmax(v) for v in eachcol(j_probs[:,:,j])] .==
                                   [argmax(v) for v in eachcol(logit_probs[:,:,j])]) for j in 1:S]),
        j_space = spaces, interventions, structure, cka,
        sanity = (; pullback_tests = isfile(joinpath(@__DIR__,"../artifacts/verification.json")) ?
                read_json(joinpath(@__DIR__,"../artifacts/verification.json")) : nothing,
            final_j_error=maximum(abs,J[:,:,end]-I),
            final_self_error=maximum(abs,self[:,:,end]-I),final_future_norm=norm(future[:,:,end]),
            self_future_error=maximum(abs,J-self-future), final_lens_error=final_error,
            final_literal_lens_error=literal_error),
        wall_seconds=time()-started)
    path = joinpath(@__DIR__,"../artifacts/specimen-jlens.json")
    writejson(path,pack(result);sig=6)
    @printf("Specimen export: %.3f seconds; %d bytes; %d shared probability columns\n",result.wall_seconds,filesize(path),size(interventions.probability_table,2))
    @printf("Final-depth lens max probability error: %.12g; literal paper-weighted: %.12g\n",final_error,literal_error)
    filesize(path) < 3*1024^2 || @warn "Specimen exceeds approximate 3 MiB budget"
end
export_specimen()
