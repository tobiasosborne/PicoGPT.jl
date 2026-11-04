# Shared by the scripts: paths, the Tiny Shakespeare data, checkpoints, and JSON output.
# (Tooling, not part of the book.)

using PicoGPT, LinearAlgebra, Random, Printf, Downloads

const ROOT = dirname(@__DIR__)
artifact(parts...) = joinpath(ROOT, "artifacts", parts...)
const CHECKPOINTS = artifact("shakespeare", "checkpoints")

env(name, default) = haskey(ENV, name) ? parse(typeof(default), ENV[name]) : default

# ── data ─────────────────────────────────────────────────────────────────────────────

const SHAKESPEARE_URL =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

"""Tiny Shakespeare as one string (downloaded once to artifacts/data/)."""
function tiny_shakespeare()
    path = artifact("data", "tinyshakespeare.txt")
    if !isfile(path)
        mkpath(dirname(path))
        @info "downloading Tiny Shakespeare" SHAKESPEARE_URL
        Downloads.download(SHAKESPEARE_URL, path)
    end
    read(path, String)
end

"""The token sequence, split 90/10 into training and validation text (in order)."""
function shakespeare_split(text, abc)
    data = encode(text, abc)
    n = round(Int, 0.9 * length(data))
    data[1:n], data[n+1:end]
end

"""Mean loss over the windows starting at `starts`, in parallel over pairs of sequences."""
function meanloss(θ, data, starts, c)
    x, y = windows(data, c.T, starts)
    isempty(x) && throw(ArgumentError("evaluation needs a nonempty batch"))
    groups = Iterators.partition(1:size(x, 2), 2)
    PicoGPT.without_blas_threads() do
        tasks = @sync [Threads.@spawn length(J) * first(loss(θ, x[:, J], y[:, J], c)) for J in groups]
        sum(fetch.(tasks)) / size(x, 2)
    end
end

# ── checkpoints ──────────────────────────────────────────────────────────────────────
# A small binary format, little-endian:
#   "PICOGPT1" | Int64 × 6: V D H L T F | Int64 step | Int64 t (Adam steps)
#   | Int64 n, n bytes: the alphabet as UTF-8 | Int64 k, then k × (Int64 m, m × Float32)
# The k arrays are θ, and optionally Adam's m and v.
# This training format stores Float32 exactly; reject other types rather than silently
# losing precision. The optimiser hyperparameters are supplied by the training recipe.

function save_checkpoint(path; c, abc, step, θ, opt = nothing)
    arrays = opt === nothing ? (θ,) : (θ, opt.m, opt.v)
    all(a -> eltype(a) == Float32 && length(a) == nparams(c), arrays) ||
        throw(ArgumentError("checkpoints require Float32 parameter/state vectors of length nparams(c)"))
    length(abc) == c.V || throw(DimensionMismatch("the alphabet must have V characters"))
    mkpath(dirname(path))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, "PICOGPT1")
        write(io, htol.(Int64[c.V, c.D, c.H, c.L, c.T, c.F, step, opt === nothing ? 0 : opt.t]))
        s = String(abc)
        write(io, htol(Int64(sizeof(s))), s)
        write(io, htol(Int64(length(arrays))))
        for a in arrays
            write(io, htol(Int64(length(a))), htol.(reinterpret(UInt32, a)))
        end
    end
    mv(tmp, path; force = true)
end

function load_checkpoint(path)
    open(path) do io
        String(read(io, 8)) == "PICOGPT1" || error("$path is not a PicoGPT checkpoint")
        V, D, H, L, T, F, step, t = ltoh.(read!(io, Vector{Int64}(undef, 8)))
        c = Config(V, D, H, L, T, F)
        n = ltoh(read(io, Int64))
        0 < n ≤ 4V || error("invalid alphabet byte count in $path")
        abc = collect(String(read(io, n)))
        length(abc) == V || error("alphabet size differs from V in $path")
        k = ltoh(read(io, Int64))
        k in (1, 3) || error("invalid state array count in $path")
        arrays = map(1:k) do _
            ltoh(read(io, Int64)) == nparams(c) || error("invalid parameter count in $path")
            copy(reinterpret(Float32, ltoh.(read!(io, Vector{UInt32}(undef, nparams(c))))))
        end
        eof(io) || error("unexpected trailing data in $path")
        (c, abc, step, t, θ = arrays[1], adam = arrays[2:end])
    end
end

checkpoint_path(step) = joinpath(CHECKPOINTS, @sprintf("step_%06d.bin", step))

# ── JSON ─────────────────────────────────────────────────────────────────────────────
# Named tuples and dicts become objects (in order), vectors and tuples arrays. Floats are
# printed in full (shortest round-trip form) unless `sig` asks for significant digits.

json(x; sig = nothing) = sprint(io -> json(io, x; sig))
writejson(path, x; sig = nothing) = (mkpath(dirname(path)); write(path, json(x; sig) * "\n"); path)

json(io::IO, x::AbstractString; sig = nothing) = print(io, '"', escape_json(x), '"')
json(io::IO, x::AbstractChar; sig = nothing) = json(io, string(x))
json(io::IO, x::Symbol; sig = nothing) = json(io, string(x))
json(io::IO, x::Bool; sig = nothing) = print(io, x)
json(io::IO, x::Integer; sig = nothing) = print(io, x)
json(io::IO, ::Nothing; sig = nothing) = print(io, "null")
function json(io::IO, x::AbstractFloat; sig = nothing)
    isfinite(x) || error("JSON has no $x")
    iszero(x) ? print(io, 0) : print(io, sig === nothing ? x : round(Float64(x); sigdigits = sig))
end
function json(io::IO, xs::Union{AbstractVector,Tuple}; sig = nothing)
    print(io, '[')
    for (k, x) in enumerate(xs)
        k > 1 && print(io, ',')
        json(io, x; sig)
    end
    print(io, ']')
end
function json(io::IO, d::Union{NamedTuple,AbstractDict}; sig = nothing)
    print(io, '{')
    for (k, (key, x)) in enumerate(pairs(d))
        k > 1 && print(io, ',')
        json(io, string(key)); print(io, ':'); json(io, x; sig)
    end
    print(io, '}')
end

function escape_json(s)
    b = IOBuffer()
    for ch in s
        ch == '"'  ? print(b, "\\\"") :
        ch == '\\' ? print(b, "\\\\") :
        ch == '\n' ? print(b, "\\n") :
        ch == '\t' ? print(b, "\\t") :
        ch == '\r' ? print(b, "\\r") :
        ch < ' '   ? print(b, "\\u", string(UInt16(ch); base = 16, pad = 4)) : print(b, ch)
    end
    String(take!(b))
end

"""An array as `{shape, data}`, the data in column-major order."""
tensor(A) = (shape = collect(size(A)), data = vec(collect(A)))
tensor(A::Number) = A
