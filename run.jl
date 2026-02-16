# run.jl -- Entry point: data -> model -> train -> generate
#
# Usage: julia run.jl

using Printf

include("config.jl")
include("gradcheck.jl")
include("tensor_ops.jl")
include("attention.jl")
include("model.jl")
include("data.jl")
include("train.jl")
include("generate.jl")

function main()
    println("="^60)
    printstyled("PicoGPT.jl — From-Scratch GPT in Pure Julia\n", bold=true)
    println("="^60)

    # --- Data ---
    println("\n[1/4] Loading data...")
    text = download_shakespeare()
    char_to_idx, idx_to_char = build_vocab(text)
    vocab_size = length(char_to_idx)
    println("  Vocab size: $vocab_size")
    println("  Text length: $(length(text)) chars")

    data = encode(text, char_to_idx)
    train_data, val_data = train_val_split(data)

    cfg_m = GPTConfig(vocab_size=vocab_size)
    cfg_t = TrainConfig()

    train_batches = make_batches(train_data, cfg_m.seq_len, cfg_t.batch_size)
    val_batches   = make_batches(val_data, cfg_m.seq_len, cfg_t.batch_size)
    println("  Train batches: $(length(train_batches))")
    println("  Val batches:   $(length(val_batches))")

    # --- Model ---
    println("\n[2/4] Initializing model...")
    model = init_gpt(cfg_m)
    n_params = count_parameters(model)
    @printf("  Parameters: %d (~%.2fM)\n", n_params, n_params / 1e6)

    # --- Train ---
    println("\n[3/4] Training...")
    best_val = train!(model, train_batches, val_batches, cfg_t)
    @printf("\n  Best val loss: %.4f\n", best_val)

    # --- Generate ---
    println("\n[4/4] Generating sample text...\n")
    prompt = "\nFIRST CITIZEN:\n"
    context = encode(prompt, char_to_idx)
    tokens = generate(model, context, 500; temperature=0.8)
    println(decode(tokens, idx_to_char))

    println("\n", "="^60)
    println("Done.")
end

function count_parameters(model::GPT)
    n = 0
    n += length(model.E_tok)
    n += length(model.W_out) + length(model.b_out)
    n += length(model.ln_f_gamma) + length(model.ln_f_beta)
    for block in model.blocks
        for f in [:Wq, :bq, :Wk, :bk, :Wv, :bv, :Wo, :bo]
            n += length(getfield(block.attn, f))
        end
        n += length(block.W1) + length(block.b1)
        n += length(block.W2) + length(block.b2)
        n += length(block.ln1_gamma) + length(block.ln1_beta)
        n += length(block.ln2_gamma) + length(block.ln2_beta)
    end
    n
end

main()
