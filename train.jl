# train.jl -- Training loop: SGD, LR schedule, gradient clipping
#
# Depends on: config.jl, tensor_ops.jl, model.jl

# ===================================================================
# LR Schedule: linear warmup + cosine decay
# ===================================================================

# eta(t) = eta_max * (t / warmup_steps)                              if t < warmup_steps
# eta(t) = eta_min + 0.5*(eta_max - eta_min)*(1 + cos(π * progress)) otherwise
# where progress = (t - warmup_steps) / (total_steps - warmup_steps)

function lr_schedule(step::Int, cfg::TrainConfig, total_steps::Int)
    if step < cfg.warmup_steps
        return cfg.learning_rate * (step / cfg.warmup_steps)
    end
    progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
    progress = min(progress, 1.0)
    cfg.lr_min + 0.5 * (cfg.learning_rate - cfg.lr_min) * (1.0 + cos(π * progress))
end

# ===================================================================
# Gradient clipping: global norm
# ===================================================================

# g <- g * min(1, c / ||g||_2)

function _sum_sq(grads::Dict{Symbol,Any})
    total = 0.0
    for (_, v) in grads
        if v isa AbstractArray{Float64}
            total += sum(abs2, v)
        elseif v isa AttentionGrads
            for f in fieldnames(AttentionGrads)
                total += sum(abs2, getfield(v, f))
            end
        elseif v isa Vector  # block grads list
            for blk_g in v
                total += _sum_sq(blk_g)
            end
        elseif v isa Dict
            total += _sum_sq(v)
        end
    end
    total
end

function grad_global_norm(grads::Dict{Symbol,Any})
    sqrt(_sum_sq(grads))
end

function clip_gradients!(grads::Dict{Symbol,Any}, max_norm::Float64)
    gnorm = grad_global_norm(grads)
    if gnorm > max_norm
        scale = max_norm / gnorm
        scale_grads!(grads, scale)
    end
    gnorm
end

function scale_grads!(grads::Dict{Symbol,Any}, scale::Float64)
    for (k, v) in grads
        if v isa AbstractArray{Float64}
            v .*= scale
        elseif v isa AttentionGrads
            v.Wq .*= scale; v.bq .*= scale
            v.Wk .*= scale; v.bk .*= scale
            v.Wv .*= scale; v.bv .*= scale
            v.Wo .*= scale; v.bo .*= scale
        elseif v isa Vector  # block grads list
            for blk_g in v
                scale_grads!(blk_g, scale)
            end
        elseif v isa Dict
            scale_grads!(v, scale)
        end
    end
end

# ===================================================================
# Training loop
# ===================================================================

function train_epoch!(model::GPT, batches, cfg::TrainConfig, step::Int, total_steps::Int)
    total_loss = 0.0
    n_batches = length(batches)

    for (input, target) in batches
        step += 1
        lr = lr_schedule(step, cfg, total_steps)

        logits, cache = gpt_forward(model, input)
        loss, cache_ce = cross_entropy_forward(logits, target)
        grad_logits = cross_entropy_backward(cache_ce)
        grads = gpt_backward(model, cache, grad_logits)

        clip_gradients!(grads, cfg.max_grad_norm)
        sgd_update!(model, grads, lr)

        total_loss += loss
    end

    avg_loss = total_loss / n_batches
    (avg_loss, step)
end

function evaluate(model::GPT, batches)
    total_loss = 0.0
    for (input, target) in batches
        logits, _ = gpt_forward(model, input)
        loss, _ = cross_entropy_forward(logits, target)
        total_loss += loss
    end
    total_loss / max(length(batches), 1)
end

function train!(model::GPT, train_batches, val_batches, cfg::TrainConfig)
    total_steps = cfg.n_epochs * length(train_batches)
    step = 0
    best_val = Inf

    for epoch in 1:cfg.n_epochs
        train_loss, step = train_epoch!(model, train_batches, cfg, step, total_steps)

        if epoch % cfg.eval_interval == 0
            val_loss = evaluate(model, val_batches)
            best_val = min(best_val, val_loss)
            @printf("Epoch %3d | train loss: %.4f | val loss: %.4f | lr: %.2e | best val: %.4f\n",
                    epoch, train_loss, val_loss, lr_schedule(step, cfg, total_steps), best_val)
        end
    end

    best_val
end
