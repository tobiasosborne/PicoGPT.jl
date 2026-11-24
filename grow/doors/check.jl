# Finite-difference check of the looped model's gradient (all of θ and the gate), k = 1 and 2.
include("doors.jl")
rng = Xoshiro(1); c = config(8, 2, 16)
for (k, g) in ((1, 0.0), (2, 0.37), (2, 0.0), (2, 1.0)), world in (fine,)
    θ = newθ(c, rng; g); θ .+= 0.1 .* randn(rng, length(θ)); θ[end] = g
    data = world(rng, 3)
    f(t) = looped(t, data..., c, k)[1]
    θ̄ = looped(θ, data..., c, k)[3]()
    e = gradcheck(f, θ, θ̄)
    eg = abs((f(vcat(θ[1:end-1], g + 1e-6)) - f(vcat(θ[1:end-1], g - 1e-6))) / 2e-6 - θ̄[end])
    @printf("k=%d g=%.2f  p=%d  relative error of ∇ℓ: %.2e   gate: analytic %+.6e, |FD − analytic| %.1e\n", k, g, length(θ), e, θ̄[end], eg)
end
