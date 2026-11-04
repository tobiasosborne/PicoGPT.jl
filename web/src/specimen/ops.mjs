// The kernels deliberately use scalar Float64 arithmetic. No autodiff, BLAS, or runtime deps.

/** Y = W X + b; matrices are column-major. */
export function affine(W, b, X, Y, M, K, N) {
  Y.fill(0);
  for (let n = 0; n < N; n++) {
    const yn = n * M, xn = n * K;
    for (let k = 0; k < K; k++) {
      const x = X[xn + k], wk = k * M;
      for (let i = 0; i < M; i++) Y[yn + i] += W[wk + i] * x;
    }
    for (let i = 0; i < M; i++) Y[yn + i] += b[i];
  }
}

/** Accumulate W̄ = Ȳ Xᵀ, b̄ = Ȳ 1, X̄ = Wᵀ Ȳ. */
export function affinePullback(W, X, dY, dW, db, dX, M, K, N) {
  for (let n = 0; n < N; n++) {
    const yn = n * M, xn = n * K;
    for (let i = 0; i < M; i++) db[i] += dY[yn + i];
    for (let k = 0; k < K; k++) {
      const x = X[xn + k], wk = k * M;
      let dx = 0;
      for (let i = 0; i < M; i++) {
        const dy = dY[yn + i];
        dW[wk + i] += dy * x;
        dx += W[wk + i] * dy;
      }
      dX[xn + k] += dx;
    }
  }
}

/** Centre, scale with population variance + eps, then diagonal affine map. */
export function layerNorm(X, gamma, beta, Y, cache, D, N, eps) {
  const { normalized, invstd } = cache;
  for (let n = 0; n < N; n++) {
    const off = n * D;
    let mean = 0, variance = 0;
    for (let i = 0; i < D; i++) mean += X[off + i];
    mean /= D;
    for (let i = 0; i < D; i++) variance += (X[off + i] - mean) ** 2;
    const r = 1 / Math.sqrt(variance / D + eps);
    invstd[n] = r;
    for (let i = 0; i < D; i++) {
      const z = (X[off + i] - mean) * r;
      normalized[off + i] = z;
      Y[off + i] = gamma[i] * z + beta[i];
    }
  }
}

export function layerNormPullback(gamma, dY, dgamma, dbeta, dX, cache, D, N) {
  const { normalized, invstd } = cache;
  for (let n = 0; n < N; n++) {
    const off = n * D;
    let mean = 0, projection = 0;
    for (let i = 0; i < D; i++) {
      const dy = dY[off + i], z = normalized[off + i], g = dy * gamma[i];
      mean += g; projection += g * z;
      dgamma[i] += dy * z;
      dbeta[i] += dy;
    }
    mean /= D; projection /= D;
    for (let i = 0; i < D; i++) {
      dX[off + i] += invstd[n] * (dY[off + i] * gamma[i] - mean - normalized[off + i] * projection);
    }
  }
}

/** S[i,j] = q_i · k_j / √d, including the future entries; A alone is masked. */
export function attentionScores(Q, K, S, D, H, T, B) {
  const d = D / H, scale = 1 / Math.sqrt(d);
  for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) {
    const base = (b * H + h) * T * T, xbase = b * T * D + h * d;
    for (let j = 0; j < T; j++) for (let i = 0; i < T; i++) {
      let s = 0;
      for (let r = 0; r < d; r++) s += Q[xbase + i * D + r] * K[xbase + j * D + r];
      S[base + i + j * T] = s * scale;
    }
  }
}

export function attentionValues(V, A, O, D, H, T, B) {
  const d = D / H;
  O.fill(0);
  for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) {
    const base = (b * H + h) * T * T, xbase = b * T * D + h * d;
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
      const a = A[base + i + j * T];
      for (let r = 0; r < d; r++) O[xbase + i * D + r] += V[xbase + j * D + r] * a;
    }
  }
}

// Separate operations let a figure intervene at S or A and let gradcheck check their adjoints.
export function attentionProbabilities(S, A, T, H, B) {
  A.fill(0);
  for (let bh = 0; bh < B * H; bh++) for (let i = 0; i < T; i++) {
    const base = bh * T * T;
    let max = -Infinity, sum = 0;
    for (let j = 0; j <= i; j++) max = Math.max(max, S[base + i + j * T]);
    for (let j = 0; j <= i; j++) {
      const a = Math.exp(S[base + i + j * T] - max);
      A[base + i + j * T] = a; sum += a;
    }
    for (let j = 0; j <= i; j++) A[base + i + j * T] /= sum;
  }
}

export function attentionPullback(Q, K, V, A, dO, dQ, dK, dV, dS, dA, D, H, T, B) {
  const d = D / H, scale = 1 / Math.sqrt(d);
  for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) {
    const base = (b * H + h) * T * T, xbase = b * T * D + h * d;
    for (let i = 0; i < T; i++) {
      // O = V Aᵀ; expose also Ā above the mask (its pullback through softmax is zero).
      for (let j = 0; j < T; j++) {
        const ix = base + i + j * T, a = A[ix];
        let da = 0;
        for (let r = 0; r < d; r++) {
          const oi = xbase + i * D + r, vj = xbase + j * D + r;
          da += dO[oi] * V[vj];
          dV[vj] += dO[oi] * a;
        }
        dA[ix] += da;
      }
      let projection = 0;
      for (let j = 0; j <= i; j++) projection += A[base + i + j * T] * dA[base + i + j * T];
      for (let j = 0; j <= i; j++) {
        const ix = base + i + j * T;
        dS[ix] += A[ix] * (dA[ix] - projection);
      }
      for (let j = 0; j < T; j++) {
        const ds = dS[base + i + j * T] * scale;
        for (let r = 0; r < d; r++) {
          const qi = xbase + i * D + r, kj = xbase + j * D + r;
          dQ[qi] += ds * K[kj]; dK[kj] += ds * Q[qi];
        }
      }
    }
  }
}

export function add(X, Y, Z) { for (let i = 0; i < Z.length; i++) Z[i] = X[i] + Y[i]; }
export function accumulate(X, Y) { for (let i = 0; i < X.length; i++) X[i] += Y[i]; }
