---
title: "Mathematical backbone: values, adjoints, and learning"
lang: en
---

# Scope and conventions {#derivations-scope}

These are author and reviewer notes for the exact model in `BRIEF.md`, §§2–3.
All vector spaces are finite-dimensional real Euclidean spaces. Matrix spaces
carry the Frobenius inner product
$\langle U,V\rangle_F=\operatorname{tr}(U^\top V)=\sum_{ij}U_{ij}V_{ij}$.
Columns are tokens: $X\in\mathbb R^{D\times N}$, $N=TB$.
Sequences occupy contiguous groups of $T$ columns; attention never crosses
between groups. A bar denotes a cotangent (identified with a vector by the
Euclidean inner product), not a time average or a complex conjugate.
The pullback is the transpose of the derivative applied to that cotangent.

$V$ is the vocabulary size. The brief also names the projected value array
`V`; below that array is written $V_h$ within a head, or described as the
value array when all heads are meant. A local probability vector $p$ is
distinct from the scalar parameter count $p=\dim\theta$; the meaning is
always stated. $\mathbf1_k$ is the all-ones vector in $\mathbb R^k$.
Means in LayerNorm are over the $D$ features of one token, never over token
columns. Softmax is columnwise for logits and rowwise for attention.
Every loss gradient includes the single averaging factor $1/N$ at the loss;
there is no further averaging in affine, embedding, or LayerNorm pullbacks.

The notation $dX$ means a perturbation, not the head width $d=D/H$.
An expression such as $\bar X\mathrel{+}=G$ means accumulation into a
cotangent initially set to zero. Parameters are independent unless explicitly
shared: in particular, $W_{\rm out}$ is **not tied** to $E^\top$.

**Decisions and qualifications for the authors.**

1. The binding numerical LayerNorm semantics are
   $\sigma=\sqrt{D^{-1}\|x-\mu\mathbf1\|^2+10^{-5}}$. Its normalised
   vector lies inside the sphere of radius $\sqrt D$, not on it.
   The sphere description in §3 is exact only for $\varepsilon=0$ and
   nonconstant inputs. The compact pullback below is exact in both cases.
2. Free energy is $\mathcal F_\beta(E)=-\beta^{-1}\log\mathcal Z_\beta(E)$.
   Its gradient with respect to **energies** is the Gibbs distribution.
   With logits $z=-E$, the gradient of free energy with respect to logits
   is its negative. The positive log-sum-exp has gradient equal to softmax.
3. ReLU has no derivative at zero. The implementation convention is
   a backward value of zero there; this is a selected subgradient, not an
   ordinary derivative. Finite differences must avoid these kinks.
4. §3 does not fix the initialisation distribution, AdamW hyperparameters
   or decay exclusions, sinusoidal frequency convention, position origin,
   or constraints for odd $D$. These must be fixed for trace conformance.
   The conventional sinusoidal construction below uses even $D$, paired
   sine/cosine rows, base $10000$, and positions $0,1,\ldots,T-1$.
   $D/H$ must be an integer. The specimen and real configuration satisfy
   both divisibility and evenness requirements.
5. Residual variance scaling and independent-entry attention variance are
   initialisation arguments with stated hypotheses, not exact identities
   about trained activations. No change to the exact model is needed.

Each numbered result below has a derivation, a physicist's intuition, and a
code-role pointer. Source names are deliberately not a dependency: `src/`
is being rewritten concurrently. The independent verifier is
`book/notes/verify_derivations.jl`; its local function names are also given
where useful. It does not import the repository model.

# Pullbacks and the architecture {#derivations-pullbacks}

## 1. A pullback is an adjoint {#derivation-adjoint}

**Statement.** For differentiable $f:\mathbb R^n\to\mathbb R^m$ and scalar
loss $\mathcal L=\ell\circ f$,
$$
\boxed{\bar x=Df(x)^\top\bar y},\qquad y=f(x),\quad
\bar y=\nabla_y\ell.
$$
This formula defines the pullback for any output cotangent $\bar y$, whether
or not a loss has yet been chosen.

**Proof.** The derivative is the unique linear map satisfying
$f(x+h)=f(x)+Df(x)h+o(\|h\|)$. Thus
$$
d\mathcal L[h]=\langle\bar y,Df(x)h\rangle
=\langle Df(x)^\top\bar y,h\rangle.
$$
Uniqueness of the Euclidean representative of a linear functional gives
the formula. For $f_k\circ\cdots\circ f_1$, write $J_i=Df_i$ at its
forward input. The chain rule and $(AB)^\top=B^\top A^\top$ give
$$
D(f_k\circ\cdots\circ f_1)^\top
=J_1^\top J_2^\top\cdots J_k^\top.
$$
Consequently the computation starts at the loss with cotangent $1$,
applies $J_k^\top$ first, and visits the maps in reverse order.
No full Jacobian needs to be constructed.

**Cost statement and proof.** For a fixed collection of elementary
operations whose pullbacks cost at most a constant times their forward
cost, a scalar loss and all $p$ parameter derivatives cost
$O(C)$ arithmetic when the forward program costs $C$. A forward evaluation
records needed intermediates; a reverse traversal applies each local
pullback once and accumulates its contributions. Summing local bounds
proves the claim. Matrix multiplication and its two transposed products,
elementwise operations, reductions, gather, and attention all satisfy
this condition. Materialising the output gradient costs $\Omega(p)$,
already within the cost of evaluating the parameter-dependent model.
Storage for the recorded intermediates is additional; checkpointing can
trade storage for recomputation. This is an arithmetic bound, not a promise
of identical memory use or wall-clock time.

An ordinary forward-mode pass propagates one tangent direction $v$ and
computes $D\mathcal L(\theta)v$. Recovering all $p$ coordinates by seeding
$e_1,\ldots,e_p$ takes $p$ such passes, or $O(pC)$ arithmetic when done
together. Forward mode remains useful for a small number of directions;
the comparison concerns a full gradient of one scalar.

**Fan-out statement and proof.** For the duplication map $\Delta(x)=(x,x)$,
$D\Delta[h]=(h,h)$ and
$\langle(\bar u,\bar v),(h,h)\rangle=\langle\bar u+\bar v,h\rangle$.
Hence $\Delta^*(\bar u,\bar v)=\bar u+\bar v$.
Every use of a shared value or parameter contributes additively to its
cotangent. In particular, $y=x+f(x)$ gives
$\bar x=\bar y+Df(x)^\top\bar y$.

**Intuition for a physicist.** A derivative transports a small displacement;
its adjoint transports the linear response of a measurement to that
displacement. The scalar loss is the measurement. At a junction, responses
from all possible downstream paths add, just as the variation of a sum
of actions is the sum of their variations. The residual stream has such
junctions at every sublayer.

**Code role.** Every operation returns its value and a closure implementing
this transpose action. The model backward traversal and gradient buffer
accumulation implement the chain rule and fan-out. Verifier: `apply!`,
`Node`, and `model` compose the separately written primitive pullbacks.

## 2. Affine maps and broadcast bias {#derivation-affine}

**Statement.** Let $W\in\mathbb R^{r\times s}$,
$X\in\mathbb R^{s\times N}$, $b\in\mathbb R^r$, and
$Y=WX+b\mathbf1_N^\top$. Then
$$
\boxed{\bar W=\bar YX^\top,\qquad
\bar X=W^\top\bar Y,\qquad \bar b=\bar Y\mathbf1_N.}
$$

**Derivation.** Since
$dY=(dW)X+W(dX)+(db)\mathbf1_N^\top$, cyclicity of the trace gives
$$
\begin{aligned}
\langle\bar Y,dY\rangle_F
&=\operatorname{tr}(X\bar Y^\top dW)
 +\operatorname{tr}(\bar Y^\top WdX)
 +\operatorname{tr}(\bar Y^\top db\mathbf1_N^\top)\\
&=\langle\bar YX^\top,dW\rangle_F
 +\langle W^\top\bar Y,dX\rangle_F
 +\langle\bar Y\mathbf1_N,db\rangle.
\end{aligned}
$$
In indices, $\bar W_{ij}=\sum_n\bar Y_{in}X_{jn}$,
$\bar X_{jn}=\sum_iW_{ij}\bar Y_{in}$ and
$\bar b_i=\sum_n\bar Y_{in}$. The bias is reused in every column;
its cotangent is a sum, not a mean.

**Intuition for a physicist.** A matrix mixes coordinates; its transpose
pulls a sensitivity back through that mixing. The weight response is an
outer product of the response at the output and the signal at the input,
summed over observations. A uniform applied bias couples to every token,
so its conjugate response is extensive in the number of tokens.

**Code role.** Query, key, value and output projections, both MLP maps,
and the vocabulary projection. Verifier: `affine`.

## 3. Embedding: gather forwards, scatter-add backwards {#derivation-embedding}

**Statement.** For fixed token IDs $x_n\in\{1,\ldots,V\}$, let
$C=[e_{x_1}\ \cdots\ e_{x_N}]\in\mathbb R^{V\times N}$.
The embedding and position addition are
$X_0=EC+P_{\rm batch}$. Their pullbacks are
$$
\boxed{\bar E=\bar X_0C^\top,\qquad
\bar E_{:,a}=\sum_{n:x_n=a}\bar X_{0,:,n}.}
$$
There is no derivative with respect to a discrete token ID. Positions
are fixed data and receive no trainable cotangent.

**Proof.** With IDs held fixed, $dX_0=(dE)C$; apply the affine formula
without a bias. In indices, $C_{an}=1$ precisely when $a=x_n$, yielding
the stated sum. If $P$ were trainable, its gather would have the same
scatter-add pullback, including sums over sequences reusing a position.
It is not trainable in this model.

**Intuition for a physicist.** Looking up a column is applying a basis
vector to a linear map. Each occurrence of the same character probes the
same column, so all those probes contribute to the response of that
column. Overwriting rather than adding would discard observations.

**Code role.** Token embedding lookup, fixed positional addition, and
embedding gradient accumulation. Verifier: `gather`, `add_fixed`.

## 4. ReLU {#derivation-relu}

**Statement and derivation.** $r(u)_i=\max(0,u_i)$ has
$Dr(u)=\operatorname{diag}(\mathbf1_{u_i>0})$ when all $u_i\ne0$.
Therefore
$$
\boxed{\bar u=\bar r\odot\mathbf1_{u>0}.}
$$
For a positive coordinate the local map is the identity; for a negative
coordinate it is constant. At zero the left and right derivatives differ.
The convex scalar subdifferential is $[0,1]$ and the code selects $0$.
At a kink this is a backward convention; it does not establish a classical
derivative of the composed loss or a convex subgradient of a nonconvex model.

**Intuition for a physicist.** ReLU is an ideal rectifier. A small signal
passes through an open gate and has no effect through a closed gate.
Exactly at the switching threshold linear response has two one-sided limits.

**Code role.** The MLP nonlinearity after $W_1\widehat F+b_1$.
Verifier: `relu`; zero convention checked separately from finite differences.

## 5. Softmax and its adjoint {#derivation-softmax}

**Statement.** For $z\in\mathbb R^k$ and
$p_i=e^{z_i}/\sum_je^{z_j}$,
$$
\boxed{Dp(z)=\operatorname{diag}(p)-pp^\top,\qquad
\bar z=p\odot\bigl(\bar p-\langle p,\bar p\rangle\mathbf1_k\bigr).}
$$
The same vector formula applies separately to each logit column or
attention row.

**Proof.** The quotient rule yields
$\partial p_i/\partial z_j=p_i\delta_{ij}-p_ip_j$.
This Jacobian is symmetric. Multiplying it by $\bar p$ gives the pullback.
It kills $\mathbf1$, consistently with shift invariance. Moreover,
$$
v^\top Dp(z)v=\sum_ip_iv_i^2-\left(\sum_ip_iv_i\right)^2
=\operatorname{Var}_{i\sim p}(v_i)\ge0.
$$
For finite logits $p_i>0$, the nullspace consists exactly of constant
vectors (or the whole one-dimensional space for $k=1$).

**Intuition for a physicist.** A change of one energy level redistributes
probability among all levels. The response is a covariance: only deviations
from a common shift matter. The subtraction of the probability-weighted
mean is the adjustment required to conserve total probability.

**Code role.** Stable softmax and its pullback; row reduction for attention,
column reduction for output probabilities. Verifier: `softmax_op`.

## 6. Log-sum-exp cross-entropy {#derivation-loss}

**Statement.** For targets $y_n\in\{1,\ldots,V\}$ and logits
$Z\in\mathbb R^{V\times N}$,
$$
\mathcal L=\frac1N\sum_{n=1}^N\left[
\log\sum_{a=1}^Ve^{Z_{an}}-Z_{y_n,n}\right],\qquad
\boxed{\bar Z_{:,n}=\frac{p_{:,n}-e_{y_n}}N.}
$$
Here $p_{:,n}=\operatorname{softmax}(Z_{:,n})$ and the incoming loss
cotangent is $1$; multiply the formula by $\bar{\mathcal L}$ otherwise.

**Derivation and numerical form.** For $a(z)=\log\sum_ie^{z_i}$,
$da=\sum_ip_i\,dz_i$. The target term contributes $-dz_y$.
Summation and division by $N$ prove the result; $\mathbf1^\top\bar Z_{:,n}=0$.
For any real $c$,
$$
a(z)=c+\log\sum_i e^{z_i-c}.
$$
Set $c=\max_i z_i$ to avoid exponential overflow. Although $\max$ itself
has kinks at ties, this expression is identically the smooth function
$a$ for finite logits. Its hand-derived pullback is therefore valid at
ties as well. Compute the loss from logits, not from
$\log(p_y+\delta)$: adding $\delta$ changes both the objective and its
gradient, and cannot repair probability underflow consistently.

**Intuition for a physicist.** The target energy and the partition-function
normalisation compete. The derivative is the model's occupation vector
minus the observed occupation vector: predicted counts minus observed
counts, with the batch average supplying the unit of one training token.

**Code role.** Mean next-token loss, stable log-sum-exp, initial backward
seed. Verifier: `ce`, including extreme finite logits and zero column sums.

## 7. LayerNorm as geometry, including epsilon {#derivation-layernorm}

**Statement.** For one column $x\in\mathbb R^D$, define
$$
\Pi=I-\frac{\mathbf1_D\mathbf1_D^\top}D,\quad
\mu=\frac{\mathbf1_D^\top x}D,\quad u=\Pi x=x-\mu\mathbf1_D,
\quad\sigma=\sqrt{\frac{u^\top u}D+\varepsilon},\quad
\hat x=\frac u\sigma,\quad y=\gamma\odot\hat x+\beta.
$$
The variance divisor is $D$, not the unbiased sample divisor $D-1$.
With $g=\gamma\odot\bar y$ (the cotangent of $\hat x$),
$$
\boxed{\bar x=\frac1\sigma\left[
g-\operatorname{mean}(g)\mathbf1_D
-\hat x\,\operatorname{mean}(g\odot\hat x)\right].}
$$
If the cotangent of $\hat x$ is instead called $\bar g$, the identical
requested formula reads
$\bar x=\sigma^{-1}(\bar g-\operatorname{mean}(\bar g)\mathbf1_D
-\hat x\operatorname{mean}(\bar g\odot\hat x))$.
The symbol $g$ above is already a cotangent; it must not be pulled back twice.
For all token columns,
$$
\boxed{\bar\gamma=\sum_n\bar Y_{:,n}\odot\hat X_{:,n},\qquad
\bar\beta=\sum_n\bar Y_{:,n}.}
$$

**Proof of the geometric construction at $\varepsilon=0$.**
$\Pi^\top=\Pi$ and $\Pi^2=\Pi$; its kernel is
$\operatorname{span}(\mathbf1_D)$ and its image is $\mathbf1_D^\perp$.
Thus centring is an orthogonal projection. For $u\ne0$, let
$\hat u=u/\|u\|$. Differentiating the norm gives
$d\|u\|[v]=\hat u^\top v$, and hence
$$
D\!\left(\frac{\sqrt D\,u}{\|u\|}\right)
=\frac{\sqrt D}{\|u\|}(I-\hat u\hat u^\top).
$$
The second factor is the orthogonal projection onto the tangent space
of the sphere at $\sqrt D\hat u$. Since $u\perp\mathbf1_D$,
centring and tangent projections commute. Their product is
$$
D\hat x=\frac1\sigma\left(\Pi-\frac{\hat x\hat x^\top}D\right),
\qquad\varepsilon=0.
$$
It annihilates constant shifts and radial rescalings of $u$.

**Exact proof for $\varepsilon\ge0$.** Assume $\sigma>0$, which is automatic
when $\varepsilon>0$. Because $u^\top\Pi=u^\top$,
$$
du=\Pi\,dx,\qquad d\sigma=\frac{u^\top dx}{D\sigma},\qquad
d\hat x=\left(\frac\Pi\sigma-
\frac{uu^\top}{D\sigma^3}\right)dx
=\frac1\sigma\left(\Pi-\frac{\hat x\hat x^\top}D\right)dx.
$$
This Jacobian is symmetric. Applying it to $g=\gamma\odot\bar y$ and
using $\hat x^\top g/D=\operatorname{mean}(g\odot\hat x)$ gives the compact
formula exactly. The affine differential
$dy=(d\gamma)\odot\hat x+\gamma\odot d\hat x+d\beta$ gives the two
parameter cotangents and their sums over reused columns.

**What epsilon changes.** For $u\ne0$,
$$
\|\hat x\|^2=D\left(1-\frac{\varepsilon}{\sigma^2}\right)<D,
\quad D\hat x\,\mathbf1_D=0,\quad
D\hat x\,u=\frac{\varepsilon}{\sigma^3}u.
$$
Thus the radial derivative is small, but is not zero. Directions
orthogonal to both $u$ and $\mathbf1_D$ have eigenvalue $1/\sigma$.
At a constant column, $u=0$ and the Jacobian is
$\Pi/\sqrt\varepsilon$; this can amplify a perturbation substantially.
For $D=1$ it is zero and the output is simply $\beta$.
At $\varepsilon=0$, constant columns make normalisation undefined.

There is an exact sphere picture with epsilon: lift the centred vector to
$\tilde u=(u,\sqrt{D\varepsilon})\in\mathbb R^{D+1}$, normalise it to
radius $\sqrt D$, then keep its first $D$ coordinates. Their value is
$u/\sqrt{\|u\|^2/D+\varepsilon}$. The extra coordinate explains why the
retained coordinates lie strictly inside the sphere.

**Intuition for a physicist.** Centring removes the uniform mode; exact
normalisation removes the radial mode. Only fluctuations tangent to the
fixed-radius sphere survive linear response. Epsilon regularises the
singularity at zero radius and leaves a finite radial response; it is a
soft constraint, not an exact fixed-length constraint. The final diagonal
affine map restores independently learned feature scales and offsets.

**Code role.** Two pre-normalisations per block and the final LayerNorm;
featurewise reductions and sums of parameter gradients over tokens.
Verifier: `ln`, `centre`, `sphere`, and explicit Jacobian equivalence checks
at $\varepsilon=0,10^{-5},0.3$, constant columns, and $D=1$.

## 8. One attention head, with orientation and mask {#derivation-attention}

**Statement.** For one head and one sequence, let
$Q_h,K_h,V_h\in\mathbb R^{d\times T}$ and
$$
S=\frac{Q_h^\top K_h}{\sqrt d},\quad
A_{ij}=\begin{cases}
\displaystyle\frac{e^{S_{ij}}}{\sum_{k\le i}e^{S_{ik}}},&j\le i,\\
0,&j>i,
\end{cases}\qquad O_h=V_hA^\top.
$$
For an incoming $\bar O_h\in\mathbb R^{d\times T}$,
$$
\boxed{\bar V_h=\bar O_h A,\qquad \bar A=\bar O_h^\top V_h,}
$$
$$
\boxed{\bar S_{ij}=\begin{cases}
A_{ij}\left(\bar A_{ij}-\sum_{k\le i}A_{ik}\bar A_{ik}\right),&j\le i,\\
0,&j>i,
\end{cases}}
$$
$$
\boxed{\bar Q_h=\frac{K_h\bar S^\top}{\sqrt d},\qquad
\bar K_h=\frac{Q_h\bar S}{\sqrt d}.}
$$
Rows of $A$ are readers (queries), columns are sources (keys).
In particular $O_{h,:,i}=\sum_{j\le i}A_{ij}V_{h,:,j}$.

**Derivation.** From $dO_h=(dV_h)A^\top+V_h(dA)^\top$,
$$
\langle\bar O_h,dO_h\rangle_F
=\langle\bar O_hA,dV_h\rangle_F
+\langle\bar O_h^\top V_h,dA\rangle_F.
$$
Apply the softmax pullback independently on each allowed row prefix
$1{:}i$. The mask is fixed: its meaning is restriction of the function
to allowed coordinates, not differentiation with respect to $-\infty$.
Masked raw scores have zero derivative. Finally
$dS=((dQ_h)^\top K_h+Q_h^\top dK_h)/\sqrt d$; taking Frobenius adjoints
gives the last two formulas. In indices,
$$
\bar Q_{h,:,i}=\frac1{\sqrt d}\sum_j\bar S_{ij}K_{h,:,j},\qquad
\bar K_{h,:,j}=\frac1{\sqrt d}\sum_i\bar S_{ij}Q_{h,:,i}.
$$
The first attention row has one allowed source, hence $A_{11}=1$ and
its entire score cotangent is zero. Masked entries and each row sum of
$\bar S$ are zero as well. These are useful checks of orientation.

**Intuition for a physicist.** A token reads a convex mixture of past
source vectors. The value response is the adjoint of that mixture; the
score response says how changing the mixture's weights would change the
measurement. Softmax then subtracts the weighted mean to preserve the
total mixture weight. Query and key responses split the bilinear score
variation into its two arguments.

**Code role.** Per-head, per-sequence masked attention. Cache unmasked
$S$ for the trace and $A$ for the pullback; never replace the cached
unmasked trace by $S+M$. Verifier: `head`, `masked_softmax`.

## 9. Multi-head attention is a direct sum {#derivation-multihead}

**Statement and proof.** Let $R_h\in\mathbb R^{d\times D}$ select rows
$(h-1)d+1{:}hd$. Then $R_hR_j^\top=\delta_{hj}I_d$ and
$\sum_hR_h^\top R_h=I_D$. After the three full-width affine maps, each
head reads $(R_hQ,R_hK,R_hV)$ independently and concatenation is
$O=\sum_hR_h^\top O_h$. The head map is a direct sum (also over
independent sequences); its derivative and adjoint are direct sums.
Thus $\bar O_h=R_h\bar O$, and the returned head cotangents are
placed into their corresponding row slices without overlap.
For the shared normalised input $U=\widehat A$,
$$
\bar U=W_q^\top\bar Q+W_k^\top\bar K+W_v^\top\bar V.
$$
Each projection's parameter gradients follow Result 2. The final mixing
$W_oO+b_o\mathbf1_N^\top$ also uses Result 2.
The direct-sum statement applies to the independent heads before $W_o$;
$W_o$ mixes their outputs, and all heads depend on the same $U$ through
the learned projections.

**Intuition for a physicist.** Splitting into heads is a decomposition
into independent channels for the mixing operation. Putting the channels
back together is an isometry, so its adjoint simply separates them again.
At the earlier common input, the responses of all channels add.

**Code role.** Row slicing, per-sequence attention, concatenation, output
projection, and accumulation at the shared LayerNorm output.
Verifier: `multihead` with two heads and two sequences.

## 10. Residual sublayers and the entire model {#derivation-whole-model}

**Statement.** A residual sublayer $Y=X+f(X;\phi)$ has
$$
\boxed{\bar X=\bar Y+D_Xf(X;\phi)^*\bar Y,\qquad
\bar\phi=D_\phi f(X;\phi)^*\bar Y.}
$$
Here $*$ is the Euclidean/Frobenius adjoint, that is, transpose in
coordinates. A block is two such sublayers:
$$
X_\ell^{\rm mid}=X_{\ell-1}+\operatorname{Attn}_\ell
(\operatorname{LN}_{1,\ell}(X_{\ell-1})),
$$
$$
X_\ell=X_\ell^{\rm mid}+W_{2,\ell}
\operatorname{relu}(W_{1,\ell}\operatorname{LN}_{2,\ell}
(X_\ell^{\rm mid})+b_{1,\ell}\mathbf1_N^\top)
+b_{2,\ell}\mathbf1_N^\top.
$$
Attention here includes its output projection and bias.

**Derivation and complete reverse recipe.** The residual formula follows
from $dY=dX+D_Xf[dX]+D_\phi f[d\phi]$ and inner products.
The full model is a composition of the maps just derived; the following
recipe completely specifies its gradient, including every fan-out.

1. Form $\bar Z=(\operatorname{softmax}_{\rm columns}(Z)-Y_{\rm onehot})/N$.
   Pull back through $W_{\rm out}U_f+b_{\rm out}\mathbf1_N^\top$, obtaining
   $\bar W_{\rm out},\bar b_{\rm out},\bar U_f$.
   Pull $\bar U_f$ through final LayerNorm to get
   $\bar X_L,\bar\gamma_f,\bar\beta_f$.
2. For $\ell=L,L-1,\ldots,1$, start with the MLP residual's incoming
   $B=\bar X_\ell$. Write its cached forward values as
   $U=\operatorname{LN}_{2,\ell}(X_\ell^{\rm mid})$,
   $C=W_1U+b_1\mathbf1_N^\top$, $R=\operatorname{relu}(C)$.
   Then $\bar W_2=BR^\top$, $\bar b_2=B\mathbf1_N$,
   $\bar R=W_2^\top B$, $\bar C=\bar R\odot\mathbf1_{C>0}$,
   $\bar W_1=\bar CU^\top$, $\bar b_1=\bar C\mathbf1_N$,
   $\bar U=W_1^\top\bar C$.
   Pull $\bar U$ through LN$_2$ and add its input cotangent to the
   identity branch $B$ to obtain $\bar X_\ell^{\rm mid}$.
3. Set $B=\bar X_\ell^{\rm mid}$ for the attention residual.
   $\bar W_o=BO^\top$, $\bar b_o=B\mathbf1_N$,
   $\bar O=W_o^\top B$. Split $\bar O$ into heads and sequences,
   use Result 8, and reassemble $\bar Q,\bar K,\bar V$.
   Pull each through its own affine projection. Add the three input
   cotangents at $\widehat A$, pull through LN$_1$, and add the result
   to $B$ to obtain $\bar X_{\ell-1}$. Keep all parameter cotangents.
4. At $X_0=E[:,x]+P[:,\mathrm{pos}]$, scatter-add $\bar X_0$ into
   $\bar E$. Flatten the accumulated parameter arrays in the binding
   order: `E`; per block `γ1 β1 Wq bq Wk bk Wv bv Wo bo γ2 β2 W1 b1 W2 b2`;
   then `γf βf Wout bout`. Matrices flatten column-major.

This is also a proof by repeated application of the adjoint chain rule.
Reshape, slice, and concatenate operations are fixed coordinate maps;
their adjoints reshape, scatter, and split. No unlisted learned operation
remains in the model.

**Intuition for a physicist.** At each residual junction the observable
has two routes back to the earlier state: the unchanged stream and the
nonlinear branch. Their linear responses add. Traversing the composition
from the measurement back to its inputs is the discrete adjoint method
for this layered dynamical system.

**Code role.** Residual block and whole-model forward/pullback closures,
parameter packing, and zeroed accumulation buffers. Verifier: `residual`,
`model`, `layout`; the end-to-end check differentiates every parameter
and, separately, every continuous input coordinate.

# Probability, geometry, and initial scales {#derivations-model-facts}

## 11. Softmax is a Gibbs distribution {#derivation-gibbs}

**Statement.** On a finite set $\{1,\ldots,V\}$ with energies $E_a=-z_a$,
inverse temperature $\beta>0$, and units $k_B=1$,
$$
\mathcal Z_\beta(E)=\sum_ae^{-\beta E_a},\qquad
p_a^{(\beta)}=\frac{e^{-\beta E_a}}{\mathcal Z_\beta(E)}
=\operatorname{softmax}(\beta z)_a.
$$
Temperature is $\tau=1/\beta$; the model's training softmax uses $\beta=1$.
For fixed finite energies,
$$
\lim_{\beta\to0^+}p_a^{(\beta)}=\frac1V,\qquad
\lim_{\beta\to\infty}p_a^{(\beta)}=
\begin{cases}1/|G|,&a\in G,\\0,&a\notin G,\end{cases}
$$
where $G=\arg\min_aE_a=\arg\max_az_a$. In particular the zero-temperature
limit is uniform over tied ground states, not an arbitrary tie-break.
For any $c\in\mathbb R$,
$\operatorname{softmax}(\beta(z+c\mathbf1))=
\operatorname{softmax}(\beta z)$.

**Proof.** Normalisation is the finite partition sum. As $\beta\to0$,
every exponential tends to $1$. For the other limit subtract the ground
energy $E_*$. Each weight becomes $e^{-\beta(E_a-E_*)}$, which tends to
$1$ on $G$ and $0$ elsewhere. Finiteness permits termwise limits in the
denominator, giving $|G|$. A common logit shift multiplies numerator and
denominator by $e^{\beta c}$ and cancels.

**Free energy and the sign of its gradient.** Define
$$
\mathcal F_\beta(E)=-\frac1\beta\log\mathcal Z_\beta(E),\qquad
\operatorname{LSE}_\beta(z)=\frac1\beta\log\sum_ae^{\beta z_a}.
$$
They obey $\mathcal F_\beta(-z)=-\operatorname{LSE}_\beta(z)$, and direct
differentiation gives
$$
\boxed{\frac{\partial\mathcal F_\beta}{\partial E_a}=p_a^{(\beta)},
\qquad \frac{\partial\operatorname{LSE}_\beta}{\partial z_a}
=p_a^{(\beta)},\qquad
\frac{\partial\mathcal F_\beta(-z)}{\partial z_a}=-p_a^{(\beta)}.}
$$
This distinction is essential: calling a positive log-sum-exp itself
the free energy loses a minus sign with the stated energies $-z$.
Also $Dp^{(\beta)}=\beta(\operatorname{diag}(p)-pp^\top)$.

For a distribution $r$, write
$H(r)=-\sum_ar_a\log r_a$ and $0\log0=0$. Substituting
$\log p_a^{(\beta)}=-\beta E_a-\log\mathcal Z_\beta$ yields
$$
\sum_ar_aE_a-\beta^{-1}H(r)
=\mathcal F_\beta(E)+\beta^{-1}\operatorname{KL}(r\|p^{(\beta)}).
$$
Nonnegativity of KL (proved next) shows that the Gibbs distribution is
the unique equilibrium minimiser of mean energy minus temperature times
entropy. Equivalently,
$\operatorname{LSE}_\beta(z)=\max_r\{\langle r,z\rangle+H(r)/\beta\}$.
The elementary bound
$\max_a z_a\le\operatorname{LSE}_\beta(z)
\le\max_a z_a+\log V/\beta$ follows by factoring out the largest
exponential. It proves that log-sum-exp tends to the maximum at zero
temperature, and free energy tends to the ground energy.

**Intuition for a physicist.** The logits really are negative energies.
Sampling temperature is the physical trade-off between selecting a low
energy and occupying many states. A uniform energy shift changes free
energy by the same shift but cannot change occupations. The familiar
partition-function derivatives therefore give the exact learning formulas.

**Code role.** Vocabulary softmax, numerically stable log-sum-exp, and
temperature-controlled sampling. Verifier: softmax shift and temperature
limits, log-partition and free-energy gradients.

## 12. Cross-entropy, KL, and maximum likelihood {#derivation-cross-entropy}

**Statement.** For distributions $p,q$ on the vocabulary, define
$$
H(p,q)=-\sum_{a:p_a>0}p_a\log q_a,\qquad
\operatorname{KL}(p\|q)=\sum_{a:p_a>0}p_a\log\frac{p_a}{q_a}.
$$
If $p_a>0$ and $q_a=0$, both are $+\infty$. Otherwise,
$$
\boxed{H(p,q)=H(p)+\operatorname{KL}(p\|q)\ge H(p),}
$$
with equality, and hence the minimum over the probability simplex,
exactly at $q=p$. With a uniform prediction $q_a=1/V$, every target
incurs loss $\log V$ nats.

**Proof.** Add and subtract $-\sum_ap_a\log p_a$ to obtain the identity.
For $q$ positive on the support of $p$, the inequality
$-\log t\ge1-t$ gives
$$
\operatorname{KL}(p\|q)
\ge\sum_{a:p_a>0}p_a\left(1-\frac{q_a}{p_a}\right)
=1-\sum_{a:p_a>0}q_a\ge0.
$$
Equality requires $q_a/p_a=1$ on the support and no mass off it, so
$q=p$. The divergent case cannot minimise. The uniform baseline follows
from $-\log(1/V)=\log V$. There is no claim that loss cannot exceed
$\log V$: confident wrong predictions can have arbitrarily large loss.
If $p$ has zero entries, the softmax family with finite logits cannot
attain $p$ exactly; the simplex minimum is then only an infimum within
that strictly positive family. A single one-hot target is the simplest
example: zero loss is approached by diverging logit gaps.

For observations $(c_n,y_n)$, where $c_n$ is the supplied context, the
conditional likelihood is
$$
\mathscr L(\theta)=\prod_{n=1}^Nq_\theta(y_n\mid c_n),\qquad
-\frac1N\log\mathscr L(\theta)
=-\frac1N\sum_n\log q_\theta(y_n\mid c_n).
$$
Thus minimising mean cross-entropy is equivalent to maximising this
likelihood: the logarithm is strictly increasing and multiplication by
$-1/N$ reverses the order. More generally the expectation over a true
conditional distribution has the entropy-plus-KL decomposition separately
for each context; the true entropy term is independent of $\theta$.
Overlapping training windows reuse observations. Their objective is still
the stated average conditional negative log-likelihood, but must not be
identified with the joint likelihood of one corpus with every factor used
exactly once. Windowing also changes the contexts supplied to the factors.

**Intuition for a physicist.** Cross-entropy is the mean surprise under
the actual occupation statistics. Its irreducible part is the entropy;
the extra cost is the discrepancy between actual and predicted occupations.
Likelihood multiplies probabilities of the observed outcomes, and taking
its logarithm turns that product into the additive training energy.

**Code role.** Mean target loss, the uniform-guessing reference line,
evaluation, and the objective used by both teaching and training optimisers.
Verifier: entropy/KL identity, nonnegativity, and uniform-loss check.

## 13. Probability chain rule and autoregression {#derivation-autoregression}

**Statement.** For finite-valued random tokens $x_1,\ldots,x_m$,
$$
P(x_1,\ldots,x_m)=P(x_1)\prod_{t=1}^{m-1}P(x_{t+1}\mid x_1,\ldots,x_t).
$$
Replacing each conditional by a normalised model distribution defines
an autoregressive joint distribution once an initial distribution or
fixed prompt is specified. A finite-context model uses only the most
recent $T$ available tokens, with the position convention fixed by the
implementation.

**Proof.** The definition of conditional probability gives
$P(x_{1:t+1})=P(x_{t+1}\mid x_{1:t})P(x_{1:t})$ whenever the prefix
probability is positive. Iterating proves the product. On zero-probability
prefixes conditional distributions can be assigned arbitrarily; the
preceding zero factor makes the joint zero. Conversely any family of
normalised conditional kernels gives a normalised joint: summing the
product over $x_m$, then $x_{m-1}$, and so on removes one factor at a time.
Sampling one factor at a time therefore samples that joint. Dropping
old context is a modelling restriction, not a consequence of the chain rule.

At output column $t$, the input token is $x_t$ and its target is $x_{t+1}$.
For a length-$T$ training window one needs $T+1$ corpus characters to
form all $T$ shifted targets. The first output is allowed to see $x_1$;
it is not meant to predict $x_1$ from itself.

**Intuition for a physicist.** A joint law can be specified by successive
conditional experiments. No independence is required. Autoregressive
generation performs those experiments in temporal order, while the
finite context is an imposed truncation of the system's memory.

**Code role.** Shifted input/target windows, next-token distributions,
context truncation, and token-by-token sampling.

## 14. Why divide the score by sqrt(d)? {#derivation-attention-scale}

**Statement.** Suppose $q_1,\ldots,q_d,k_1,\ldots,k_d$ are independent,
each centred with unit variance. Then
$$
\mathbb E[q^\top k]=0,\qquad
\operatorname{Var}(q^\top k)=d,\qquad
\operatorname{Var}\!\left(\frac{q^\top k}{\sqrt d}\right)=1.
$$

**Proof.** Independence and centring give $\mathbb E[q_rk_r]=0$.
For $r\ne s$, $\mathbb E[q_rk_rq_sk_s]=0$; for $r=s$ it is
$\mathbb E[q_r^2]\mathbb E[k_r^2]=1$. Expanding the squared sum gives $d$.
More generally, independent centred vectors with covariances
$\Sigma_q,\Sigma_k$ obey
$\mathbb E[(q^\top k)^2]=\operatorname{tr}(\Sigma_q\Sigma_k)$,
by expanding indices and factoring expectations between the two vectors.
The head-width formula is the isotropic unit-covariance case. If the
component variances are $v_q,v_k$, it becomes $dv_qv_k$.

Queries and keys in the actual model need not remain independent or
unit-variance, especially after training; this is a scale-setting argument.
Without the factor, increasing $d$ alone makes typical scores grow like
$\sqrt d$, corresponding to cooling the softmax and often concentrating
it prematurely.

**Intuition for a physicist.** A dot product is a sum of fluctuating
contributions. Independent fluctuations add in variance, so their typical
sum is proportional to the square root of the number of terms. Dividing
by that factor keeps the effective temperature comparable across widths.

**Code role.** Score construction and both query/key pullbacks.
Verifier: exact enumeration of independent $\pm1$ entries for $d=1,2,3,4$,
avoiding Monte Carlo uncertainty in the variance check.

## 15. The causal mask and the exact causality theorem {#derivation-causality}

**Statement.** For finite allowed scores, the masked attention matrix has
$A_{ij}\ge0$, $A_{ij}=0$ for $j>i$, and $\sum_jA_{ij}=1$.
It is therefore lower-triangular and row-stochastic, with strictly
positive entries on every allowed position in exact arithmetic.

For fixed parameters, fixed positions, and one sequence, output column
$t$ of the full model is a function only of input columns $1,\ldots,t$.
Precisely: two sequences with identical input prefix through $t$ and the
same positional codes through $t$ have identical logits through $t$.
Other sequences in the batch have no effect. At differentiable inputs,
$\partial Z_{:,t}/\partial X_{0,:,s}=0$ for $s>t$.
This is a statement about dependence on data with parameters held fixed;
training on a later token can change shared parameters and hence earlier
predictions in the next forward pass.

**Proof.** The explicit prefix-normalisation formula in Result 8 proves
triangularity, nonnegativity, and unit row sums. Every row has at least
one finite score (its diagonal), so no empty softmax is required.
All input embedding and position operations are columnwise.
Assume the stream at some stage is prefix-causal. LayerNorm, affine maps,
and ReLU act on individual columns, so preserve this property. The
query at $i$ depends only on inputs through $i$. Each contributing key
and value is at $j\le i$ and depends only on inputs through $j$, hence
through $i$. The row's normalisation also sums only over those sources.
Its output therefore depends only on the prefix through $i$.
Concatenation, output projection, and residual addition preserve the
property. Induction over sublayers and blocks, then final LayerNorm and
projection, proves the theorem. The same reasoning proves batch isolation.
The derivative statement follows because changing a later column leaves
the earlier output function constant; the functional statement remains
valid even at ReLU kinks.

The orientation matters: because values are columns, the mixing is
$V_hA^\top$, not $V_hA$. A batch must use separate $T\times T$ attention
matrices (or an equivalent block-diagonal mask), never a single causal
$N\times N$ mask that allows one sequence to read another.

**Intuition for a physicist.** The mask is a discrete domain-of-dependence
condition. Local operations cannot enlarge it, and attention enlarges it
only towards the past. Repeated composition preserves this direction of
influence, just as composing causal propagation steps preserves causality.

**Code role.** Causal mask, rowwise softmax, per-sequence head iteration,
and the input/target offset. Verifier: triangularity, row sums, zero masked
cotangents, and full-model prefix and batch-isolation perturbations.

## 16. Sinusoidal positions carry translations as rotations {#derivation-positions}

**Statement.** For even $D$, define $\omega_r=10000^{-2r/D}$ for
$r=0,\ldots,D/2-1$, and let $P_t\in\mathbb R^D$ have paired entries
$$
(P_t)_{2r+1}=\sin(\omega_rt),\qquad
(P_t)_{2r+2}=\cos(\omega_rt).
$$
For any real $t,k$, in particular integer positions and offsets,
$$
\boxed{P_{t+k}=R(k)P_t,\quad
R(k)=\bigoplus_{r=0}^{D/2-1}
\begin{pmatrix}
\cos(\omega_rk)&\sin(\omega_rk)\\
-\sin(\omega_rk)&\cos(\omega_rk)
\end{pmatrix}.}
$$
These blocks are rotations by $-\omega_rk$ in the usual coordinate
orientation because the coordinates are ordered sine, cosine.
$R(k)^\top R(k)=I$, $R(k+\ell)=R(k)R(\ell)$, and $R(0)=I$.

**Proof.** The angle-addition identities give
$\sin(\omega_r(t+k))=\cos(\omega_rk)\sin(\omega_rt)
+\sin(\omega_rk)\cos(\omega_rt)$ and
$\cos(\omega_r(t+k))=-\sin(\omega_rk)\sin(\omega_rt)
+\cos(\omega_rk)\cos(\omega_rt)$. This is exactly the displayed block
action. Orthogonality and the group law follow either by block
multiplication or by the same angle-addition identities. Furthermore,
$$
P_s^\top P_t=\sum_{r=0}^{D/2-1}\cos(\omega_r(t-s)),
$$
by $\sin a\sin b+\cos a\cos b=\cos(a-b)$.
Thus each fixed relative offset is represented by a linear map independent
of the absolute position, and the unprojected positional inner product
depends only on relative position.

This does not imply that learned attention is translation invariant:
arbitrary $W_q,W_k$, content-position cross terms, boundaries, and
context resets can retain absolute-position dependence. Odd $D$ with
an unpaired sine does not have this exact rotation property within the
same $D$ coordinates; use even $D$ or specify a different construction.

**Intuition for a physicist.** Positions are Fourier phases. Translation
advances each oscillator by a fixed phase, so it acts linearly as a
rotation on the corresponding sine/cosine plane. Relative displacement
is therefore encoded by a group action rather than by unrelated labels.

**Code role.** Fixed positional-code construction and its addition at
embedding time; position origin must agree in Julia and JavaScript.
Verifier: `positions`, `rotations`, negative and positive offsets, group
law and orthogonality for $D=2,8,128$.

## 17. The residual stream is an Euler discretisation {#derivation-residual-flow}

**Statement.** Explicit Euler for $\dot x=f(s,x)$ on a depth grid
$s_{k+1}=s_k+h$ is
$x_{k+1}=x_k+h f(s_k,x_k)$. A residual sublayer
$X_{k+1}=X_k+f_k(X_k)$ is exactly an Euler-form update with $h=1$ and
a depth-dependent vector field, or any chosen $h$ after absorbing $h$
into the branch definition. A block contains two such updates.

**Derivation.** The integral equation is
$x(s+h)-x(s)=\int_s^{s+h}f(r,x(r))\,dr$.
Replacing the integrand by its left-endpoint value gives Euler. If
$f$ is sufficiently smooth, Taylor expansion gives a one-step local
error $O(h^2)$; under standard Lipschitz bounds over a fixed interval,
the accumulated global error is $O(h)$. The architecture alone supplies
neither a common smooth vector field nor a small $h$, so it does not
by itself assert convergence to any continuum ODE.
Linearising the residual update gives
$dX_{k+1}=(I+Df_k)dX_k$, whose adjoint is precisely
$\bar X_k=\bar X_{k+1}+Df_k^*\bar X_{k+1}$.
The identity term supplies a direct gradient route; it does not guarantee
absence of exploding or vanishing gradients in the product of these maps.

**Intuition for a physicist.** Depth plays the role of time and a sublayer
adds an increment to the current state. The residual stream is the state
being evolved, rather than a state discarded and rebuilt at every step.
The backward equations are the discrete adjoint of that evolution.

**Code role.** The two residual additions per block, and their identity
cotangent paths. Verifier: `residual` and the two-block composition.

## 18. Initialisation and variance preservation {#derivation-initialisation}

**Statement.** Let $y=Wx$ with fan-in $n$, independent weights of mean
zero and variance $s^2$, independent of $x$. Conditional on $x$,
$$
\mathbb E_W[y_i\mid x]=0,\quad
\operatorname{Var}_W(y_i\mid x)=s^2\|x\|^2.
$$
If $\mathbb E[x_j^2]=v$, then $\operatorname{Var}(y_i)=ns^2v$.
Thus $s^2=1/n$ preserves the input second-moment scale through a linear
map. Centred inputs make their second moment equal to their variance;
without centring it is the second moment that enters the formula.

**Proof.** Expand $y_i=\sum_jW_{ij}x_j$. Zero mean removes all cross terms
in $\mathbb E_W[y_i^2\mid x]$, leaving $s^2\sum_jx_j^2$.
Taking expectations over $x$ proves the formula. Independent input
components are not needed for this argument. Backward propagation
$\bar x=W^\top\bar y$ has the analogous fan-out scale under the
corresponding independence approximation. The compromise
$s^2=2/(n+m)$ for fan-out $m$ is often called Xavier initialisation;
it balances, rather than exactly preserves, both scales for a nonsquare map.

If the preactivation law is symmetric,
$\mathbb E[\operatorname{relu}(y_i)^2]=\tfrac12\mathbb E[y_i^2]$.
This follows by symmetry of the positive and negative half-axes.
Taking $s^2=2/n$ before a ReLU therefore preserves its output second
moment (He initialisation). For a centred Gaussian of variance $v$,
the ReLU's **variance** is $v(1/2-1/(2\pi))$, because its mean is
$\sqrt{v/(2\pi)}$. Calling $v/2$ its variance would be incorrect.

**Residual-depth scaling.** There are $2L$ residual branches. In one
coordinate write a simplified initialisation model as
$$
X_K=X_0+\alpha\sum_{k=1}^Kr_k,\qquad K=2L.
$$
If $\operatorname{Cov}(X_0,r_k)=0$, the $r_k$ are mutually uncorrelated,
and $\operatorname{Var}(r_k)=c$, then exactly
$\operatorname{Var}(X_K)=\operatorname{Var}(X_0)+K\alpha^2c$.
Setting $\alpha=1/\sqrt{2L}$ makes the accumulated branch variance $c$
instead of $2Lc$. Since each branch's final projection enters linearly,
this is implemented by scaling the initial $W_o$ and $W_2$ by
$1/\sqrt{2L}$ relative to their base standard deviations (variance
factor $1/(2L)$), with initial branch biases zero.
Equivalently, for well-scaled branch inputs one may use residual
projection variances of order $1/(2LD)$ and $1/(2LF)$ respectively.
The base scale still depends on the chosen preactivation second moments.

These independence/covariance assumptions are approximations in the
actual nonlinear stack, where $r_k$ depends on earlier states. Independent
zero-mean final projections and zero biases can justify zero conditional
means and zero cross covariances at initialisation when the unscaled
branch variance stays bounded; pre-LayerNorm helps maintain that scale.
The brief does not prescribe a unique base weight distribution, so do
not claim this exact model mandates a particular Gaussian standard
deviation, or that the scaling controls trained variances exactly.
When $L=0$ there are no residual projections to scale.

**Intuition for a physicist.** A linear map sums many fluctuating inputs;
its coupling strengths should shrink with the square root of fan-in.
Residual increments likewise accumulate like random-walk steps under
weak-correlation assumptions. Shrinking their size with the square root
of the number of steps prevents the depth alone from setting a large
fluctuation scale. Coherent increments would add differently, which is
why the assumptions must remain visible.

**Code role.** Parameter initialiser, fan-in/fan-out scales, zero biases,
unit LayerNorm scales and zero offsets, and special scaling of $W_o,W_2$.
These are author recommendations where §3 leaves initialisation open,
not additional binding model semantics. Verifier: exact finite sign
ensembles for conditional forward variance, ReLU second moment, and
the scaled sum of independent residual increments.

# Learning as numerical dynamics {#derivations-learning}

## 19. Gradient descent, gradient flow, and the descent lemma {#derivation-gradient-flow}

**Statement.** Euclidean gradient flow for a differentiable loss is
$\dot\theta=-\nabla\mathcal L(\theta)$. Along a differentiable solution,
$$
\frac{d}{ds}\mathcal L(\theta(s))=-\|\nabla\mathcal L(\theta(s))\|^2\le0.
$$
Explicit Euler with step size $\eta$ is gradient descent:
$$
\boxed{\theta_{k+1}=\theta_k-\eta\nabla\mathcal L(\theta_k).}
$$
If the gradient is $\Lambda$-Lipschitz on the line segment in question
(the usual phrase is “$L$-smooth”, but $L$ already counts blocks), then
$$
\mathcal L(\theta+s)\le\mathcal L(\theta)
+\langle\nabla\mathcal L(\theta),s\rangle+\frac\Lambda2\|s\|^2.
$$
Consequently, for $g=\nabla\mathcal L(\theta)$,
$$
\boxed{\mathcal L(\theta-\eta g)
\le\mathcal L(\theta)-\eta(1-\Lambda\eta/2)\|g\|^2.}
$$
For $\Lambda>0$, $0<\eta<2/\Lambda$ guarantees strict descent whenever
$g\ne0$; the customary sufficient choice $0<\eta\le1/\Lambda$ gives a
decrease of at least $\eta\|g\|^2/2$. At $\eta=2/\Lambda$ the bound
guarantees only nonincrease. If $\Lambda=0$, the gradient is constant
on the relevant region and no reciprocal smoothness restriction is needed.

**Proof.** The flow identity is the chain rule. Euler follows by replacing
the flow velocity during one step by its initial value. For the lemma,
the fundamental theorem of calculus gives
$$
\mathcal L(\theta+s)-\mathcal L(\theta)-\langle g,s\rangle
=\int_0^1\langle\nabla\mathcal L(\theta+ts)-g,s\rangle\,dt
\le\int_0^1\Lambda t\|s\|^2\,dt.
$$
Insert $s=-\eta g$. No convexity is required. No general global
$\Lambda$ is supplied for this model: ReLU introduces kinks, and smoothness
claims need a region where the gradient exists and satisfies the bound.
Thus the theorem explains step-size constraints; it is not a guarantee
that every training step, SGD step, or Adam step lowers the displayed loss.

**Schedule.** The same update allows a prescribed $\eta_k$. One explicit
warmup-plus-cosine convention, to be shared by implementations, is
$$
\eta_k=\begin{cases}
\eta_{\max}k/W,&1\le k\le W,\\
\eta_{\min}+\frac{\eta_{\max}-\eta_{\min}}2
\left[1+\cos\left(\pi\frac{k-W}{K-W}\right)\right],&W<k\le K,
\end{cases}
$$
where $0<W<K$ and $K$ is the number of updates. It rises linearly to
$\eta_{\max}$ and falls to $\eta_{\min}$. These endpoint properties follow
from $\cos0=1$ and $\cos\pi=-1$. §3 requests the schedule but does not
fix indexing or endpoints; this is a proposed convention, not an
additional claim about its exact semantics.

**Intuition for a physicist.** Gradient flow is overdamped motion down
the loss landscape. Euler takes a finite displacement using the local
slope; excessive step length overshoots the curvature scale. The descent
lemma quantifies this competition between the favourable linear slope
term and the possible quadratic curvature cost. A learning-rate schedule
varies the integration step in this discrete evolution.

**Code role.** Teaching gradient-descent update, learning-rate schedule,
and the distinction between an update and a guaranteed decrease.
Verifier: descent bound on a quadratic with known largest curvature.

## 20. SGD is an unbiased estimator under specified sampling {#derivation-sgd}

**Statement.** For a finite training population of $M$ examples,
$\mathcal L(\theta)=M^{-1}\sum_{i=1}^M\ell_i(\theta)$.
Sample a batch of size $b$ uniformly with replacement, independently of
the current parameters conditionally on past history. Then
$$
\hat g(\theta)=\frac1b\sum_{j=1}^b\nabla\ell_{I_j}(\theta),\qquad
\boxed{\mathbb E[\hat g(\theta)\mid\theta]=\nabla\mathcal L(\theta).}
$$
Uniform sampling without replacement also gives this expectation for a
fresh uniform subset. If the samples are independent with replacement,
$\operatorname{Cov}(\hat g\mid\theta)=\Sigma(\theta)/b$, where
$\Sigma$ is the covariance of one example's gradient.

**Proof.** Finite sums permit differentiation term by term.
Linearity of expectation gives
$\mathbb E[\nabla\ell_I]=M^{-1}\sum_i\nabla\ell_i$ and then the batch
identity. Independence kills the cross covariances and leaves
$b\Sigma/b^2$. Without replacement the first identity follows from each
example having inclusion probability $b/M$; the covariance has the usual
finite-population correction. In nonuniform sampling with probabilities
$\pi_i>0$, use $\nabla\ell_I/(M\pi_I)$ to recover the uniform objective.

For this model an example may be a window with its mean token loss. If
all windows have length $T$, averaging $B$ such losses equals averaging
their $N=TB$ token losses. Overlap does not spoil unbiasedness for the
**chosen finite window objective**. During a random-permutation epoch,
later batches are uniform among remaining examples, so conditional
unbiasedness for the whole dataset given the entire training history
must not be asserted without qualification. Clipping and Adam transform
the estimator nonlinearly; their transformed directions are generally biased.

**Intuition for a physicist.** The full force is an average over many
observations. A uniformly chosen subset measures that force with noise;
independent averaging reduces the noise variance inversely with sample
count. The mean is right only for the sampling measure actually used.

**Code role.** Training-window sampler, minibatch loss averaging,
and SGD/gradient-descent training loop. Verifier: exact finite average
of per-example gradients of a least-squares objective.

## 21. Adam as a diagonal preconditioner {#derivation-adam}

**Statement.** For gradients $g_k$, decay constants
$0\le\beta_1,\beta_2<1$, $m_0=v_0=0$, and a separate numerical
$\epsilon_{\rm Adam}>0$,
$$
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\quad
v_k=\beta_2v_{k-1}+(1-\beta_2)(g_k\odot g_k),
$$
$$
\hat m_k=\frac{m_k}{1-\beta_1^k},\qquad
\hat v_k=\frac{v_k}{1-\beta_2^k},\qquad
\boxed{\theta_{k+1}=\theta_k-\eta_kG_k\hat m_k,\quad
G_k=\operatorname{diag}\!\left(
\frac1{\sqrt{\hat v_{k,i}}+\epsilon_{\rm Adam}}\right).}
$$
The square root and division are coordinatewise. This is diagonal
preconditioning of a momentum estimate, not generally of the current
gradient. $v_k$ estimates a **second moment**, not a centred variance.
Epsilon is outside the square root in this stated convention and is
distinct from LayerNorm epsilon.

**Derivation.** Expanding the recurrences gives
$m_k=(1-\beta_1)\sum_{j=1}^k\beta_1^{k-j}g_j$ and its squared-gradient
analogue for $v_k$. The geometric-series weights sum to $1-\beta_i^k$;
division makes each corrected estimate a weighted average with weights
summing to one. Under a stationary gradient law this removes the
zero-initialisation bias in the first and second moment estimates. It
does not make their nonlinear quotient unbiased. Rearranging the
elementwise update gives the displayed diagonal matrix.

For a fixed symmetric positive-definite $G$ and no momentum, the direction
$-Gg$ minimises
$\langle g,s\rangle+\tfrac12s^\top G^{-1}s$: differentiating in $s$
gives $g+G^{-1}s=0$. This identifies preconditioning as steepest descent
in a different local metric. Adam uses a changing diagonal metric and
momentum; the interpretation is algebraic, not a convergence theorem.

If $\hat v_{k,i}\approx g_{k,i}^2$, $\hat m_{k,i}\approx g_{k,i}$, and
$\epsilon_{\rm Adam}\ll|g_{k,i}|$, then
$$
(G_k\hat m_k)_i\approx\frac{g_{k,i}}{|g_{k,i}|}
=\operatorname{sign}(g_{k,i}).
$$
The momentum condition is necessary: a second moment close to $g_k^2$
alone does not imply sign descent if the momentum points elsewhere.
For an exactly constant gradient, bias correction yields
$\hat m_k=g$, $\hat v_k=g^2$ at every $k$, proving this limit directly.
For zero coordinates the regularised update is zero, not an undefined
division; finite epsilon softens the sign step near zero.

**Intuition for a physicist.** A diagonal mobility changes the speed of
motion independently along each coordinate. Adam estimates that mobility
from recent force magnitudes and smooths the force itself with memory.
When the estimated scale is just the current force magnitude, this
rescaling removes magnitude and retains direction alone.

**Code role.** Adam state buffers, bias correction, and elementwise
parameter update. Verifier: diagonal-preconditioner/sign-limit identity
and bias-corrected moments for an exactly constant gradient.

## 22. AdamW decay is decoupled from L2 regularisation {#derivation-adamw}

**Statement.** Adding $\tfrac\lambda2\|\theta\|^2$ to the loss changes
the raw gradient to $g+\lambda\theta$. Under plain SGD this gives
$$
\theta_{k+1}=(1-\eta_k\lambda)\theta_k-\eta_kg_k,
$$
so L2 regularisation and multiplicative weight decay coincide.
AdamW instead feeds only the data gradient $g_k$ into Adam's moment
estimates and applies shrinkage separately:
$$
\boxed{\theta_{k+1}=(1-\eta_k\lambda)\theta_k
-\eta_kG_k\hat m_k.}
$$
Both terms on the right use the old parameters. Applying a factor to
the already-updated parameters would introduce an extra cross term
and be a different convention.
Here $\lambda\ge0$; $0\le\eta_k\lambda\le1$ makes this factor an ordinary
nonnegative shrinkage. Arbitrarily large step sizes need not act as decay.

**Derivation.** The derivative of $\tfrac\lambda2\theta^\top\theta$ is
$\lambda\theta$, proving the SGD identity. For a fixed diagonal
preconditioner $G$, coupled regularisation gives a decay term
$-\eta\lambda G\theta$, whereas decoupled decay gives
$-\eta\lambda\theta$. Their difference (decoupled minus coupled) is
$\eta\lambda(G-I)\theta$. In actual Adam, including $\lambda\theta$ in
the raw gradient also changes both moment histories, making the
difference larger than this fixed-$G$ comparison. The isolated decay
step is Euler for $\dot\theta=-\lambda\theta$; exact flow over a time
$\eta$ would multiply by $e^{-\lambda\eta}$ instead.

The brief leaves decay exclusions unspecified. If biases and LayerNorm
parameters are excluded, replace $\lambda\theta$ in the decay step by
$\lambda C\theta$, with a fixed diagonal selector $C$ of zeros and ones,
and record the selector in the training contract.

**Intuition for a physicist.** L2 adds a harmonic restoring force before
the mobility acts; diagonal mobility then makes that force anisotropic.
Decoupled decay shrinks all selected coordinates directly at the same
relative rate. The two operations agree for ordinary gradient descent
and cease to agree when coordinate-dependent mobility intervenes.

**Code role.** AdamW update, parameter-group decay policy, and optimiser
state updates. Verifier: fixed-preconditioner difference identity.

## 23. Global-norm clipping is Euclidean projection {#derivation-clipping}

**Statement.** For radius $c\ge0$ and concatenated parameter gradient
$g\in\mathbb R^p$, projection onto the closed Euclidean ball is
$$
\boxed{\operatorname{clip}_c(g)
=\arg\min_{\|v\|\le c}\frac12\|v-g\|^2
=\begin{cases}g,&\|g\|\le c,\\
c\,g/\|g\|,&\|g\|>c.
\end{cases}}
$$
The norm is global over all parameter arrays, not taken separately for
each array or coordinate. At $g=0$ the first branch avoids division by
zero; $c=0$ returns zero.

**Proof.** If $g$ is feasible, zero distance is optimal. Otherwise for
any feasible $v$, the reverse triangle inequality gives
$\|g-v\|\ge\|g\|-\|v\|\ge\|g\|-c$. Taking $v=cg/\|g\|$ attains
this bound. Strict convexity of squared distance on a convex set gives
uniqueness. Equivalently the Lagrange multiplier rescales $g$ radially.

Clipping preserves direction when it acts and bounds magnitude. It
does not guarantee descent or preserve unbiasedness of an SGD gradient.
With Adam it is normally applied before the moment update; clipping
the parameter update instead is a distinct operation. §3 does not fix
the order, so that detail must be made explicit in the implementation.

**Intuition for a physicist.** Of all forces with an allowed magnitude,
choose the one nearest to the measured force. Rotating it would waste
distance, so only its radial component is shortened. It is a spherical
constraint on force, not a coordinatewise saturation.

**Code role.** Concatenated gradient norm reduction and a common scaling
factor applied to every gradient buffer before the optimiser.
Verifier: ball membership, direction, and the projection variational
inequality at an interior comparison point.

## 24. Central differences, rounding, and the cube-root step {#derivation-gradcheck}

**Statement.** For a scalar smooth function of one coordinate,
$$
\frac{f(x+h)-f(x-h)}{2h}=f'(x)+O(h^2).
$$
If computed function values have absolute rounding errors bounded by
$O(uF_*)$, where $u$ is machine roundoff and $F_*$ a local function
evaluation scale, then the derivative error has the form
$$
\boxed{O(M_3h^2)+O(uF_*/h),}
$$
where $M_3$ bounds the third derivative locally. Balancing terms gives
$h\asymp(uF_*/M_3)^{1/3}$, or $h\asymp u^{1/3}$ in well-scaled units.
This is a scale law, not an exact universally optimal perturbation.

**Proof.** Taylor's formula with bounded third derivative gives
$f(x\pm h)=f(x)\pm hf'(x)+\tfrac12h^2f''(x)+O(h^3)$.
Subtracting cancels the even terms and division by $2h$ gives $O(h^2)$.
More precisely the truncation bound is $M_3h^2/6$; if five derivatives
exist the expansion is $f'(x)+h^2f'''(x)/6+O(h^4)$.
If each evaluation has absolute error at most $a uF_*$, subtraction
and division can contribute about $a uF_*/h$, plus lower-order
arithmetic terms. Minimising an upper-bound model
$Ah^2+Bu/h$ gives $2Ah-Bu/h^2=0$, hence
$h=(Bu/(2A))^{1/3}$. Roundoff in the represented input also matters;
perturbations must be large enough that $x\pm h$ differ from $x$.
As $h\to0$ the truncation term improves while cancellation amplifies
roundoff, explaining why ever-smaller perturbations eventually worsen
the check. Exact cancellation of $f'''$ or evaluation errors can alter
the optimal scale in special cases.

For a map $F$ and a chosen cotangent $\bar y$, finite-difference the scalar
$\phi(x)=\langle\bar y,F(x)\rangle$ and compare each coordinate with
$DF(x)^\top\bar y$. This tests the pullback directly. Test parameter
arguments separately, including broadcast biases and shared embeddings.
At ReLU kinks there is no smooth derivative to check; crossing a kink
invalidates the Taylor error model. Near constant LayerNorm inputs the
local scale is $\sqrt\varepsilon$, so a smaller perturbation may be
needed even with Float64 arithmetic. Loss shifts can also worsen the
absolute evaluation scale without changing the true derivative.

**Intuition for a physicist.** A symmetric perturbation experiment cancels
the leading even response and isolates linear response with a quadratic
truncation error. But it subtracts two nearly equal measured numbers,
amplifying their finite precision. The useful perturbation balances those
two limitations; zero perturbation is not a more accurate experiment.

**Code role.** Generic coordinate gradient checker and the independent
verification script's `finite_gradient`, `check_gradient`, `check_pullback`.
The script uses `eps(Float64)^(1/3)` times a coordinate scale; machine
epsilon is twice the usual round-to-nearest unit roundoff, changing only
an order-one factor. It checks quadratic convergence before roundoff
dominates on a smooth scalar example.

# Parameter count and executable evidence {#derivations-evidence}

## 25. The exact parameter count {#derivation-parameter-count}

**Statement.** With independent embedding and output weights, all biases
present, two affine LayerNorms per block, and a final affine LayerNorm,
$$
\boxed{p=\dim\theta=2DV+V+2D+L(4D^2+2DF+9D+F).}
$$
For $F=4D$ this is $p=2DV+V+2D+L(12D^2+13D)$.
Neither context length $T$ nor number of heads $H$ contributes parameters
at fixed $D$; the sinusoidal positions and the mask are fixed data.

**Derivation.** Count independent entries in precisely the binding order.

| Part | Arrays | Entries |
|:--|:--|--:|
| Embedding | $E:D\times V$ | $DV$ |
| Block LN$_1$ | $\gamma_1,\beta_1:D$ | $2D$ |
| Block attention | $W_q,W_k,W_v,W_o:D\times D$; four $D$-biases | $4D^2+4D$ |
| Block LN$_2$ | $\gamma_2,\beta_2:D$ | $2D$ |
| Block MLP | $W_1:F\times D$, $b_1:F$, $W_2:D\times F$, $b_2:D$ | $2DF+F+D$ |
| Final LN | $\gamma_f,\beta_f:D$ | $2D$ |
| Output | $W_{\rm out}:V\times D$, $b_{\rm out}:V$ | $VD+V$ |

Adding one block's rows gives $4D^2+2DF+9D+F$; adding the outer rows
gives $2DV+V+2D$. Flattening in column-major order changes no entry
counts. The number of heads only partitions already-counted $D$ rows.

For the specimen $(V,D,F,L)=(7,8,32,1)$,
$$
p=2(8)(7)+7+2(8)+[4(8)^2+2(8)(32)+9(8)+32]
=135+872=\boxed{1007}.
$$
For $(V,D,F,L)=(65,128,512,4)$,
$$
p=16640+65+256+4[65536+131072+1152+512]
=16961+4(198272)=\boxed{810049}.
$$
These are formula evaluations for the notes, not constants to hard-code
in the model. If weights were tied, biases removed, or LayerNorm scales
fixed, the count would change; none of those changes is in §3.

**Intuition for a physicist.** Count coordinates of the parameter space,
not work performed by a forward pass. Reusing the same linear map at
many times increases computation but introduces no new couplings.
Splitting a fixed-width space into heads likewise creates no coordinates.

**Code role.** Parameter-layout builder, contiguous slices of $\theta$,
and printed model summary. Verifier: `layout` enumerates actual array
shapes independently of the polynomial `parameter_count`; both
configurations and additional edge cases are compared.

## 26. Reproducible numerical verification {#derivation-verification}

Run from the repository root:

```sh
julia --startup-file=no book/notes/verify_derivations.jl
```

Only `LinearAlgebra`, `Random`, `Statistics`, `Test`, and `Printf` from
the Julia standard library are used. The random fixture generator is a
local `MersenneTwister(20261104)` and arithmetic is Float64. This is an
independent mathematical reference, not a test of the concurrently
rewritten `src/` implementation or Julia/JavaScript trace conformance.
It implements each analytic pullback explicitly; a small local tape
composes those pullbacks to verify the complete model. No automatic
derivative generator or external package is used.

For each local operation, an arbitrary output cotangent defines a scalar
objective. Central differences check **every input coordinate** against
the claimed pullback, separately for every differentiable argument.
The reported scaled error is
$$
\frac{\|g_{\rm FD}-g_{\rm analytic}\|_\infty}
{\max(1,\|g_{\rm FD}\|_\infty,\|g_{\rm analytic}\|_\infty)}.
$$
The tolerance is $2\cdot10^{-7}$. The unit floor makes this an absolute
comparison for small derivatives and a relative comparison for large
ones; it is not a coordinatewise relative error. The generic base step
is $6.05545445\cdot10^{-6}$ times $\max(1,|x_i|)$.
For constant-column LayerNorm, that step is additionally multiplied by
$\sqrt{10^{-5}}$ to resolve its local regularisation scale. This keeps
the same tolerance and avoids mistaking truncation error for a bad pullback.

The end-to-end fixture has $(V,D,F,L,H,T,B)=(4,4,7,2,2,3,2)$ and
370 independent parameters. The nonstandard $F=7$ intentionally checks
that formulas do not accidentally assume $F=4D$ internally; the two
requested model counts use $F=4D$. Repeated IDs check scatter-add,
two sequences check isolation, and two blocks check repeated residual
fan-out. Finite differences also check its 24 continuous embedding-input
coordinates. Hidden biases keep these fixtures away from ReLU kinks.

**Recorded results (2026-11-04).** Run with Julia 1.13.3; all 143 assertions
passed, with no warnings. The testset summaries were:

| Testset | Passed | Total |
|:--|--:|--:|
| Primitive and composed pullbacks | 37 | 37 |
| Geometric and probabilistic identities | 27 | 27 |
| Sinusoidal group action | 27 | 27 |
| Layout and full exact model | 17 | 17 |
| Optimisation and variance identities | 35 | 35 |
| **Total** | **143** | **143** |

The 40 coordinate finite-difference comparisons covered 762 coordinates.
Selected errors from the script's printed output are:

| Comparison | Scaled maximum error |
|:--|--:|
| Affine map, weights | $5.910\cdot10^{-11}$ |
| Embedding with repeated IDs | $1.346\cdot10^{-11}$ |
| Masked row softmax | $4.490\cdot10^{-11}$ |
| Mean log-sum-exp cross-entropy | $2.136\cdot10^{-11}$ |
| LayerNorm input, $\varepsilon=0$ | $4.692\cdot10^{-11}$ |
| LayerNorm input, $\varepsilon=10^{-5}$ | $1.694\cdot10^{-11}$ |
| Constant-column LayerNorm input | $6.083\cdot10^{-9}$ |
| Attention head, queries | $3.174\cdot10^{-11}$ |
| Multi-head, two sequences, keys | $5.780\cdot10^{-11}$ |
| Residual fan-out, input | $2.801\cdot10^{-11}$ |
| Whole model, all 370 parameters | $5.274\cdot10^{-11}$ |
| Whole model, all 24 continuous inputs | $9.110\cdot10^{-11}$ |

The maximum across all comparisons was $6.083\cdot10^{-9}$, from the
constant-column LayerNorm input. No tolerance was relaxed for that case.
Explicit LayerNorm Jacobians agreed with the compact formula at all
three tested epsilon values, and the pure sphere formula agreed at zero
epsilon. Rotation, group-law, and orthogonality checks passed at widths
2, 8, and 128. Both requested parameter counts were obtained independently
by shape enumeration: **1007** and **810049**.

The algebraic checks also cover the KL decomposition, temperature limits
including tied maxima, the sign of the free-energy gradient, autoregressive
normalisation, exact score and initialisation variance identities,
residual random-walk scaling, the descent bound, unbiased finite-population
gradients, Adam bias correction, the sign limit, decoupled decay, clipping,
and the quadratic central-difference truncation rate. Full-model key-bias
invariance, zero key-bias gradients, prefix causality and batch isolation
all passed.

**What numerical checks do and do not prove.** Finite fixtures can expose
wrong transposes, reductions, masks, factors, and accumulation. They
cannot prove asymptotic complexity, arbitrary-input causality, unbiasedness
for every sampling policy, continuum convergence, or an optimal roundoff
step for all functions. Those claims are established by the derivations
and their explicit hypotheses above. Float64 sums may differ across
BLAS implementations in their last few bits; the tolerances allow for that.

**Intuition for a physicist.** The verifier is an independent perturbation
experiment on each response law, followed by an experiment on the composed
system. Algebra supplies the theorem; independent numerical response
measurements make sign, orientation, and normalisation mistakes visible.

**Code role.** This reference script and the future generic pullback
gradient tests. Use the binding parameter order and tensor orientations
when adapting these checks to the package implementation.

## 27. A structural surprise: key biases are unidentifiable {#derivation-key-bias}

**Statement.** Each key-projection bias $b_k$ in §3 has no effect on any
output for fixed other parameters. Its exact loss gradient is zero.
The parameter count nevertheless includes these $LD$ independent stored
coordinates because the binding architecture includes them.

**Proof.** In one head let $K_h=\widetilde K_h+b\mathbf1_T^\top$. Then
$$
S=\frac{Q_h^\top\widetilde K_h}{\sqrt d}
+\frac{(Q_h^\top b)\mathbf1_T^\top}{\sqrt d}.
$$
The second term is constant across the sources in each query row.
Restricting to an allowed causal prefix does not change that constancy,
so row-softmax shift invariance removes it. Thus $A$, $O_h$, and the
whole model are unchanged by arbitrary changes in $b$.
The backward formulas confirm this directly:
$$
\bar b=\bar K_h\mathbf1_T
=\frac{Q_h\bar S\mathbf1_T}{\sqrt d}=0,
$$
since the softmax score cotangent has zero row sums.
Sum over sequences and concatenate heads to obtain $\bar b_k=0$.
The verifier changes every block's key bias independently and checks
unchanged full logits and zero gradients.

There is also the familiar common-logit gauge
$b_{\rm out}\mapsto b_{\rm out}+c\mathbf1_V$: probabilities and loss
are unchanged although the recorded logits shift. The key-bias gauge is
stronger: even recorded logits are unchanged. Neither observation means
that subtracting $LD$ gives the full identifiable dimension; other
reparameterisation symmetries remain.

This is a small architectural inefficiency, not a contradiction. Keep the
biases and the count required by §3; tell the code and prose authors that
zero key-bias gradients are expected. With decoupled decay selected for
these entries, they may still shrink despite zero data gradient.

**Intuition for a physicist.** A query can detect only energy differences
between its allowed sources. Adding the same energy offset to every
source is a choice of zero, so the associated coupling cannot be measured
by the model's predictions. Its conjugate data response vanishes.

**Code role.** Key affine bias, softmax invariance, optimiser decay policy,
and parameter-layout count. Verifier: full-model key-bias shift and
zero-gradient assertions in the layout/model testset.
