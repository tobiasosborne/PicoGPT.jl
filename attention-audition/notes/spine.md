# Attention as an ansatz for the next-symbol conditional

## 1. Introduce an uncertain index

Fix a prefix $c=(a_1,\ldots,a_i)$ of positive probability. Write $Y$ for the next character, with $V$ possible values. At length $i$, an unrestricted conditional table has $V^i(V-1)$ free entries, one simplex per prefix. The chain rule tells us how to multiply its rows across time. It supplies no rule for filling them. References below by result number refer to [the verified derivations](../../book/notes/derivations.md).

Adjoin $J\in\{1,\ldots,i\}$. For **any** probability kernel $r(j\mid y,c)$, set $\tilde P(y,j\mid c)=P(y\mid c)r(j\mid y,c)$. Summing over $j$ recovers $P$. To recover $r$ from **any** extension, divide its joint probabilities by $P(y\mid c)$ wherever that is positive and choose $r$ arbitrarily on null outcomes. Write $\pi_j=\sum_y\tilde P(y,j\mid c)$ and $p_j(y)=\tilde P(y,j\mid c)/\pi_j$ for $\pi_j>0$. A zero-weight component can have any conditional. Exactly,
$$P(y\mid c)=\sum_{j\le i}\pi_j p_j(y).$$
Taking $r=1/i$ gives $\pi_j=1/i$ and $p_j(y)=P(y\mid c)$ for **every** $j$. Even the simplest choice of index can leave the entire original table to be learned.

**First assumption.** Some extension admits simple shared descriptions of *both* factors, or a mixture with such descriptions approximates the desired conditional. Existence of an unrestricted extension gives no such guarantee. Only under this hypothesis does “which position to consult” acquire a modelling meaning.

## 2. Use a shared rule on position vectors

In the specimen, $x_j=Ee_{a_j}+P_{:,j}$ and $\hat x_j=\mathrm{LN}_1(x_j;\gamma_1,\beta_1)$, including learned gain and offset. LayerNorm uses the features of this one column and $\varepsilon=10^{-5}$. In later blocks $x_j$ is the incoming residual column and may already depend on $a_1,\ldots,a_j$ through earlier attention. (Results 3, 7, 10, 15.)

For one head, write $v_j=W_v^{(h)}\hat x_j+b_v^{(h)}$. Choose one shared source-scoring rule on $(\hat x_i,\hat x_j)$, normalised over the permitted sources, and one shared probability map $g(x_i,v_j)\in\Delta^{V-1}$. This proposes $\sum_j\pi_j g(x_i,v_j)$. The reader enters the score through $\hat x_i$, can select itself because $j=i$ is allowed, and supplies its **unnormalised** $x_i$ to the downstream map via the residual addition. These are three distinct uses.

The assumption is sufficiency of the retained features for these shared rules. For example, for a given reader any two equal values give identical component predictions. If the desired components distinguish them, this choice of value map and readout cannot realise those components.

## 3. Choose the source energy and its domain

Every strictly positive row has Gibbs form at any $\tau>0$ by choosing $E_{ij}=-\tau\log\pi_j$ up to an additive row constant. Finite energies give strictly positive probabilities on the allowed set. PicoGPT chooses $E_{ij}=-q_i^\top k_j$ and $\tau=\sqrt d$. Suppress the head superscript on its two row blocks and biases. Expanding the negative energy gives
$$
q_i^\top k_j
=\underbrace{\hat x_i^\top W_q^\top W_k\hat x_j+b_q^\top W_k\hat x_j}_{\text{can vary with }j}
+\underbrace{(W_q\hat x_i+b_q)^\top b_k}_{\text{constant in row }i}.
$$
Thus $b_q$ can favour a source independently of reader content. Changing $b_k$ shifts the whole row, cancels out of its softmax, and leaves the exact loss gradient $\bar b_k=0$ (Results 5, 27). A proposed positive row is representable across contexts precisely when its log-odds between any allowed $j,k$ are $q_i^\top(k_j-k_k)/\sqrt d$ for shared maps. Arbitrary context-dependent odds can violate this condition. There is no distance penalty in the score. Position can affect either input through the initial positional code, and deeper columns also inherit information from earlier causal reads.

Under independent, centred unit-variance query and key coordinates, $\operatorname{Var}(q_i^\top k_j)=\sum_{r=1}^d\operatorname{Var}(q_{ir}k_{jr})=d$. Division by $\sqrt d$ gives unit variance under these hypotheses. This motivates the scale at initialisation. Neither score variance nor attention entropy is fixed by it after training. (Result 14.)

The range of $J$ gives the causal mask directly. With $S_{ij}=q_i^\top k_j/\sqrt d$, $A_{ij}=e^{S_{ij}}/\sum_{k\le i}e^{S_{ik}}$ for $j\le i$, and $A_{ij}=0$ for $j>i$. Each position is a reader. Its sources depend only on their own prefixes, and the other maps act columnwise. Induction over blocks gives a block lower-triangular input-to-logit Jacobian wherever differentiable. The mask discards $T(T-1)/2$ of the $T^2$ scores (28 of 64 in the specimen). One pass yields all $T$ model factors $p_\theta(a_{i+1}\mid a_{1:i})$ of the chain rule conditional on $a_1$ (Results 8, 13, 15).

## 4. Factor the bilinear interaction

$B_h=W_q^{(h)\top}W_k^{(h)}$ is $D\times D$ and factors through $\mathbb R^d$, so its rank is at most $d$. Conversely a rank factorisation of any matrix of rank $r\le d$ supplies two such factors after padding with zero rows. Symmetry is optional because the factors are freely chosen. Query and key are the two projected vectors whose dot product evaluates this interaction, with the affine offsets of §3. A higher-rank bilinear form on the full feature space cannot equal this form on all inputs. This rank bound is on $B_h$, before softmax.

The factors store $2dD$ weights per head against $D^2$ for a dense bilinear matrix. At $D=8,d=4,H=2$ the comparison is **64 against 64 per head**, or 128 against 128 across both. The constraint buys rank here with no reduction in stored matrix entries. The two affine projections also store $2d=8$ biases per head. The comparison of 64 with 64 counts the bilinear part alone (Result 25 gives the full architecture count).

## 5. Replace a selected value by its mean

Fix the reader and abbreviate $g_y(x_i,m)$ to $g_y(m)$. Set $\pi_j=A_{ij}$, $o_i=\sum_j\pi_jv_j$ and $\delta_j=v_j-o_i$. The mixture predicts $\sum_j\pi_jg_y(v_j)$. Attention uses $g_y(o_i)$, evaluating the readout once. Including $b_v^{(h)}$ shifts all values and the mean equally. “Mean field” here denotes this replacement of a random vector by its expectation.

For affine $g_y$ on the convex hull of the values equality follows from $\sum_j\pi_j\delta_j=0$. For twice continuously differentiable $g_y$, Taylor expansion with integral remainder gives the signed Jensen gap
$$
\sum_j\pi_jg_y(v_j)-g_y(o_i)=
\sum_j\pi_j\int_0^1(1-t)\,\delta_j^\top \nabla^2g_y(o_i+t\delta_j)\delta_j\,dt .
$$
If the Hessian norm is at most $K$ there, the absolute gap is at most $K s^2/2$, where $s^2=\sum_j\pi_j\|\delta_j\|^2$. Its sign can differ by character. With value diameter $R$, $s^2\le R^2$. If one value has weight $1-\eta$, then $s^2\le\eta R^2$, since the mean minimises the expected squared distance and that value is a candidate centre. Thus clustered values or a near-copy with bounded diameter suppress the gap. Separated values with appreciable weights can expose substantial curvature. Across a ReLU kink even the quadratic bound may fail (§6). Training fits $g(o_i)$ directly. The mixture at those same weights is a different predictor whose accuracy is an empirical question.

## 6. Verify the geometric mixture

Let $z(m)\in\mathbb R^V$ be the logits and $p_j=\operatorname{softmax}z(v_j)$. For **any** such readout the normalised weighted geometric mixture is exactly
$$
G_y=\frac{\prod_jp_j(y)^{\pi_j}}{\sum_a\prod_jp_j(a)^{\pi_j}}
=\operatorname{softmax}\!\left(\sum_j\pi_j z(v_j)\right)_y .
$$
Indeed $\log p_j(y)=z_y(v_j)-\log\sum_a e^{z_a(v_j)}$, and the subtracted term is common to all characters. Therefore $G=\operatorname{softmax}z(o_i)$ **if and only if** $z(o_i)-\sum_j\pi_jz(v_j)\in\operatorname{span}(\mathbf1_V)$, by taking pairwise log-odds. Affine logits satisfy this exactly. Finite source logits give positive probabilities throughout, so a literal single-source veto is impossible.

For the specimen let $m$ contain both head outputs, $u=x_i+W_om+b_o$, and $w=u+W_2\operatorname{ReLU}(W_1\mathrm{LN}_2(u)+b_1)+b_2$. Then $z(m)=W_{\rm out}\mathrm{LN}_f(w)+b_{\rm out}$. LayerNorm is smooth for positive $\varepsilon$ but generally nonlinear, even with all ReLU gates fixed (Results 4, 7, 10). On a smooth neighbourhood with bounded curvature, §5 applied to each logit gives $\mathbb E z(m)-z(\mathbb E m)=O(s^2)$, and the bounded softmax derivative preserves that order (Result 5). Thus the geometric and actual predictions agree through first order in a local smooth expansion. A gate boundary can give a first-order discrepancy. For equally weighted $m=\pm t$ and logits $(\max(0,m),0)$ the mean logit gap is $t/2$ while the logit gap at the mean is zero.

**Chapter sentence.** Averaging source values gives the normalised geometric mixture of their predictions when the readout logits are affine in those values, while PicoGPT’s nonlinear readout generally gives a different distribution.

## 7. Give each head its own index

In the proposed mixture choose conditionally independent $J_h$ with joint weights $\lambda_{\mathbf j}=\prod_{h=1}^H A^{(h)}_{ij_h}$ given $c$. Set $m_{\mathbf j}=\bigoplus_h v^{(h)}_{j_h}$. Linearity of expectation gives $\mathbb E m_{\mathbf J}=\bigoplus_h o_i^{(h)}$, with $o_i^{(h)}=\sum_{j\le i}A^{(h)}_{ij}v^{(h)}_j$. In matrix form $O_h=V_h A^{(h)\top}$ for value-column matrix $V_h$. Stacking the $d$ rows per head puts the result in $\mathbb R^D=\bigoplus_h\mathbb R^d$ for recombination by $W_o$ (Results 8–9). The actual forward pass uses only the marginals. Any coupling with those marginals yields the same mean, while a nonlinear per-tuple mixture can depend on the coupling. The product law is a choice for the comparison.

## 8. What each restriction replaces

| Restriction | Freedom it replaces |
|---|---|
| Simple selector and component law shared across prefixes | Arbitrary extension of the next-symbol table |
| Shared pair score on reader and source, with prefix normalisation | Source odds depending freely on the entire context |
| Bilinear score of rank at most $d$, plus the source-dependent term from $b_q$ | An arbitrary interaction on those features |
| Shared affine value map and common decoder using the reader | A separate transformation and prediction rule for each source and context |
| Product law for a multi-head mixture | Arbitrary correlations among selected indices given the context |
| Decode the vector of head means once | Retain the distribution of selected tuples and average their predictions |

## 9. Two comparisons with physics

**Matrix product state (MPS).** Across a cut, contracting the site matrices on either side gives an amplitude $\psi(u,v)=\sum_{r=1}^{\chi}L_r(u)R_r(v)$ for a prefix $u$ and suffix $v$. The bond dimension $\chi$ thus bounds the rank of the amplitude table across the cut. Attention keeps one width-$D$ column per observed position up to the context limit, and can address each by the same score rule. Its bound $d$ concerns a *pair score*. The stream represents one supplied sequence, so counting its coordinates gives no rank or correlation-length bound for the joint law of all sequences. The distillation’s blanket exponential-decay claim for Markov chains also fails: a stationary fair bit copied forever is Markov and has constant autocorrelation.

**One-particle analogy.** Exactly, stacking token columns gives $|X\rangle=\sum_j e_j\otimes x_j\in\mathbb R^T\otimes\mathbb R^D\cong\bigoplus_j\mathbb R^D$. Position and internal coordinates motivate the analogy. For fixed attention rows define $C_h=W_o^{(h)}W_v^{(h)}$, where $W_o^{(h)}$ is a **column** block. Applying the operator below gives precisely the source sum at each reader, since each row sums to one:
$$
|X_{\rm mid}\rangle=|X\rangle+
\left(\sum_h A^{(h)}\otimes C_h\right)|\hat X\rangle
+\mathbf1_T\otimes(b_o+W_ob_v).
$$
The displayed operator on the normalised stream has at most $H$ product terms, each $C_h$ of rank at most $d$. The residual uses the raw stream. Dropping biases alone cannot give $(I+K)|X\rangle$ in this pre-norm model. The attention rows also change with the input. Heads count terms in an input-dependent classical update. Calling them particles or treating the frozen operator as the full Jacobian exceeds the analogy.

## 10. What the specimen experiment measures

The proposed enumeration is well defined as written. At the trained parameters, fix each reader’s $x_i$, values and attention rows. The specimen has one block, so after substituting $m_{\mathbf j}$ all later operations in §6 are columnwise. The three predictions at that reader are
$$
p=\operatorname{softmax}z(\mathbb E m),\qquad
p^{\rm mix}=\mathbb E[\operatorname{softmax}z(m)],\qquad
p^{\rm geo}=\operatorname{softmax}(\mathbb E z(m)).
$$
Use the product law of §7. The first comparison measures the effect of decoding before or after taking the value mean. The comparison with $p^{\rm geo}$ isolates failure to commute averaging with the logits, modulo a common shift. Use averaged logits for a stable geometric mixture. A one-head substitution fixes the other head at its observed mean and measures the gap along that slice. In general the two slice gaps need not add to the joint gap.

Each head’s spread is its contribution to $\mathbb E\|m-\mathbb E m\|^2$ by the direct-sum norm. Row entropy measures dispersion of addresses and can be large even when all values coincide. Neither diagnostic alone determines the prediction gap. For $i=1$ all three predictions coincide exactly. The ordinary prediction should reproduce the final training loss, and the affine-logit control should reproduce $p^{\rm geo}$ to rounding. Give KL directions explicitly, use $\tfrac12\sum_y|p_y-q_y|$ for total variation, and average target surprises over the 80 readers with equal weight. These are comparisons at one trained parameter vector on overlapping training windows. They establish neither the best achievable mixture loss after refitting nor the existence of a true latent source.
