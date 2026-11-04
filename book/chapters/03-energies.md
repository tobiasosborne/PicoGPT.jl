# Energies and probabilities {#ch-energies}

::: claim
The model ends every column with $V$ unconstrained real numbers, the logits; read as negative energies, softmax turns them into the Gibbs distribution of statistical mechanics, so temperature, the indifference to a common shift, and log-sum-exp are all facts about a finite system in equilibrium.
:::

::: names
| maths | physics | ML |
|---|---|---|
| $z\in\R^V$ | negative energies $-E_a$ | logits |
| $\softmax(z)\in\Delta^{V-1}$ | Gibbs distribution | softmax output, probabilities |
| $\sum_a e^{z_a}$ | partition function $\mathcal Z$ | normaliser |
| $\lse(z)=\log\sum_a e^{z_a}$ | $\log\mathcal Z=-F$ at $\tau=1$ | log-sum-exp |
| $\tau=1/\beta$ | temperature | temperature |
| $\arg\max_a z_a$ | ground state | greedy choice |
| $z\mapsto z+c\one$ | choice of the zero of energy | shift invariance |
:::

## The last map {#sec-last-map}

The model's final map takes the residual stream, normalises each column, and applies one affine map:
$$Z=W_{\rm out}\,\LN(X;\gamma_f,\beta_f)+b_{\rm out}\one^\T\in\R^{V\times N},\qquad W_{\rm out}\in\R^{V\times D}.$$
Column $n$ of $Z$, which we call $z\in\R^V$, holds one real number for each character of the vocabulary. For the specimen that is seven numbers, one each for `␣ b e n o r t`. The model's job at this column is to state a probability distribution over the next character, that is, a point of the *probability simplex*
$$\Delta^{V-1}=\Bigl\{p\in\R^V:\ p_a\ge 0,\ \textstyle\sum_a p_a=1\Bigr\}.$$
The entries of $z$ obey no such constraint. They may be negative, and they need not sum to anything. We need a map $\R^V\to\Delta^{V-1}$, and we want it to be positive, normalised, increasing in each $z_a$, and smooth, because @ch-pulling-back will differentiate it. The map the model uses is
$$p_a=\frac{e^{z_a}}{\sum_{b=1}^{V}e^{z_b}},\qquad\text{in vector form}\qquad p=\softmax(z)=\frac{e^{z}}{\one^\T e^{z}},$$
with the exponential taken entry by entry. The entries of $z$ are called *logits*.

::: aside
"Logit" began as the name of the log-odds function $\operatorname{logit}(q)=\log\frac{q}{1-q}$ of a probability $q$ (Berkson, 1944). In machine learning it has drifted to mean any real number that becomes a probability after an exponential and a normalisation. Here: an entry of $z$.
:::

Why this map? One could normalise any positive function of the logits. The answer is that this particular one is an old acquaintance, and nearly everything we need to know about it was worked out in the nineteenth century.

## Softmax is the Gibbs distribution {#sec-gibbs}

Consider a physical system with finitely many states $a\in\{1,\dots,V\}$, where state $a$ has energy $E_a$. In equilibrium at temperature $\tau$ (in units where Boltzmann's constant is $1$), the probability of finding it in state $a$ is
$$p_a=\frac{1}{\mathcal Z}\,e^{-E_a/\tau},\qquad \mathcal Z=\sum_{a=1}^{V}e^{-E_a/\tau}.$$
This is the Gibbs (or Boltzmann) distribution, and $\mathcal Z$ is the *partition function*. Set
$$E_a=-z_a,\qquad \tau=1,$$
and it is softmax, symbol for symbol. A logit is a negative energy. A large logit is a deep energy level, and a deep level is heavily occupied. The model's last layer computes the energies of seven levels, and the prediction is the thermal occupation of those levels at temperature one.

@fig-energy-ladder draws exactly this, for one context of the trained specimen. Each character is a rung of a ladder at the height of its energy, and the bar along the rung is its probability. Every rung can be dragged, which changes one logit: lowering a rung fills it, and since the bars must sum to one, every other bar shrinks.

::: {.figure #fig-energy-ladder window=3 position=1}
The trained specimen's prediction for the character after "␣", at the start of the window "␣be␣or␣n", drawn as an energy ladder. Each character is a rung at its energy $E_a=-z_a$, and its bar is its probability $p_a$. The ruler on the left stands on the lowest rung and is $\tau$ long. Drag a rung, drag the axis to shift the whole ladder, or drag the end of the ruler to change the temperature.
:::

Look at the three lowest rungs in the figure. Three of the specimen's ten training windows begin with a space, and in them the space is followed by `b`, by `o` and by `n`. At the first position of a window the model sees nothing but that space, so it cannot tell the three windows apart, and it has put the three characters at almost exactly the same energy, far below the other four. A physicist would call this a threefold (nearly) degenerate ground state. In @ch-surprise we will see that this is not a failure of the model but the best it can possibly do.

The physics also tells us what $p$ optimises. For any distribution $r$ on the $V$ states, write its mean energy $U(r)=\sum_ar_aE_a$ and its entropy $H(r)=-\sum_ar_a\log r_a$. A two-line computation (@sec-lse) shows that the Gibbs distribution is the unique minimiser of
$$U(r)-\tau H(r)$$
over the whole simplex. At low temperature the minimum is dominated by the energy, so all mass goes to the lowest levels. At high temperature it is dominated by the entropy, so the mass spreads out. Softmax is the compromise between "put the probability where the logits are large" and "do not commit more than you must".

## Temperature {#sec-temperature}

Dividing the logits by $\tau>0$ before the softmax gives the Gibbs distribution at temperature $\tau$. Physicists often prefer the inverse temperature $\beta=1/\tau$:
$$p^{(\tau)}_a=\frac{e^{z_a/\tau}}{\sum_be^{z_b/\tau}}=\softmax(\beta z)_a .$$
The ratio of the occupations of two levels is $p_a/p_b=e^{-(E_a-E_b)/\tau}$. So temperature is the unit in which energy differences are measured: a rung $k\tau$ above another is $e^k$ times less occupied. That is what the ruler in @fig-energy-ladder shows. Its length is $\tau$, and its faint ticks mark $2\tau,3\tau,\dots$ above the lowest rung. Rungs far above the end of the ruler are empty.

The two limits are worth stating exactly. As $\tau\to\infty$, every weight $e^{z_a/\tau}$ tends to $1$, and
$$\lim_{\tau\to\infty}p^{(\tau)}_a=\frac1V,$$
the uniform distribution: the system melts. As $\tau\to 0$, let $G=\arg\max_a z_a$ be the set of lowest levels, the ground states. Dividing numerator and denominator by $e^{\max z/\tau}$ leaves weights $e^{-(E_a-E_{\min})/\tau}$, which tend to $1$ on $G$ and to $0$ elsewhere, so
$$\lim_{\tau\to 0}p^{(\tau)}_a=\begin{cases}1/|G|,&a\in G,\\ 0,&a\notin G.\end{cases}$$
The frozen limit is uniform on the ground states. It is not an arbitrary choice of one of them: tied levels share the probability equally, at every temperature.

::: {.figure #fig-temperature contexts="1:1,3:1"}
Two of the specimen's predictions, each at seven temperatures. Top: after "t", where the corpus is unambiguous; bottom: after "␣", where it is not. Each ladder is the same set of energies; only the temperature changes, and with it the occupations and the entropy $H$ printed beneath. On the left, the same distributions as points of the simplex: the path runs from the uniform centre ($\tau=\infty$) towards the ground states ($\tau\to0$), with the seven temperatures marked.
:::

The bottom row of @fig-temperature shows what "nearly" degenerate means in practice. The three low rungs after "␣" are not exactly level. At $\tau=1$ they share the probability almost equally, and they still do at $\tau=\frac14$, because the gaps between them are tiny compared with $\tau$. Only when $\tau$ falls far below those gaps does the lowest of the three take everything. The limit $\tau\to0$ is honest: it picks the true ground state. But at any temperature you are likely to use, the three behave as one degenerate level.

::: aside
**The frontispiece.** The drawing that opens this chapter collects such paths for every logit column the specimen produced while it was being trained. The seven characters sit at the corners of a heptagon, in vocabulary order, clockwise from the top. Each hairline is one column $z$, followed from $\tau=100$ (the centre) down to $\tau=0.04$. A path that reaches a corner is a context with one clear successor; a path that stops on an edge or inside a face is one the text leaves open. The map from the simplex to the page is linear, so straight lines stay straight; but it squashes six dimensions into two, so it is a shadow of the simplex, not a chart of it.
:::

The model is trained at $\tau=1$, and inside the model the softmax always runs at the temperature it was trained at. Temperature becomes a free parameter in two other places: in attention, where the scores are divided by $\sqrt d$ (@ch-attention), and when the model writes text, where it is the most important dial there is (@ch-speaking).

## Only differences matter {#sec-shift}

For every constant $c\in\R$,
$$\softmax(z+c\one)=\softmax(z),$$
because numerator and denominator both gain the factor $e^{c}$. In the physics this is the statement that energy has no absolute zero: shifting every level by the same amount is a change of convention, not of the system. Drag the axis in @fig-energy-ladder and the whole ladder slides, the partition function and the free energy change, and not one probability moves.

So the map $z\mapsto\softmax(z)$ is not injective. It is constant on every line $z+\R\one$, and its derivative must annihilate $\one$. Differentiating the quotient gives the Jacobian, in index form and in matrix form:
$$\frac{\partial p_a}{\partial z_b}=p_a(\delta_{ab}-p_b),\qquad Dp(z)=\diag(p)-pp^\T .$$
Indeed $(\diag(p)-pp^\T)\one=p-p\,(\one^\T p)=0$. This symmetric matrix is a covariance matrix in disguise: for any $v\in\R^V$,
$$v^\T Dp(z)\,v=\sum_ap_av_a^2-\Bigl(\sum_ap_av_a\Bigr)^2=\operatorname{Var}_{a\sim p}(v_a)\ge0 .$$
A physicist recognises the fluctuation–response relation: the response of the occupations to a change of the energies is the covariance of the fluctuations.

Shift invariance has consequences inside the model. Adding $c\one$ to the output bias $b_{\rm out}$ adds $c$ to every logit of every column and changes no prediction. The loss therefore cannot see that direction of $b_{\rm out}$, and its gradient is always orthogonal to $\one$; @ch-surprise shows why directly.

::: aside
A puzzle to carry forward. Somewhere else in the model there is a whole vector of parameters whose only effect is to add the same number to every input of a softmax. By this section it has no effect on anything the model computes, and its gradient is exactly zero: one group of the specimen's numbers can never learn. Which group? @ch-attention gives the answer in its exercises.
:::

## Log-sum-exp and the free energy {#sec-lse}

The logarithm of the partition function deserves its own name. At temperature $\tau=1/\beta$ define
$$\lse_\beta(z)=\frac1\beta\log\sum_ae^{\beta z_a},\qquad \mathcal F_\beta(E)=-\frac1\beta\log\sum_ae^{-\beta E_a},$$
so that the free energy and log-sum-exp are the same function up to signs: $\mathcal F_\beta(-z)=-\lse_\beta(z)$. We write $\lse=\lse_1$. Three facts make it the most useful function in this book.

*Its gradient is the Gibbs distribution.* Differentiating,
$$\frac{\partial\,\lse_\beta(z)}{\partial z_a}=p^{(\beta)}_a,\qquad \frac{\partial\mathcal F_\beta(E)}{\partial E_a}=p^{(\beta)}_a .$$
Mind the signs: with $E=-z$, the derivative of $\mathcal F_\beta(-z)$ with respect to $z_a$ is $-p_a$. Calling log-sum-exp itself "the free energy" loses exactly that minus sign.

*It is a smooth maximum.* Factor out the largest exponential: with $m=\max_az_a$,
$$m\;\le\;\lse_\beta(z)\;\le\;m+\frac{\log V}{\beta}.$$
As $\tau\to0$, log-sum-exp tends to the maximum logit, and the free energy to the ground-state energy. In @fig-energy-ladder the free energy is the small triangle $F$ on the axis. It sits below the lowest rung, never more than $\tau\log V$ below it, and it follows the ladder when the ladder slides, because $\lse(z+c\one)=\lse(z)+c$.

*It is a variational principle.* Substitute $\log p_a=-\beta E_a-\log\mathcal Z_\beta$ into the definitions of mean energy and entropy. For every distribution $r$,
$$\sum_ar_aE_a-\tau H(r)=\mathcal F_\beta(E)+\tau\KL(r\,\|\,p^{(\beta)}),$$
where $\KL(r\,\|\,p)=\sum_ar_a\log(r_a/p_a)$ is the Kullback–Leibler divergence.
The divergence $\KL$ is never negative and vanishes only at $r=p$; we prove this in @ch-surprise. So the Gibbs distribution is the unique minimiser of $U-\tau H$, and the minimum value is the free energy. In the logits' language, $\lse_\beta(z)=\max_r\{\langle r,z\rangle+H(r)/\beta\}$.

### Computing it {.unnumbered}

The definition is a recipe for disaster on a computer. A Float64 number cannot exceed about $1.8\times10^{308}$, so $e^{x}$ overflows to $\infty$ once $x$ passes about $709.8$, and for very negative $x$ it underflows to $0$. Logits of size $710$ are rare, but a logit of $10$ at temperature $1/100$ is the same thing, and then softmax computes $\infty/\infty$, which is not a number. The cure is shift invariance itself. For any $c$,
$$\lse(z)=c+\log\sum_ae^{z_a-c},\qquad \softmax(z)=\softmax(z-c\one),$$
and with $c=\max_az_a$ every exponent is at most $0$ and one of them is exactly $0$. The sum then lies between $1$ and $V$: it cannot overflow, and the logarithm never sees zero. The maximum has kinks where two logits tie, but the expression on the right is identically the smooth function on the left, so its derivative is still softmax.

::: {.figure #fig-logsumexp window=3 position=1}
Why subtract the maximum. The seven exponentials of the logits of @fig-energy-ladder, multiplied by $\beta$, on a logarithmic axis that spans the whole range of Float64. Top: computed as written, $e^{\beta z_a}$. Bottom: computed after subtracting the maximum, $e^{\beta(z_a-m)}$. Drag $\beta$ to the right: the naive exponentials leave the representable range, their sum becomes $\infty$, and the probabilities become $\infty/\infty$; the shifted ones never exceed $1$.
:::

The same care decides how the loss of @ch-surprise is computed: never as the logarithm of a probability, which may have underflowed to zero, but directly from the logits, as $-\log p_y=\lse(z)-z_y$.

## In the code {#sec-code-energies}

Both functions are one line each:

::: {.code file="src/softmax.jl" fn="logsumexp"}
:::

::: {.code file="src/softmax.jl" fn="softmax"}
:::

`softmax!` overwrites its argument, and the keyword `dims` normalises slices of an array: `dims = 1` for the columns of $Z$, one distribution per token, and `dims = 2` for the rows of an attention matrix. Note the subtraction of the maximum, along the same dimension.

## Exercises {.unnumbered}

::: exercise
Derive $\partial p_a/\partial z_b=p_a(\delta_{ab}-p_b)$. Show that the Jacobian is symmetric, positive semidefinite, and that its kernel is exactly $\R\one$ when every $p_a>0$. What happens to it in the frozen limit $\tau\to0$ with a unique ground state?
:::

::: exercise
Prove $\max_az_a\le\lse_\beta(z)\le\max_az_a+\log V/\beta$. When is each bound attained?
:::

::: exercise
Let $U(\beta)=\sum_ap^{(\beta)}_aE_a$ be the mean energy. Show that $dU/d\beta=-\operatorname{Var}_{p^{(\beta)}}(E)\le0$: cooling never raises the mean energy. Show also that the entropy $H(p^{(\beta)})$ never increases as $\beta$ grows.
:::

::: exercise
In @fig-energy-ladder, drag one of the upper rungs down until it snaps level with the lowest one, then drag the end of the ruler down to cool the ladder. What happens to the tied pair, and to everyone else? Compare with the formula for the frozen limit.
:::

::: exercise
In @fig-energy-ladder, drag the axis. Which of the numbers on the right change, and by how much? Explain why $F$ must move by exactly the shift.
:::

::: exercise
In @fig-energy-ladder, click through the positions of the window "o␣be␣or␣". At which position is the trained model unsure of the next character, and between which characters? Find, in the corpus, the reason. (@ch-surprise makes this exact.)
:::
