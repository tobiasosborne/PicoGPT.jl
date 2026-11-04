# Measuring surprise {#ch-surprise}

::: claim
Training charges the model, for every prediction, its surprise $-\log q(y)$ at the character that actually came; the mean surprise is a cross-entropy, which exceeds the entropy of the text by exactly a KL divergence, and its gradient with respect to the logits is the prediction minus the observation.
:::

::: names
| maths | physics | ML |
|---|---|---|
| $-\log q_y$ | surprisal | per-token loss, negative log-likelihood |
| $\mathcal L(\theta)=\frac1N\sum_n-\log q_\theta(y_n\mid c_n)$ | mean surprisal | loss, cross-entropy loss |
| $H(p)=-\sum_ap_a\log p_a$ | Gibbs entropy | entropy |
| $H(p,q)=-\sum_ap_a\log q_a$ | mean surprisal under $p$ | cross-entropy |
| $\KL(p\,\|\,q)$ | relative entropy | KL divergence |
| $\log V$ | entropy of the uniform distribution | random baseline |
| $\prod_nq_\theta(y_n\mid c_n)$ | likelihood | likelihood |
| $\bar Z=(P-Y)/N$ | predicted minus observed occupation | logit gradient |
:::

## The price of a prediction {#sec-price}

At each position the model states a distribution $q\in\Delta^{V-1}$ over the next character (@ch-energies). Then the text reveals the character $y$ that actually comes. How much should the model be charged?

Three requirements pin the answer down. A prediction of certainty that comes true, $q_y=1$, should cost nothing. The cost should grow as $q_y$ falls. And the chain rule of probability (@ch-task) writes the probability of a text as a *product* of next-character probabilities, so the cost of a text should be the *sum* of the costs of its characters. A continuous function $f:(0,1]\to\R$ with $f(st)=f(s)+f(t)$ is a multiple of the logarithm, $f(s)=-k\log s$ (@ex-cauchy). We take $k=1$ and the natural logarithm, and call
$$s(q,y)=-\log q_y$$
the *surprise* of the prediction $q$ when $y$ occurs. It is zero for an event that was certain, infinite for one that was declared impossible, $\log 2$ for a fair coin, and $\log V$ for a guess spread evenly over $V$ characters. Its unit is the *nat*.

::: aside
With $\log_2$ instead of $\log$ the unit is the *bit*; one nat is $1/\log 2\approx1.44$ bits. [Shannon (1948)](https://people.tamu.edu/~jml/entropy.pdf) built information theory on this quantity; [Tribus (1961)](https://archive.org/details/thermostaticsthe00trib) calls it “surprisal”.
:::

The model does not compute $q_y$ and then its logarithm. Since $q=\softmax(z)$, @ch-energies gives the surprise directly from the logits, with no exponential that could overflow and no probability that could underflow to zero:
$$-\log q_y=\lse(z)-z_y .$$

## The loss {#sec-loss}

A training batch of $B$ windows of length $T$ makes $N=TB$ predictions; for the specimen that is $10\times8={{value:predictions}}$. Write $c_n$ for the context of prediction $n$, the characters of its window up to and including position $t$, and $y_n$ for the character that follows. The *loss* is the mean surprise,
$$\mathcal L(\theta)=\frac1N\sum_{n=1}^{N}-\log q_\theta(y_n\mid c_n)=\frac1N\sum_{n=1}^{N}\Bigl[\lse(z_n)-Z_{y_n,n}\Bigr].$$
In matrix form, with $Y\in\{0,1\}^{V\times N}$ the matrix whose column $n$ is the one-hot vector $e_{y_n}$,
$$\mathcal L=\frac1N\Bigl[\sum_n\lse(Z_{:,n})-\langle Y,Z\rangle_F\Bigr].$$
The number $\log V$ is the yardstick. A model that knows nothing and says so, $q_a=1/V$, pays exactly $\log V$ for every prediction. The specimen is initialised with small random parameters, so its logits start near zero and its predictions near uniform, and its first loss is close to $\log V$. That is not a bound, however: a confident wrong prediction costs more than $\log V$, and without limit.

@fig-surprise shows all eighty charges at once. Each row is one training window, set under the corpus where it lies, and each bar stands under the character it predicts. The outline is what the untrained model pays; the solid bar is what the model pays after the chosen number of training steps. [Run the training and watch the bars fall.]{.web} [The stages show the bars falling during training.]{.print}

::: {.figure #fig-surprise}
The surprise $-\log q(y)$ of each of the specimen's eighty predictions. Row $b$ is training window $b$, placed under the corpus; each bar stands under the character it predicts. Outlines: the untrained model. Solid bars: the model during training, replayed exactly as it was trained. The dashed hairline is $\log V$, uniform guessing; the loss is the mean of all the bars.
:::

Almost every bar falls to nothing. Five do not. Each is the first bar of its row, and each settles at a level of its own: two at one height, three at another. These five are the subject of @sec-floor-surprise.

::: aside
**The frontispiece** draws the same eighty quantities as curves: each hairline is the surprise of one prediction through a training run, with time to the right on a logarithmic scale and surprise upward. The value-ink line is their mean, the loss. The two plateaux that refuse to fall are the five predictions of @sec-floor-surprise.
:::

## Maximum likelihood {#sec-likelihood}

The loss is a logarithm in disguise:
$$\mathcal L(\theta)=-\frac1N\log\prod_{n=1}^{N}q_\theta(y_n\mid c_n).$$
The product is the *likelihood*: the probability that the model assigns to the targets that were actually observed, given their contexts. The logarithm is strictly increasing and $-1/N$ reverses the order, so minimising the mean surprise and maximising the likelihood are the same problem. Training a language model is maximum-likelihood estimation, and the loss is the negative log-likelihood per prediction.

::: aside
The windows overlap, so most characters of the corpus are targets in several windows, each time with a different context. The product above is therefore not the probability of the corpus read once from left to right; it is a product of conditional probabilities, one per (context, target) pair of the training set, which is all that training needs.
:::

## Cross-entropy and the KL divergence {#sec-kl}

Suppose the next character really is random, with distribution $p$, and the model predicts $q$. The expected surprise is the *cross-entropy*
$$H(p,q)=-\sum_ap_a\log q_a .$$
When $q=p$ it is the *entropy* $H(p)=-\sum_ap_a\log p_a$, the surprise that even a perfect predictor must expect. The difference is the *Kullback–Leibler divergence*:
$$H(p,q)=H(p)+\KL(p\,\|\,q),\qquad \KL(p\,\|\,q)=\sum_{a:\,p_a>0}p_a\log\frac{p_a}{q_a}.$$
(Add and subtract $\sum_ap_a\log p_a$.) The divergence is never negative. For every $t>0$, $-\log t\ge1-t$, with equality only at $t=1$. So
$$\KL(p\,\|\,q)=\sum_{a:\,p_a>0}p_a\Bigl(-\log\frac{q_a}{p_a}\Bigr)\;\ge\;\sum_{a:\,p_a>0}p_a\Bigl(1-\frac{q_a}{p_a}\Bigr)=1-\sum_{a:\,p_a>0}q_a\;\ge\;0,$$
with equality exactly when $q_a=p_a$ wherever $p_a>0$ and $q$ puts no mass anywhere else, that is, when $q=p$. This is Gibbs' inequality, and it is the fact @ch-energies borrowed to prove that softmax minimises the free energy. Here it says something just as useful: *the expected surprise is smallest when the model predicts the true distribution, and the smallest value is the entropy of the truth.* The KL divergence measures, in nats, how much a prediction costs above that.

::: aside
In statistics a loss with this property, minimised in expectation by reporting the true distribution, is called a *proper scoring rule*. With three or more outcomes the logarithmic score is, up to affine changes, the only proper rule that depends on $q$ only through the probability $q_y$ of what actually happened.
:::

## The floor {#sec-floor-surprise}

Now apply this to the training set itself. Two predictions with the same context receive the same input, so the model gives them the same distribution $q_c$. Group the $N$ predictions by context: let $n_c$ be the number of predictions with context $c$, and $\hat p_c$ the distribution of the characters that followed it. Regrouping the sum in $\mathcal L$ gives, exactly,
$$\mathcal L(\theta)=\sum_c\frac{n_c}{N}\,H(\hat p_c,q_c)=\underbrace{\sum_c\frac{n_c}{N}\,H(\hat p_c)}_{\text{the text}}\;+\;\underbrace{\sum_c\frac{n_c}{N}\,\KL(\hat p_c\,\|\,q_c)}_{\text{the model}} .$$
The first term does not depend on $\theta$ at all; it is a property of the corpus and of the way it was cut into windows. The second is never negative, and vanishes only if the model predicts, for every context, exactly what followed it. So the first term is a floor that no model, however large, can go below.

For most contexts $\hat p_c$ is a single character with probability one, and its entropy is zero. Which contexts are left open? A context is a window prefix, and every prefix of length two or more occurs in the specimen's training set either once or always with the same successor. The contexts of length one are different: there the model sees one character and nothing before it. Two windows begin with "o": in one the "o" of "to" is followed by a space, in the other the "o" of "or" is followed by "r". Three windows begin with a space, followed by "b", "o" and "n". For those contexts the best possible prediction is $\hat p_c$, uniform on two and on three characters, and the floor is
$$\mathcal L_{\min}=\frac{2\log2+3\log3}{{{value:predictions}}}.$$
@fig-cross-entropy-kl computes this from the corpus and the windows, and splits the loss into floor and divergence at every step of training.

::: {.figure #fig-cross-entropy-kl}
The loss regrouped by context. The two contexts the training set leaves open are shown in full: what followed them (outlined bars, one per character) and what the model predicts (solid bars); every other context is followed by one character only. On the right, the cross-entropy of each context split into the entropy of what followed (outlined) and the divergence of the model from it (solid). Bottom: the loss, which is the floor plus the mean divergence.
:::

The ambiguity is an artefact of windowing, not of the text. Each window starts in the middle of the corpus with no memory of what came before; had its first prediction been allowed to see one more character, it would have known whether its "o" was the "o" of "to" or of "or", and the floor would be zero (@ex-floor-zero). Real models have the same problem at the start of every window, which is one reason long contexts help.

Two more things in the figure are worth noticing. The open contexts end up predicting almost exactly the uniform $\hat p_c$, which is what maximum likelihood demands. And the decided contexts never quite reach divergence zero: that would need $q_c$ to be exactly one-hot, which a softmax with finite logits never is (@ch-energies). The logit gaps would have to grow without bound. So the loss approaches the floor but never meets it.

## The gradient {#sec-gradient}

Training needs the derivative of $\mathcal L$ with respect to everything, and it starts with the logits. For one prediction, $\ell_n=\lse(z_n)-Z_{y_n,n}$. @ch-energies showed that $\partial\lse/\partial z_a=p_a$, and $\partial Z_{y,n}/\partial Z_{a,n}=\delta_{ay}$. Averaging over the $N$ predictions,
$$\bar Z_{an}=\frac{\partial\mathcal L}{\partial Z_{an}}=\frac{p_{an}-\delta_{a,y_n}}{N},\qquad \bar Z=\frac{P-Y}{N},$$
where $P=\softmax(Z)$ column by column. The bar is the notation of @ch-pulling-back for the adjoint, the sensitivity of the loss to a quantity; in this book adjoints are drawn in the second ink, hatched. In words: the gradient with respect to the logits is the *predicted* occupation minus the *observed* one. Gradient descent moves $z_n$ against it. It raises the logit of the character that came, by $(1-p_y)/N$ times the step size, and lowers every other logit in proportion to the probability the model gave it.

::: {.figure #fig-gradient-p-minus-e window=3 position=1}
One prediction, forward and backward. Going down (value ink): the logits $z$, their softmax $p$, and the surprise $\ell$ of the character that came, whose one-hot vector $e_y$ is outlined. Going up (covector ink, hatched): the gradient $N\bar z=p-e_y$, which arrives above the logits it belongs to. When other windows share the context, the bottom row adds their gradients. [Choose any of the eighty predictions on the corpus line.]{.web}
:::

Each column of $\bar Z$ sums to zero, $\one^\T(p-e_y)=1-1=0$. That is shift invariance again: the loss cannot see a common shift of the logits, so its gradient has no component along $\one$. In particular the gradient of the output bias, $\bar b_{\rm out}=\bar Z\one_N$, is always orthogonal to $\one$, as promised in @ch-energies.

The gradient also explains the floor from the other side. The three predictions whose context is a single space have the same logits $z$ and the same $p$, but different targets. Their gradients add up to
$$(p-e_b)+(p-e_o)+(p-e_n)=3\,(p-\hat p),\qquad \hat p=\tfrac13(e_b+e_o+e_n).$$
Each window on its own pulls its own character up. Together they pull towards $\hat p$, and they stop pulling exactly when $p=\hat p$. [Click the first "␣" of a window in @fig-gradient-p-minus-e to see the three pulls and their sum.]{.web}

## In the code {#sec-code-surprise}

The loss and its pullback are one function. It subtracts each column's maximum $m_n$, exponentiates, sums each column to $s_n$, and returns the mean surprise together with a closure `back` that returns $\bar Z$:

::: {.code file="src/softmax.jl" fn="crossentropy"}
:::

The surprise of prediction $n$ is computed as $\log s_n-(Z_{y_n,n}-m_n)$ rather than as $\lse(z_n)-Z_{y_n,n}$. The two are equal in exact arithmetic. In floating point, a large common shift of the logits (@ch-energies) would make the second form subtract two nearly equal large numbers and lose the surprise in the rounding; the first subtracts the shift from both terms before they meet. `back` divides the shifted exponentials by their sums, which is $P$, subtracts one at each target, and scales by $\bar{\mathcal L}/N$. It is the first pullback the backward pass calls, and every other gradient in the model is computed from its output.

## Exercises {.unnumbered}

::: {.exercise #ex-cauchy}
Let $f:(0,1]\to\R$ be continuous with $f(st)=f(s)+f(t)$. Show that $f(s)=-k\log s$ for some constant $k$. (Consider $g(x)=f(e^{-x})$ on $[0,\infty)$.)
:::

::: exercise
A coin lands heads. One forecaster said $q(\text{heads})=0.9$, another $0.5$. What does each pay, in nats and in bits? What would each have paid had it landed tails? Show that a forecaster who always says $1$ pays an infinite amount the first time it is wrong.
:::

::: {.exercise #ex-floor-zero}
Suppose the first prediction of every window could also see the character before the window. List the contexts of length one that become contexts of length two, and show that the specimen's floor would then be zero.
:::

::: exercise
Show that plain gradient descent never changes the sum $\sum_a(b_{\rm out})_a$ of the output biases. Does AdamW, whose step is not proportional to the gradient (@ch-learning), preserve it too?
:::

::: exercise
Consider @fig-cross-entropy-kl at step 0. [Drag the training back to that step.]{.web} How much of the loss is floor and how much divergence? Find the step at which the divergence first falls below the floor.
:::

::: exercise
In @fig-surprise, identify the five bars that do not fall. For each, write down the context the model sees and the characters that follow that context in the training set, and check that the bar settles at $\log 2$ or $\log 3$.
:::
