# Learning {#ch-learning}

::: claim
Learning is a walk downhill in $\R^p$: gradient descent is Euler's method for the gradient flow $\dot\theta=-\nabla\ell(\theta)$, and everything else in a modern training loop (minibatches, Adam, schedules, clipping, initial variances) is a decision about how long, and how well aimed, each step may be.
:::

::: names
| maths | physics | ML |
|---|---|---|
| $\dot\theta=-\nabla\ell(\theta)$ | overdamped motion in a potential | gradient flow |
| $\theta_{k+1}=\theta_k-\eta\nabla\ell(\theta_k)$ | an explicit Euler step | gradient descent, update, training step |
| $\eta$ | the time step | learning rate, step size |
| $\lambda_{\max}$ of the Hessian | the stiffest mode, the curvature scale | sharpness |
| $\hat g$ with $\E\hat g=\nabla\ell$ | a noisy measurement of the force | minibatch gradient, stochastic gradient |
| $\theta\leftarrow\theta-\eta\,G\hat m$, $G$ diagonal | a coordinate-dependent mobility | Adam, preconditioning |
| $\theta\leftarrow(1-\eta\lambda)\theta-\cdots$ | friction towards the origin | weight decay, AdamW |
| projection onto $\{\lVert g\rVert\le c\}$ | a bound on the force | gradient clipping |
:::

## Watching it learn {#sec-watch}

Fix the training set: the ten windows of the corpus, eighty predictions in all. The loss of @ch-surprise is then one function on the space of parameters,
$$\ell:\R^p\to\R,\qquad\theta\longmapsto\frac1N\sum_{n=1}^N-\log p_\theta(y_n\mid\text{context}_n),$$
with $p={{value:specimenParams}}$ and $N={{value:predictions}}$, and @ch-pulling-back computes its gradient $\nabla\ell(\theta)\in\R^p$ at the cost of about one forward pass. Training means finding a $\theta$ where $\ell$ is small. The loss is convex as a function of the logits $Z$: each column contributes a positive multiple of the covariance matrix of @ch-energies to its Hessian. Its composition with the model is not convex in $\theta$, so the local descent argument below gives no guarantee of a global minimum. Two numbers frame the answer. A model that has learned nothing and spreads its probability evenly over the $V$ characters pays $\ln V$ nats per prediction: that is *uniform guessing*. And no model can pay less than the *entropy floor* of @sec-floor-surprise, because five of the eighty predictions are made from a context of a single character that the windows continue in different ways. Two windows begin with "o" and three with a space, so the floor is $(2\ln2+3\ln3)/{{value:predictions}}$ nats, and not zero.

[@fig-train-live runs the whole of this chapter in front of you.]{.web} [@fig-train-live summarises the recorded training run.]{.print} Every step is a forward pass on the ten windows, a backward pass, and one update of all the parameters. It starts at $\theta_0$, the specimen's initial parameters. [With its opening settings the live figure replays the run used throughout the book.]{.web}

::: {.figure #fig-train-live window=1 reader=8}
The specimen learning. Every step is a forward pass, a backward pass and an update of all the parameters. The curve is the loss, with the line of uniform guessing at $\ln V$ and the entropy floor of the corpus; above it, what the model writes after "to be or" at a few steps along the way. [Each parameter is one square; the arcs show what the last character reads.]{.web} [The printed plate records the loss and samples; @fig-specimen-anatomy shows the parameters.]{.print} [Press ▶ to train live; under the live curve is the learning rate of every step.]{.web} [Drag the learning rate on its axis, change the optimiser or the schedule, or start again from another seed.]{.web}
:::

::: aside
[The faint curve is the run recorded in `artifacts/specimen.json` by the Julia code. The browser's run is a separate computation, by the JavaScript engine, from the same $\theta_0$ with the same settings; the replay is checked against the recorded losses.]{.web} [The printed curve replays the run recorded in `artifacts/specimen.json`, from its stored initial parameters and recipe.]{.print} Arrays agree to $10^{-12}$ absolute; the recorded loss curve to $10^{-9}$ absolute. A replay that fails the latter check is discarded.
:::

Three things happen, in this order. For the first few steps almost nothing does: the loss sits on the line of uniform guessing and the model writes noise. Then the loss falls steeply, and the samples pass through a phase of spaces and frequent letters, then of the right words in the wrong places, before they settle on "not to be". Finally the curve flattens onto the floor and stays there. The step axis is $\log(1+\text{step})$, with $+1$ so step 0 can be drawn; it gives the early steps more room than a linear axis would.

::: aside
The cover of this book is the same story told with every parameter: each of its {{value:specimenParams}} hairlines is one coordinate $\theta_i$ followed through a training run, with time running down the page on a logarithmic scale. The trace records its own initialisation and recipe in its plate caption; a seed identifies draws only within the engine that made them. In the live figure, seed 0 means the stored $\theta_0$, not a fresh random draw.
:::

## Gradient flow and its Euler steps {#sec-flow}

The simplest way to make a function smaller is to move against its gradient. As a differential equation in a fictitious time $s$, this is the *gradient flow*
$$\frac{d\theta}{ds}=-\nabla\ell(\theta).$$
Along a solution the loss can only decrease, since by the chain rule
$$\frac{d}{ds}\,\ell(\theta(s))=\inner{\nabla\ell}{\dot\theta}=-\lVert\nabla\ell(\theta(s))\rVert^2\le0,$$
with equality only where the gradient vanishes. A physicist reads this as overdamped motion: a particle in the potential $\ell$, in a fluid so viscous that its velocity is proportional to the force on it, slides downhill and dissipates the potential energy it loses.

A computer cannot follow the flow; it can take steps. Replacing the velocity during one step of length $\eta$ by its value at the start of the step is Euler's method, and Euler's method for the gradient flow is *gradient descent*:
$$\theta_{k+1}=\theta_k-\eta\,\nabla\ell(\theta_k).$$
The step size $\eta$ is called the *learning rate*. It is the time step of the integrator, and like every time step it has to be small enough. How small?

The answer depends on how fast the gradient itself changes. Suppose that on the segment between $\theta$ and $\theta+s$ the gradient is $\Lambda$-Lipschitz, $\lVert\nabla\ell(\theta')-\nabla\ell(\theta'')\rVert\le\Lambda\lVert\theta'-\theta''\rVert$. Then
$$\ell(\theta+s)\le\ell(\theta)+\inner{\nabla\ell(\theta)}{s}+\frac\Lambda2\lVert s\rVert^2,$$
by the fundamental theorem of calculus applied to $t\mapsto\ell(\theta+ts)$. Put $s=-\eta g$ with $g=\nabla\ell(\theta)$:
$$\ell(\theta-\eta g)\;\le\;\ell(\theta)-\eta\Bigl(1-\frac{\Lambda\eta}2\Bigr)\lVert g\rVert^2.$$
This is the *descent lemma*. The first-order term promises a decrease proportional to $\eta$; the curvature can take back an amount proportional to $\eta^2$. If $\eta<2/\Lambda$ the promise wins and the step lowers the loss, and $\eta=1/\Lambda$ guarantees a decrease of at least $\lVert g\rVert^2/(2\Lambda)$. No convexity is needed.

::: aside
In the literature $\Lambda$ is usually called $L$ and the loss "$L$-smooth". In this book $L$ already counts the blocks.
:::

The number $2/\Lambda$ is no artefact of the proof. On a quadratic $\ell(\theta)=\tfrac12\theta^\T H\theta$, gradient descent is the linear map $\theta\mapsto(I-\eta H)\theta$, which contracts the eigendirection of eigenvalue $\lambda$ by the factor $1-\eta\lambda$. It shrinks every direction if and only if $\lvert1-\eta\lambda\rvert<1$ for all of them, that is, if $\eta<2/\lambda_{\max}$. Beyond that the stiffest mode overshoots further at every step, the familiar instability of an explicit integrator applied to a stiff equation. The largest eigenvalue of the Hessian, which plays the role of $\Lambda$ near a point, is called the *sharpness* of the loss there.

For the specimen there is no single $\Lambda$. The rectifier of the MLP has kinks, where the gradient jumps, and the curvature changes from place to place. The lemma is a local statement, and @fig-step-size shows how local. At three learning rates it runs plain gradient descent on the specimen and measures the sharpness at the current parameters as it goes.

::: {.figure #fig-step-size}
How large a step can gradient descent take? Plain gradient descent on the specimen at three learning rates. Below each loss curve, the sharpness $\Lambda$, the largest eigenvalue of the Hessian of the loss at the current parameters, and the line $2/\eta$: the descent lemma guarantees a decrease only while the sharpness stays below it. It starts below, climbs, and then settles on the line.
:::

At the start the sharpness is well below $2/\eta$, and the loss falls as the lemma promises. But training does not leave the curvature alone: the descent drives $\theta$ into steeper and steeper regions, the sharpness grows until it reaches $2/\eta$, and there it stops. Gradient descent then walks along the edge of the region where its steps are stable, overshooting in the stiffest direction just enough to keep from going further. The loss keeps falling, though no longer at every step. The lemma explains where the edge is, and it does not explain why the run should stay on it: a sufficient condition for descent is not a law of motion.

::: aside
This is the *edge of stability*, found by [Cohen, Kaur, Li, Kolter and Talwalkar (2021)](https://arxiv.org/abs/2103.00065) in networks far larger than the specimen. The sharpness is measured here as in @ch-pulling-back: a Hessian–vector product is a central difference of two pulled-back gradients, and power iteration on it finds the largest eigenvalue without ever forming the Hessian.
:::

::: aside
Three indices play different roles: text position $t$ labels token columns, depth labels successive residual updates, and training step $k$ labels changes of $\theta$. The Euler form in @ch-block concerns depth; gradient flow concerns training time. Pulling an adjoint back reverses the depth computation, not the training trajectory.
:::

## Minibatches {#sec-minibatch}

The specimen computes its gradient on all ten windows at every step. A real training set does not allow that: the Shakespeare model of @ch-speaking has a million windows to choose from, and each step looks at a few dozen. Write the loss as a mean over $M$ examples, $\ell=\tfrac1M\sum_{i=1}^M\ell_i$, choose $b$ of them at random, $I_1,\dots,I_b$, and use
$$\hat g=\frac1b\sum_{j=1}^b\nabla\ell_{I_j}$$
in place of $\nabla\ell$. If each $I_j$ is uniform on $\{1,\dots,M\}$ then $\E\nabla\ell_{I_j}=\tfrac1M\sum_i\nabla\ell_i=\nabla\ell$, by linearity alone, and so
$$\E\,\hat g=\nabla\ell.$$
The *minibatch gradient* $\hat g$ is an unbiased estimate of the gradient. If the $I_j$ are drawn independently (with replacement), its covariance is $\Sigma/b$, where $\Sigma$ is the covariance of a single example's gradient: the noise shrinks like $1/\sqrt b$. Drawing without replacement, as in practice, keeps the mean and multiplies the covariance by $(M-b)/(M-1)$, which is zero when the batch is the whole set. Gradient descent with $\hat g$ is *stochastic gradient descent*, Euler's method with a force measured afresh, and noisily, at every step.

::: {.figure #fig-minibatch}
Every minibatch gradient of the specimen, projected onto a plane. Each dot is the gradient of the loss on $b$ of the ten windows, for every possible choice of the $b$; the cross is the full gradient, which is exactly the mean of each cloud. Before training, and after it.
:::

For the specimen nothing need be sampled: there are exactly $\binom{10}{b}$ minibatches of size $b$, and @fig-minibatch computes the gradient of every one. Before training the clouds are centred on a large full gradient, and they tighten as $b$ grows. After training the full gradient is almost zero, as it should be at the bottom of the loss, but the clouds are not. The windows that begin with "o" or with a space still pull hard: at the floor the model gives probability one half to each continuation of "o", so the window whose next character is a space pushes towards the space, the window whose next character is "r" pushes towards "r", and only their sum vanishes. Minibatch noise does not die out at a minimum of the loss. It dies out only where every single example is satisfied, and the floor exists precisely because that cannot happen.

::: aside
Unbiasedness survives only linear operations. Clipping, and Adam's division by the size of the gradient, are applied to $\hat g$ and not to its mean, and they make the step a biased estimate of anything one might want. Nothing in practice seems to suffer much from this; nothing in theory says it shouldn't.
:::

## Adam {#sec-adam}

The parameters of a network are not alike. In the specimen the embedding, the attention maps, the MLP and the readout receive gradients of very different sizes, and so do the coordinates within one array (@fig-adam). A single learning rate is then too large for some coordinates and too small for others: gradient descent moves each $\theta_i$ in proportion to $\partial\ell/\partial\theta_i$, and a coordinate with a tiny gradient barely moves at all.

The remedy is a *preconditioner*: a positive definite $G$, and the step $\theta\leftarrow\theta-\eta Gg$. Such a step is still steepest descent, measured in a different metric: $-Gg$ minimises $\inner{g}{s}+\tfrac12s^\T G^{-1}s$, the first-order change of the loss plus a penalty on the length of $s$ in the norm of $G^{-1}$. Newton's method takes $G$ to be the inverse Hessian. That costs $p^2$ numbers to store, and finding it costs more; the practical choice is diagonal, one scale per coordinate, estimated from the gradients as they arrive.

Adam evolves the state $(\theta,m,v)$, not $\theta$ alone. It is a history-dependent dynamics, not a discretisation of $\dot\theta=-\nabla\ell$.

*Adam* keeps two running averages per coordinate, of the gradient and of its square, with $m_0=v_0=0$:
$$m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\qquad v_k=\beta_2v_{k-1}+(1-\beta_2)\,g_k\odot g_k.$$
Unrolled, $m_k=(1-\beta_1)\sum_{j\le k}\beta_1^{k-j}g_j$ is an average with geometric weights, whose weights add up to $1-\beta_1^k$ rather than to one because the sum started from zero. Dividing by that sum gives the bias-corrected averages $\hat m_k=m_k/(1-\beta_1^k)$ and $\hat v_k=v_k/(1-\beta_2^k)$, and the update is
$$\theta_{k+1}=\theta_k-\eta_k\,\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon},\qquad\text{that is}\qquad\theta_{k+1}=\theta_k-\eta_k\,G_k\hat m_k,\quad G_k=\diag\Bigl(\frac1{\sqrt{\hat v_{k,i}}+\epsilon}\Bigr),$$
with square root and division taken coordinate by coordinate. So Adam is a diagonal preconditioner applied to a smoothed gradient, the *momentum* $\hat m$, and not to the gradient itself. Note that $v$ estimates the second moment of the gradient, not its variance: nothing is subtracted.

::: aside
Kingma and Ba, [*Adam: A Method for Stochastic Optimization* (ICLR 2015)](https://arxiv.org/abs/1412.6980). The name is short for adaptive moment estimation. The specimen uses $\beta_1=0.9$, $\beta_2=0.95$ and $\epsilon=10^{-8}$; the $\epsilon$ is Adam's own and has nothing to do with the $\varepsilon$ of LayerNorm. Both are written `ε` in Julia, as arguments of different functions.
:::

What does the division do? If the gradient of a coordinate were the same at every step, $g$, then $\hat m_k=g$ and $\hat v_k=g^2$ exactly, for every $k$, thanks to the bias correction, and the step would be
$$-\eta\,\frac{g}{\lvert g\rvert+\epsilon}\approx-\eta\operatorname{sign}(g).$$
In this limit Adam is *sign descent*: every coordinate moves by the same amount $\eta$, in the direction that lowers the loss, whatever the size of its gradient. The magnitude of the gradient is thrown away and only its direction kept. The very first step is always of this kind, since $\hat m_1=g_1$ and $\hat v_1=g_1^2$. Later the averages remember older gradients, and a coordinate whose gradient keeps changing sign gets $\lvert\hat m\rvert\ll\sqrt{\hat v}$ and moves less. @fig-adam compares, for every parameter, the step that gradient descent would take with the one Adam takes.

::: {.figure #fig-adam}
What Adam changes. For each parameter, grouped by array: the size of its gradient (bars), which is how far gradient descent would move it per unit of learning rate, and how far AdamW actually moves it, in the same unit (dots). At the first update every dot sits at one, except those of the key bias, whose gradient is zero up to rounding. [Scrub through the training run.]{.web}
:::

A consequence worth remembering: Adam's step does not change if the loss is multiplied by a constant, while the step of gradient descent scales with it. For gradient descent $\eta$ has the units of $\theta^2$ per unit of loss; for Adam it has the units of $\theta$, and it is roughly the distance each coordinate moves per step. That is why the two need such different learning rates[, which you can confirm in @fig-train-live by switching between them]{.web}.

::: aside
The division is also where rounding enters. The key bias $b_k$ has a gradient that is exactly zero in exact arithmetic (@sec-bk), and in floating point a few units of rounding error. Gradient descent multiplies that by $\eta$ and nothing happens. Adam divides it by its own square root plus $\epsilon$; for a gradient far below $\epsilon$ this is a division by about $\epsilon$, which turns rounding into steady motion. [The readout of @fig-train-live shows how far $b_k$ has strayed from zero, a number that should not exist.]{.web}
:::

**Weight decay.** It is common to pull the parameters gently towards zero, so that they grow no larger than the data demand. One way is to add $\tfrac\lambda2\lVert\theta\rVert^2$ to the loss, which adds $\lambda\theta$ to the gradient; for plain gradient descent the step becomes $\theta\leftarrow(1-\eta\lambda)\theta-\eta g$, a multiplicative shrinking of every coordinate. With Adam the two ideas come apart. Added to the gradient, $\lambda\theta$ passes through the preconditioner and is rescaled coordinate by coordinate, so coordinates with large gradients hardly decay at all. *AdamW* keeps the decay out of the moments and applies it directly:
$$\theta_{k+1}=(1-\eta_k\lambda)\,\theta_k-\eta_k\,G_k\hat m_k.$$
The specimen decays only the entries of its matrices. The gains $\gamma$, the offsets $\beta$ and the biases set scales and zero points, and there is no reason for them to be small.

::: {.code file="src/train.jl" fn="update!"}
:::

::: {.code file="src/train.jl" fn="decay"}
:::

::: aside
Loshchilov and Hutter, [*Decoupled Weight Decay Regularization* (ICLR 2019)](https://arxiv.org/abs/1711.05101). Physically, the decay is a friction towards the origin, $\dot\theta=-\lambda\theta$, integrated by one Euler step and added to the step of the loss; exact integration over a time $\eta$ would multiply by $e^{-\eta\lambda}$ instead.
:::

## Schedules and clipping {#sec-schedule}

Nothing forces the time step to be constant, and in practice it is not. The specimen's run uses a *warmup* followed by a *cosine*: $\eta_k$ rises linearly from zero to $\eta$ over the first $W$ updates, then falls along half a period of a cosine to $\eta_{\min}$ at the last update $K$,
$$\eta_k=\begin{cases}\eta\,k/W, & k\le W,\\[2pt] \eta_{\min}+\tfrac12(\eta-\eta_{\min})\Bigl(1+\cos\pi\,\dfrac{k-W}{K-W}\Bigr), & W<k\le K.\end{cases}$$
Each half has its reason. At the start Adam's averages rest on a handful of gradients, and the first steps, which are sign steps of full length $\eta$ in every coordinate, are taken on the least information; the warmup keeps them short. At the end a large step can only rattle around the minimum: near the bottom of a bowl, gradient descent with step $\eta$ jumps across the bowl by an amount that shrinks with $\eta$, and stochastic gradients add noise of a size that does too. Large steps early and small ones late is the usual compromise between getting somewhere and settling there.

::: {.code file="src/train.jl" fn="warmup_cosine"}
:::

The second safeguard acts on the gradient. In a long run an unlucky minibatch, or a region of high curvature, occasionally produces a gradient far longer than usual, and one such step can undo a great deal of progress. *Clipping* caps its length. Given a radius $c$,
$$\operatorname{clip}_c(g)=\begin{cases}g, & \lVert g\rVert\le c,\\ c\,g/\lVert g\rVert, & \lVert g\rVert>c.\end{cases}$$
This is the Euclidean projection of $g$ onto the ball of radius $c$: of all vectors of length at most $c$, $\operatorname{clip}_c(g)$ is the closest to $g$. For $\lVert g\rVert>c$ and any $v$ in the ball, $\lVert g-v\rVert\ge\lVert g\rVert-\lVert v\rVert\ge\lVert g\rVert-c$, and $cg/\lVert g\rVert$ attains the bound. So clipping shortens a gradient and never turns it. The norm is that of the whole of $\theta$'s gradient at once, all arrays together, so the relative sizes of the parameters' gradients are preserved; and it is applied before Adam's averages see the gradient.

::: {.code file="src/train.jl" fn="clip!"}
:::

@fig-schedule tracks the specimen's learning rate and gradient norm against the clipping radius.

::: {.figure #fig-schedule}
The learning rate and the length of the gradient through the specimen's training run. The rate rises linearly during the warmup and then falls along half a cosine. The gradient would be clipped whenever it is longer than the radius of the ball, which is hatched above the line; in this run it never is.
:::

The clip never acts. The specimen's gradients start shorter than the radius and, with a few bumps, get shorter still, and the clip is in the recipe the way a fuse is in a circuit: for the run in which something goes wrong, which this one is not. The figure also shows the gradient's length falling by orders of magnitude as the loss approaches its floor, while, under Adam, the length of the step does not have to follow it. The next section is about what happens when it doesn't.

## The slingshot {#sec-slingshot}

What does Adam do once the loss has reached its floor? Sign descent gives a hint. Near the bottom the gradient is small, but Adam divides it by its own recent size, and as long as the remaining gradient points consistently in some direction, the step stays of order $\eta$ per coordinate. The gradient flow slows to a halt at a minimum; Adam at a constant learning rate does not. It keeps moving, at nearly constant speed, through a region where the loss is almost flat, until it reaches a place where the curvature is too high for a step of that length. There the loss jumps, the gradient becomes large, $\hat v$ grows with it and shortens the steps, and the parameters fall back towards the floor, from a new place.

@fig-slingshot runs plain Adam from $\theta_0$ for four thousand steps, with no weight decay and no clipping, twice: once at a constant learning rate, and once with the warmup and cosine of @sec-schedule.

::: {.figure #fig-slingshot}
Adam at the floor. The excess of the loss over the entropy floor, on a logarithmic scale, and the length of each update, for four thousand steps of Adam: at a constant learning rate, and with the learning rate falling along a cosine. Each column of hairlines spans the smallest and the largest value over a few consecutive steps, so that no spike is lost. The labelled spikes are the steps at which the loss leaves the floor.
:::

At a constant rate the loss reaches the floor and then, again and again, leaves it: a spike of several orders of magnitude on this scale, followed by a slower return. The length of the update hardly reacts to the floor at all. Under the cosine the steps shrink with $\eta$, and the late spikes are gone; the one early spike comes while the rate is still near its peak. A falling schedule is not only a matter of settling more precisely: it is what stops the optimiser from throwing the parameters out of the minimum it has found.

::: aside
[Thilak, Littwin, Zhai, Saremi, Paiss and Susskind (2022)](https://arxiv.org/abs/2206.04817), *The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon*, found these cycles on small algorithmic tasks and linked them to sudden generalisation (“grokking”). The specimen establishes no such link.
:::

::: aside
The frontispiece of this chapter records the same experiment from many starting points at once; its caption gives the settings.
:::

## Where to start {#sec-init}

Every run starts somewhere, and gradient descent can only improve on its starting point locally. The initial parameters are random, and what matters is their scale. Consider one linear map $y=Wx$ with fan-in $n$, its entries drawn independently with mean zero and variance $s^2$, independent of $x$. Then $\E[y_i\mid x]=0$ and, since the cross terms vanish,
$$\operatorname{Var}(y_i\mid x)=s^2\lVert x\rVert^2=n\,s^2\cdot\frac{\lVert x\rVert^2}{n}.$$
The mean square of the output is $ns^2$ times that of the input. With $s^2=1/n$ the map preserves the size of what passes through it, in the sense of the root mean square of the entries; with a larger $s^2$ every layer amplifies, with a smaller one every layer damps.

The specimen applies this argument map by map. Every entry is drawn independently from a normal distribution, and only the standard deviations need choosing:

- **The maps that read a normalised column**, $W_q,W_k,W_v\in\R^{D\times D}$ and $W_1\in\R^{F\times D}$, have fan-in $D$ and get variance $1/D$, standard deviation $1/\sqrt D$. Their input comes out of a LayerNorm, which at the start has $\gamma=\one$ and $\beta=0$, so its entries have mean square $\lVert\hat x\rVert^2/D$, just below one (@sec-layernorm). The queries, keys, values and the MLP's preactivations therefore start with entries of mean square close to one, and the scores $q\cdot k/\sqrt d$ with variance of order one, the regime of @sec-sqrt-d.
- **The maps that write into the stream**, $W_o$ with fan-in $D$ and $W_2$ with fan-in $F$, get the fan-in variance divided by $2L$: $1/(2LD)$ and $1/(2LF)$. The stream receives $2L$ increments, one from each branch. If they are uncorrelated with the stream and with each other, their variances add like the steps of a random walk: $X_{2L}=X_0+\alpha\sum_kr_k$ with $\operatorname{Var}r_k=c$ has $\operatorname{Var}X_{2L}=\operatorname{Var}X_0+2L\alpha^2c$, and $\alpha^2=1/(2L)$ keeps the total of the increments at $c$ however deep the model. A final projection enters its branch linearly, so the factor is put into its variance. The rectifier halves the mean square of a symmetric input, so $W_2$'s increment starts quieter still; the variance $2/F$ would compensate for that, and the specimen does not bother.
- **The embedding** $E$ has variance one. The character's part of $x_0=Ee_a+P_{:,t}$ then has mean square one per entry, against one half for the positional code, whose pairs contribute $\sin^2+\cos^2=1$ for every two entries: neither drowns the other.
- **The readout** $W_{\text{out}}\in\R^{V\times D}$ has standard deviation $1/D$, variance $1/D^2$, far below the fan-in value $1/D$. It reads the final LayerNorm's output, whose squared length is close to $D$, so each logit starts with variance close to $D\cdot D^{-2}=1/D$. Small, independent logits make a nearly uniform first guess: to second order in the logits the expected loss is $\ln V+(V-1)/(2VD)$ (@ex-init-loss), just above uniform guessing. An untrained model should guess, not be confidently wrong, and the loss at step 0 of @fig-train-live sits just above the line of uniform guessing for that reason.
- **Gains, offsets and biases** start at $\gamma=\one$, $\beta=0$, $b=0$: every LayerNorm begins as the bare projection to the sphere, and every affine map as a linear one.

::: {.code file="src/parameters.jl" fn="initial"}
:::

@fig-init-variance checks the argument by breaking it. It draws the specimen's initial parameters with the book's variances, and with every weight variance except the embedding's multiplied by nine or divided by nine, and measures every intermediate of the first forward pass.

::: {.figure #fig-init-variance}
Why these initial variances. The size of every intermediate of the forward pass at initialisation, for the book's variances and for every weight variance multiplied by nine or divided by nine, one dot per seed. Below: how evenly the attention reads, and the initial loss against uniform guessing.
:::

With the book's variances every station of the forward pass is of order one. Nine times larger, the scores spread out, most attention rows freeze onto a single source (@ch-attention), the branches overwhelm the stream they are meant to correct, and the first predictions are confident and wrong, with a loss above $\ln V$. Nine times smaller, the attention reads everything equally and the branches add almost nothing: the model starts as the embedding alone, a bigram model with a clock. With one block either error is a factor of three in the size of a few intermediates; in a deep model the factors compound from layer to layer.

::: aside
These are statements about the start of training, under the assumption that weights and inputs are independent. Nothing keeps them true afterwards, and the trained specimen has no reason to keep its activations of order one. The variance $1/n$ is usually credited to [LeCun et al. (1998)](https://bottou.org/papers/lecun-98x) and to [Glorot and Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html); the factor 2 that a ReLU calls for, because it zeroes half of a symmetric input, to [He and colleagues (2015)](https://arxiv.org/abs/1502.01852).
:::

## In the code {#sec-learn-code}

One training step is three lines: the loss and its gradient from @ch-pulling-back, the gradient clipped, and the parameters updated by the optimiser.

::: {.code file="src/train.jl" fn="step!"}
:::

Everything else in this chapter is a choice of `opt` and of `η`. The book's run (@fig-train-live with its first settings) is two hundred such steps with AdamW, decay on the matrices, the warmup and cosine of @sec-schedule, and a clip that never acts.

## Exercises {.unnumbered}

::: exercise
Gradient descent on the quadratic $\ell(\theta)=\tfrac12\theta^\T H\theta$, with $H$ symmetric positive definite and eigenvalues $0<\lambda_1\le\dots\le\lambda_p$. Show that $\theta_k=(I-\eta H)^k\theta_0$, and that $\theta_k\to0$ from every start if and only if $0<\eta<2/\lambda_p$. Which learning rate makes the slowest direction converge fastest while keeping the fastest one stable, and what is then the contraction factor per step, in terms of $\kappa=\lambda_p/\lambda_1$?
:::

::: exercise
Show that the update of Adam (with $\epsilon=0$) is unchanged when the loss is multiplied by a constant $c>0$, and that the update of gradient descent is multiplied by $c$. If $\theta$ is measured in metres and $\ell$ in nats, what are the units of $\eta$ for each?
:::

::: exercise
Bias correction. With $m_0=0$ and $g_k=g$ for every $k$, show that $m_k=(1-\beta_1^k)g$. For a gradient that alternates, $g_k=(-1)^kg$, compute $\hat m_k$ for large $k$, and explain why Adam takes small steps along a direction in which the gradient keeps changing sign.
:::

::: {.exercise #ex-init-loss}
At initialisation, condition on the final normalised column $x$ and let $z=W_{\text{out}}x$ with the entries of $W_{\text{out}}$ independent, of mean zero and variance $1/D^2$. Show that the $V$ logits are independent with mean zero and variance $\lVert x\rVert^2/D^2\approx1/D$. Expand $\log\sum_ae^{z_a}$ to second order to show that the expected loss of one prediction is approximately $\ln V+\frac{V-1}{2V}\cdot\frac1D$. Evaluate it for the specimen and compare it with the loss at step 0 of @fig-train-live.
:::

::: exercise
Prove that $\operatorname{clip}_c(g)$ is the unique point of the closed ball $\{\lVert v\rVert\le c\}$ nearest to $g$. Is clipping each array of $\theta$ separately to radius $c$ also a projection onto a convex set? Onto which, and does it preserve the direction of the whole gradient?
:::

::: exercise
[In @fig-train-live, switch to gradient descent and train. Each drag reruns from $\theta_0$: you are comparing runs, not steering one. Then drag $\eta$ upwards until the run learns within two hundred steps, and further until it fails.]{.web} Compare the two learning rates with the sharpness in @fig-step-size.
:::

::: exercise
In @fig-train-live, choose a constant learning rate and keep pressing ▶ until the run has five thousand steps. Does the loss stay on the floor? Compare with @fig-slingshot, which has no weight decay and no clipping, and say which of the two you would expect to make a slingshot rarer.
:::
