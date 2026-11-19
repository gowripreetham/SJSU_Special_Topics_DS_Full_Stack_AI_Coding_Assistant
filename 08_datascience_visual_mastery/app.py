"""Project 08 — DS Visual Foundations: intuition → rigorous math → live simulation → quiz, for four core ideas.

All simulators run client-side (web/custom.js), so `python export_static.py` turns this into a GitHub Pages site.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403


def q(text: str, options: list[str], answer: int, why: str) -> dict:
    return {"q": text, "options": options, "answer": answer, "why": why}


def interview(items: list[tuple[str, str]]) -> dict:
    return HTML("".join(f'<details style="margin:8px 0"><summary style="cursor:pointer;font-weight:600">{i + 1}. {qq}</summary><div class="md" style="margin:6px 0 0 18px;color:var(--ink2)">{a}</div></details>' for i, (qq, a) in enumerate(items)),
                title="Interview prep — click a question to reveal a model answer")


LESSONS = {
    "nb": ("Naive Bayes", "Turning word counts into a probability of spam.", """
## Intuition
Every word in a message is a small piece of evidence. *"free"* shows up far more often in spam than in normal mail, so seeing it should raise our suspicion; *"meeting"* should lower it. Naive Bayes simply **adds up the votes**.

## The math
Bayes' rule for a class *c* and words *w₁…wₙ*:

`P(c | w₁…wₙ) ∝ P(c) · P(w₁…wₙ | c)`

The **naive** assumption — words are conditionally independent given the class — factorises the likelihood:

`P(c | w₁…wₙ) ∝ P(c) · ∏ᵢ P(wᵢ | c)`

In log space (to avoid underflow) the decision becomes a sum:

`log-odds(spam) = log P(spam)/P(ham) + Σᵢ log P(wᵢ|spam)/P(wᵢ|ham)`

Each likelihood is estimated with **Laplace smoothing**: `P(w | c) = (count(w, c) + α) / (N_c + α·|V|)`, so an unseen word never produces a zero.

## Try it
Change the message, then drag **α to 0** and type a word that only appears in one class (e.g. *prize*) to watch the zero-frequency problem veto everything else.
""", "nb", [
        q("Why is Naive Bayes called *naive*?", ["It ignores the prior", "It assumes features are conditionally independent given the class", "It only works for two classes", "It uses a linear kernel"], 1, "The independence assumption is rarely true (\"free\" and \"prize\" co-occur) but the classifier is often still accurate because only the *ranking* of classes matters."),
        q("With α = 0, a test message contains a word never seen in ham. What is P(ham | message)?", ["Unchanged", "Exactly 0", "0.5", "1"], 1, "One zero factor zeroes the whole product — the zero-frequency problem that Laplace smoothing fixes."),
        q("Why do implementations sum log-probabilities instead of multiplying probabilities?", ["Logs are more accurate", "To avoid floating-point underflow on long documents", "Because Bayes' rule requires it", "To make the model non-linear"], 1, "Hundreds of probabilities < 1 multiplied together underflow to 0.0; sums of logs do not."),
        q("Raising the prior P(spam) from 0.4 to 0.8 does what to the decision?", ["Nothing", "Adds a constant to the spam log-odds, so less word evidence is needed", "Changes the word likelihoods", "Removes the need for smoothing"], 1, "The prior is the starting vote; likelihoods are untouched."),
    ], [
        ("When does Naive Bayes beat logistic regression?", "With very little training data: NB (generative) reaches its asymptotic error faster, while LR (discriminative) has lower asymptotic error (Ng &amp; Jordan, 2001). Also when training speed matters — NB is one pass of counting."),
        ("Are Naive Bayes probabilities trustworthy?", "No — correlated features are double-counted, pushing outputs toward 0 or 1. The ranking is good, the calibration is poor; apply Platt scaling or isotonic regression if you need probabilities."),
        ("How do you handle continuous features?", "Gaussian NB models each feature as a per-class normal; alternatively discretise, or use kernel density estimates."),
    ]),
    "eval": ("Model evaluation", "Confusion matrix, Type I / II errors, ROC-AUC, cost matrix and the precision–recall trade-off.", """
## Intuition
A classifier outputs a **score**; a **threshold** turns the score into a decision. Slide the threshold right and you raise fewer false alarms but miss more real cases. There is no free lunch — only a trade-off you choose according to what each mistake *costs*.

## The math
| | predicted + | predicted − |
|---|---|---|
| **actual +** | TP | FN — *Type II error* (miss) |
| **actual −** | FP — *Type I error* (false alarm) | TN |

- **Precision** = TP / (TP + FP) — when I raise an alarm, how often am I right?
- **Recall / TPR** = TP / (TP + FN) — of the real positives, how many did I catch?
- **FPR** = FP / (FP + TN) — the Type I error rate α. **1 − recall** is the Type II rate β.
- **ROC curve**: TPR vs FPR for every threshold; **AUC** = P(score of a random positive > score of a random negative).
- **Expected cost** = FP·c_FP + FN·c_FN. For a calibrated model the optimal threshold is `c_FP / (c_FP + c_FN)`.

## Try it
1. Set prevalence to 2% — accuracy stays ~98% while precision collapses. ROC barely moves; the PR curve tells the truth.
2. Make a false negative 20× as costly as a false positive and watch the cost-optimal threshold slide left.
""", "evalsim", [
        q("A cancer screen flags a healthy patient. Which error is this?", ["Type I (false positive)", "Type II (false negative)", "Both", "Neither"], 0, "Rejecting a true null hypothesis ('patient is healthy') is a Type I error."),
        q("Prevalence drops from 30% to 2% with the same model and threshold. What happens?", ["Recall drops", "FPR drops", "Precision drops", "AUC drops"], 2, "TPR and FPR are conditioned on the true class, so they (and ROC-AUC) do not change; precision mixes the classes and collapses when negatives dominate."),
        q("ROC-AUC of 0.5 means…", ["Half of predictions are correct", "The model ranks no better than chance", "Precision equals recall", "The threshold is 0.5"], 1, "AUC is the probability a random positive outranks a random negative."),
        q("Missing fraud costs $500, a false alert costs $5. Compared with 0.5, the threshold should be…", ["Higher", "Much lower", "Exactly 0.5", "Irrelevant"], 1, "Optimal threshold ≈ 5 / (5 + 500) ≈ 0.01 for a calibrated model: flag almost anything suspicious."),
    ], [
        ("When would you prefer PR-AUC over ROC-AUC?", "Heavily imbalanced problems where the positive class is what matters (fraud, rare disease). ROC-AUC is dominated by the abundant true negatives and can look excellent while precision is terrible."),
        ("Is accuracy ever the right metric?", "When classes are balanced and both error types cost the same. Otherwise report a cost-weighted metric or precision at a required recall."),
        ("How do you choose a threshold in production?", "On a validation set (never the test set): either minimise expected cost with business-supplied error costs, or fix an operating constraint such as 'recall ≥ 95%' and maximise precision subject to it. Re-check when prevalence drifts."),
    ]),
    "gd": ("Derivatives → gradient descent", "From the slope of a secant line to an optimiser that trains every neural network.", """
## Intuition
The derivative answers: *if I nudge x a tiny bit, how much does f(x) change?* Geometrically it is the slope of the tangent line. If the slope is positive, moving left goes downhill; if negative, moving right does. **Gradient descent** just repeats that observation.

## The math
The derivative is the limit of secant slopes:

`f′(x) = lim (h→0) [ f(x + h) − f(x) ] / h`

A first-order Taylor expansion, `f(x + Δ) ≈ f(x) + f′(x)·Δ`, says choosing `Δ = −η·f′(x)` gives `f(x + Δ) ≈ f(x) − η·f′(x)² ≤ f(x)`. Hence the update rule

`xₜ₊₁ = xₜ − η · f′(xₜ)`

For `f(x) = x²` this is `xₜ₊₁ = (1 − 2η)·xₜ`: it converges iff `|1 − 2η| < 1`, i.e. **0 < η < 1**. In many dimensions f′ becomes the gradient ∇f and the rule is identical.

## Try it
- Shrink **h** and watch the dashed secant rotate onto the tangent.
- On x², set η = 0.9 (oscillates), then η = 1.05 (diverges).
- On the two-valley function, start at x₀ = −1.8 and then x₀ = 1.8: same algorithm, different minimum.
""", "gdsim", [
        q("For f(x) = x², what happens with learning rate η = 1?", ["Converges in one step", "Oscillates forever between x₀ and −x₀", "Diverges", "Does not move"], 1, "xₜ₊₁ = (1 − 2·1)xₜ = −xₜ: it jumps to the mirror point each step."),
        q("The gradient at the current point is exactly 0. Gradient descent will…", ["Stop moving", "Speed up", "Reverse direction", "Diverge"], 0, "The update is −η·0. This could be a minimum, a maximum or a saddle point — the algorithm cannot tell."),
        q("Why does gradient descent find different answers from different starts on a non-convex function?", ["Numerical error", "It follows the local slope into whichever basin it starts in", "The learning rate changes", "It is random"], 1, "GD is a local method; only convex functions guarantee the global minimum."),
        q("As h → 0, the secant slope approaches…", ["0", "f(x)", "The derivative f′(x)", "Infinity"], 2, "That limit is the definition of the derivative."),
    ], [
        ("Why use stochastic gradient descent instead of full-batch?", "Each step costs O(batch) instead of O(N); the gradient noise acts as a regulariser and helps escape saddle points and sharp minima. Mini-batches balance noise against vectorisation."),
        ("What do momentum and Adam add?", "Momentum averages past gradients to damp oscillation across steep directions and accelerate along shallow ones. Adam additionally rescales each parameter's step by a running estimate of its gradient variance."),
        ("How do you pick a learning rate?", "A learning-rate range test (increase η exponentially, pick just below where loss blows up), then a schedule — warm-up followed by cosine or step decay."),
    ]),
    "bp": ("Chain rule → backpropagation", "How one rule from calculus computes millions of gradients in a single backward sweep.", """
## Intuition
A network is a chain of simple functions. To know how the loss reacts to a weight deep inside, ask each link how much it amplifies a small change — and **multiply the answers along the path**. Backpropagation does that multiplication from the output backwards, reusing every shared piece.

## The math
For a composition `L = f(g(h(w)))` the chain rule gives

`dL/dw = (dL/df)·(df/dg)·(dg/dh)·(dh/dw)`

For the single neuron below — `z = w·x + b`, `a = σ(z)`, `L = (a − t)²` — the local derivatives are

`∂L/∂a = 2(a − t)`  ·  `∂a/∂z = σ(z)(1 − σ(z))`  ·  `∂z/∂w = x`  ·  `∂z/∂b = 1`

so `∂L/∂w = 2(a − t) · a(1 − a) · x`. Backprop computes `δ = ∂L/∂z` **once** and reuses it for both w and b; in a deep net each layer's δ is built from the next layer's, which makes the cost of *all* gradients about the same as one forward pass.

## Try it
- Push **w** to 3 with x = 3: the sigmoid saturates, σ′(z) → 0 and the gradient vanishes although the loss is large.
- Click **Gradient step** repeatedly and watch the loss fall.
""", "backprop", [
        q("In the neuron above, which quantity is shared by ∂L/∂w and ∂L/∂b?", ["x", "δ = ∂L/∂z", "The target t", "The learning rate"], 1, "Both are δ times a local derivative (x for w, 1 for b). Reusing δ is the whole point of backprop."),
        q("If x = 0, what is ∂L/∂w?", ["Equal to ∂L/∂b", "0 — the weight had no effect on the output", "Infinite", "Equal to the loss"], 1, "∂z/∂w = x = 0, so the chain is cut. (The bias still gets gradient.)"),
        q("Why do sigmoid networks suffer vanishing gradients?", ["σ′(z) ≤ 0.25, and many such factors are multiplied through depth", "Sigmoids are not differentiable", "The loss is always zero", "Weights are too large"], 0, "Each layer multiplies the backward signal by at most 0.25, so it shrinks geometrically."),
        q("Backprop's cost relative to a forward pass is roughly…", ["Exponential in depth", "One forward pass per parameter", "A small constant multiple", "Zero"], 2, "Reverse-mode autodiff computes all partial derivatives of a scalar loss in O(forward cost)."),
    ], [
        ("What is the difference between forward- and reverse-mode autodiff?", "Forward mode pushes one input's perturbation through the graph — cost ∝ number of inputs. Reverse mode (backprop) pulls one output's sensitivity back — cost ∝ number of outputs. A scalar loss with millions of weights makes reverse mode the obvious choice."),
        ("How do residual connections help gradients?", "y = x + F(x) has Jacobian I + ∂F/∂x, so the identity path carries gradient unchanged through any depth instead of a product of small factors."),
        ("How would you check a hand-written backward pass?", "Gradient checking: compare each analytic gradient to the centred finite difference (f(θ+ε) − f(θ−ε)) / 2ε and require a relative error below ~1e-6."),
    ]),
}


def lesson(key: str):
    title, sub, text, sim, quiz, iv = LESSONS[key]
    return lambda: [H(title, sub), MD(text), CUSTOM(sim, title="Live simulation"), QUIZ("Check your understanding", quiz), interview(iv)]


HOME = """
Four ideas carry most of introductory machine learning. Each module follows the same path:

1. **Intuition** in plain language
2. **Rigorous math** — the actual formulas, nothing hidden
3. **Live simulation** — drag sliders and watch the math respond
4. **Quiz** with explanations, plus **interview questions** with model answers

| Module | The big question | Connects to |
|---|---|---|
| Naive Bayes | How do I turn counts into a probability? | Bayes' rule, smoothing, log-odds |
| Model evaluation | Which mistakes can I afford? | Confusion matrix, ROC, PR, cost |
| Derivatives → gradient descent | Which way is downhill, and how far do I step? | Limits, Taylor expansion, learning rate |
| Chain rule → backpropagation | Who is responsible for the error? | Composition, local derivatives, vanishing gradients |

Everything runs in your browser — this course is also published as a static site (see `docs/visual-mastery/`).
"""

app = make_app(__file__, "08", "DS Visual Foundations", "Intuition · rigorous math · live simulation · quizzes for four core ideas", [
    ("home", "Start here", "Course", lambda: [H("Data science, visually", "A mini-course for beginners who want both the intuition and the equations."), MD(HOME)]),
    ("naive-bayes", "1 · Naive Bayes", "Modules", lesson("nb")),
    ("evaluation", "2 · Model evaluation", "Modules", lesson("eval")),
    ("gradient-descent", "3 · Gradient descent", "Modules", lesson("gd")),
    ("backprop", "4 · Backpropagation", "Modules", lesson("bp")),
])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008)
