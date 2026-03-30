
# Theory Notes for Your Demonstration

## 1. The Text-to-Numbers Problem (Foundation)

Before any algorithm can classify text, it needs to be converted into numbers. This is the fundamental challenge of NLP. Raw text has no mathematical form — a machine cannot directly compute on words. You need to understand that every technique in your pipeline (preprocessing → TF-IDF → classification) is a series of choices about how to represent text as numbers, and every choice involves tradeoffs between simplicity, information loss, and computational cost.

The **Bag of Words** assumption underlies TF-IDF: we completely discard word order and grammar, treating a document as just a multiset of tokens. *"neural network learns weights"* and *"weights learns neural network"* become identical vectors. For paper classification this is acceptable because the vocabulary of a domain (words like *"transformer"*, *"gradient"*, *"embedding"*) is highly diagnostic regardless of order.

---

## 2. TF-IDF — The Core Intuition

**Term Frequency (TF):** How often a word appears in *this* document relative to its total word count.

$$\text{TF}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total terms in } d}$$

**Inverse Document Frequency (IDF):** How *rare* a word is across *all* documents. Rare words are more informative.

$$\text{IDF}(t) = \log\left(\frac{N}{df(t)}\right)$$

Where $N$ = total documents, $df(t)$ = number of documents containing term $t$.

A high TF-IDF weight is reached by a word that has high frequency in a given document but low document frequency across the whole collection — these weights filter out common terms.

**Key things to know:**
- The **log** in IDF is not arbitrary. Without logarithmic scaling, high IDF values would have an astronomical effect. With the log scale, the effects are "smoothed" so TF-IDF values are on a more equal playing field.
- Words like *"the"* or *"is"* appear in every document → IDF approaches 0 → TF-IDF ≈ 0. They are naturally suppressed even without explicit stopword removal.
- The result is a **sparse matrix**: rows = papers, columns = vocabulary words, values = TF-IDF scores. Most entries are zero because any one paper uses only a tiny fraction of the full vocabulary.
- `max_features` in sklearn caps vocabulary size. `min_df` removes words that appear in too few documents (likely noise). These are hyperparameters you chose deliberately.

---

## 3. From Binary to Multinomial Logistic Regression

**Binary logistic regression** outputs a probability for one class vs. another using the **sigmoid function**:

$$P(y=1|x) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$$

The sigmoid squashes any real-valued score into $(0, 1)$.

**Multinomial logistic regression** generalizes this to $K$ classes. Instead of one sigmoid, you use the **softmax function** across all $K$ classes simultaneously:

$$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k \cdot \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j \cdot \mathbf{x}}}$$

A simpler workaround might be to run several separate binary logistic regressions. But that approach breaks down — predicted probabilities across all categories could sum to anywhere from 87.7% to 124% rather than the 100% they should. Multinomial logistic regression avoids this problem entirely by constraining predicted probabilities to sum to exactly 100%.

What this means for your model:
- You learn **one weight vector per class** → for 3 classes you have a weight matrix of shape `[3, vocab_size]`
- Each row answers: *"which words push a paper toward this class?"*
- The softmax turns all three scores into a probability distribution that sums to 1

**Numerical stability note:** When computing softmax, you subtract the maximum value from all logits before exponentiating. This prevents numerical overflow while producing identical outputs — a critical implementation detail when doing it from scratch.

---

## 4. The Weight Matrix and What It Learns

Each class $k$ has its own weight vector $\mathbf{w}_k$ of the same length as your vocabulary. The dot product $\mathbf{w}_k \cdot \mathbf{x}$ is a scalar score: a weighted sum of TF-IDF values for every word in the vocabulary.

- Words with **high positive weight** for class $k$ are strong evidence for that class
- Words with **high negative weight** push the paper *away* from that class
- Words with weight ~0 are irrelevant for that class

This is where interpretability lives. After training, you can rank vocabulary words by their weights per class and immediately understand what the model learned. For research papers, you'd expect words like *"proposed"*, *"framework"*, *"experiments"* for applied papers vs. *"theorem"*, *"proof"*, *"lemma"* for theoretical ones.

---

## 5. Cross-Entropy Loss

You need a way to measure how wrong your model is. For classification, the right measure is **cross-entropy loss**, not mean squared error.

$$\mathcal{L} = -\sum_{k=1}^{K} y_k \log(\hat{p}_k)$$

Where $y_k = 1$ for the true class, 0 for all others. This simplifies to:

$$\mathcal{L} = -\log(\hat{p}_{\text{true class}})$$

**Why not MSE?** MSE treats class outputs as continuous values and doesn't respect the probabilistic structure. Cross-entropy is derived from maximum likelihood estimation — minimizing it is equivalent to finding the weight matrix that makes the training data most probable.

**Intuition for the audience:** When your model confidently predicts the wrong class (predicted probability close to 0 for the true class), $-\log(0^+) \to \infty$ — the loss is enormous. When it's confident and correct (probability close to 1), $-\log(1) = 0$ — no loss. The loss function heavily punishes confident mistakes, which drives the model to be correct rather than just uncertain.

---

## 6. Gradient Descent

You minimize the loss by repeatedly computing the gradient (direction of steepest increase in loss) and stepping in the opposite direction:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \mathcal{L}$$

Where $\eta$ is the **learning rate** — a hyperparameter controlling step size.

Key things to understand:
- **Too high learning rate:** Steps are too large, loss oscillates or diverges
- **Too low learning rate:** Converges correctly but takes very long
- **Convergence:** The loss-vs-epoch curve should decrease and flatten. If it doesn't flatten, you need more epochs. If it oscillates wildly, reduce the learning rate.

A key identity that makes training efficient: the gradient of cross-entropy + softmax simplifies beautifully to just $\hat{p}_k - y_k$ (predicted probability minus true label). You don't need to differentiate through softmax separately.

---

## 7. Evaluation Metrics — Why Accuracy Alone Is Not Enough

| Metric | Formula | What it answers |
|---|---|---|
| **Accuracy** | correct / total | How often are we right overall? |
| **Precision (per class)** | TP / (TP + FP) | When we predict class X, how often is it actually X? |
| **Recall (per class)** | TP / (TP + FN) | Of all actual class X papers, how many did we catch? |
| **F1 (per class)** | 2 × (P × R) / (P + R) | Harmonic balance of precision and recall |

For prediction-focused applications, a confusion matrix is often the most intuitive evaluation tool. It shows you, for each true category, how often the model correctly classified observations and where it made mistakes.

Accuracy is misleading on imbalanced datasets. If 80% of papers are class A, a model that predicts everything as class A gets 80% accuracy without learning anything. Per-class F1 and the confusion matrix expose this.

---

# Video Resources

Here are the best videos to watch, grouped by concept and in the order you should watch them:

---

### 📌 Step 1 — Logistic Regression Foundations
**StatQuest: Logistic Regression — Clearly Explained**
▶️ https://www.youtube.com/watch?v=yIYKR4sgzI8

Josh Starmer's gold-standard explanation. Covers the S-curve, maximum likelihood, and why we predict probabilities rather than classes. Watch this before anything else. StatQuest also has follow-up detail videos on coefficients and maximum likelihood that are worth watching if you want deeper grounding.

---

### 📌 Step 2 — Gradient Descent (the optimizer)
**StatQuest: Gradient Descent — Step by Step**
▶️ https://www.youtube.com/watch?v=sDv4f4s2SB8

Essential. Explains exactly how the weight update rule works visually. Without this, the training loop feels like magic.

---

### 📌 Step 3 — TF-IDF
**TF-IDF Explained (Normalized Nerd)**
▶️ https://www.youtube.com/watch?v=D2V1okCEsiE

Clear visual walkthrough of the formula with a worked example. Good pacing for someone building it from scratch.

---

### 📌 Step 4 — Softmax and Cross-Entropy Loss
**StatQuest: Softmax Derivative Step-by-Step**
▶️ https://www.youtube.com/watch?v=M59JElEPgIg

Covers why softmax is used, the math behind it, and the elegant gradient simplification. Directly relevant to your from-scratch implementation.

**Cross-Entropy Loss — Clearly Explained (deeplizard)**
▶️ https://www.youtube.com/watch?v=6ArSys5qHAU

Intuitive explanation of log loss with visual examples of how it penalizes confident wrong predictions.

---

### 📌 Step 5 — Evaluation Metrics
**StatQuest: Confusion Matrix**
▶️ https://www.youtube.com/watch?v=Kdsp6soqA7o

**StatQuest: Sensitivity and Specificity (Precision/Recall)**
▶️ https://www.youtube.com/watch?v=vP06aMoz4v8

Both are short (~5–10 min), visually clear, and exactly map to what you'll show in your evaluation section.

---

### 📌 Bonus — Big Picture
**3Blue1Brown: But what is a Neural Network?**
▶️ https://www.youtube.com/watch?v=aircAruvnKk

Not directly about logistic regression, but the visual intuition for how weights + activations produce predictions is directly transferable and gives you a strong conceptual anchor if the audience asks "how is this different from a neural network?"

---

> **Study order recommendation:** Logistic Regression → Gradient Descent → TF-IDF → Softmax+Loss → Evaluation. This follows the exact pipeline of your notebook, so watching them in this order will make the entire demonstration feel like a single coherent story when you present it.