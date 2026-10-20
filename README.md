# Copy Heads Compete with the LM Head for Backbone Capacity

**コピーヘッドは、本体の容量を出力層と取り合う：小規模言語モデルでの副作用の発見と、2つの簡単な対策**

> Status: preliminary results (small scale, 2 seeds, one dataset). Feedback welcome.
> 状態：予備的な結果です（小規模・2シード・1データセット）。

---

## Abstract（要旨）

Copy heads, which add an explicit pointer from the context to the output distribution, were recently shown to improve language-model pretraining (Copy-Paste Language Models, arXiv:2610.19247, Oct 2026). That work reports that gradients flowing from the copy head into the backbone drive the gains and that non-copied tokens are not hurt.

In small models (~3M non-embedding parameters, WikiText-103), we find a different picture: **adding a copy head degrades the LM head's predictions of tokens that do not appear in the context** (first-occurrence tokens), even though overall loss improves a lot. Stopping the copy head's gradient into the backbone removes most of the degradation but also most of the copy gain. We interpret this as the copy path and the LM head **competing for backbone capacity**.

We propose two simple changes that reduce this competition:

1. **Auxiliary LM-head loss**: add the plain next-token loss of the LM head alongside the mixture loss, so the LM head keeps receiving learning signal on copyable tokens.
2. **Mid-layer copy keys/queries**: compute the copy head's queries and keys from an intermediate layer instead of the final layer, so copy-specific features do not reshape the final representation used by the LM head.

Together they improve over a simple pointer-mixture copy head by **+0.027** nats and over our re-implementation of the Copy-Paste LM design by **+0.053** nats (2 seeds, both positive), at no extra inference cost. Applied on top of the Copy-Paste LM design they give **+0.036** nats on both seeds.

コピーヘッド（文中の単語を出力に直接写す仕組み）は、2026年10月の Copy-Paste LM で事前学習を改善すると報告されました。私は小さいモデルで、**コピーヘッドを入れると、文中に無い新しい単語の予測が悪化する**ことを見つけました。原因は、コピーと普通の出力層が本体の容量を取り合うことだと考え、2つの簡単な工夫（普通の出力層の損失の追加、照合を途中の層から行う）で改善しました。

---

## Setup（実験設定）

| Item | Value |
|---|---|
| Data | WikiText-103 (first half of train split), GPT-2 tokenizer |
| Model | 4 layers, d=256, 4 heads, FFN 1024, RoPE, RMSNorm, QK-norm, ReLU², untied embeddings |
| Training | 1500 steps, batch 32 × 256 tokens, AdamW (lr 1e-3, β=(0.9, 0.95), wd 0.1), cosine schedule, fp16 |
| Hardware | Google Colab free tier (T4), ~5 min per run |
| Metric | Validation NLL (20 random batches); "first-occurrence" = target token not present earlier in the window |

## Methods compared（比較した方法）

All copy variants use a copy head with queries/keys of dimension 64 and causal attention over context positions; copy probability for token *v* sums attention over positions where *x_j = v*.

| Name | Description |
|---|---|
| `base` | Plain Transformer |
| `ptr` | Pointer mixture: p = (1−g)·p_LM + g·p_copy, sigmoid gate from final layer |
| `ptr_aux` | `ptr` + auxiliary LM-head loss (λ=1) |
| `iso` | `ptr_aux` with stop-gradient on the copy head's inputs and gate |
| **`mid` (ours)** | `ptr_aux` with copy queries/keys from layer 2 (of 4); gate from final layer |
| `cplm` | Re-implementation of Copy-Paste LM: `<copy>` logit in the LM softmax as gate + learnable sink key, final-layer copy keys, mixture loss only |
| `cplm_ours` | `cplm` + our two changes |

Loss for `ptr_aux`, `iso`, `mid`, `cplm_ours`:

$$\mathcal{L} = -\log\big((1-g)\,p_{\text{LM}}(y) + g\,p_{\text{copy}}(y)\big) + \lambda\,\big(-\log p_{\text{LM}}(y)\big)$$

Inference always uses the mixture distribution.

## Results（結果）

Validation NLL, WikiText-103, lower is better.

| Method | seed 0 | seed 1 | mean | first-occurrence NLL (mean) |
|---|---|---|---|---|
| base | 4.929 | – | 4.929 | 6.199 |
| ptr | 4.655 | 4.648 | 4.651 | 6.392 |
| ptr_aux | 4.629 | 4.646 | 4.637 | – |
| iso (stop-grad) | 4.710 | 4.709 | 4.709 | 6.290 |
| **mid (ours)** | **4.625** | **4.624** | **4.624** | 6.324 |
| cplm (re-impl.) | 4.693 | 4.661 | 4.677 | 6.412 |
| cplm_ours | 4.656 | 4.625 | 4.641 | 6.316 |

Paired differences (same seed):

- `ptr − mid`: +0.030 / +0.024
- `cplm − cplm_ours`: +0.036 / +0.036
- `cplm − mid`: +0.068 / +0.037

## Findings（分かったこと）

1. **Copy heads help a lot at this scale**: −0.28 nats vs. `base` (TinyStories: −0.17).
2. **But they hurt first-occurrence tokens**: 6.20 → 6.39–6.41 NLL. A mass-preserving variant that leaves non-context tokens' probabilities exactly unchanged did *not* remove this, so the degradation is not caused by the mixture weight itself but by the LM head becoming worse.
3. **Copy gradients into the backbone are a double-edged sword**: stopping them (`iso`) recovers part of first-occurrence quality but loses much of the copy gain (consistent with Copy-Paste LM's stop-gradient ablation).
4. **Reducing competition helps**: the auxiliary LM loss and mid-layer copy keys each give ~+0.013–0.014, together ~+0.027 over `ptr`, and the effect transfers to the `cplm` design (+0.036).
5. **Copy-head distillation does not work**: using the copy head only during training and removing it afterwards (a "scaffold") leaves no benefit in the plain Transformer — the gain lives entirely in the extra output path.

## Limitations（限界）

- **Scale**: ~3M non-embedding parameters. Copy-Paste LM (135M–1.7B) reports no degradation on non-copied tokens; the capacity competition we observe may shrink or vanish at larger scale. This is the most important open question.
- **Seeds and data**: 2 seeds, one dataset, 1500 steps.
- **Re-implementation**: `cplm` is our re-implementation from the paper's description, not the official code.
- **Tuning**: λ and the tap layer were not tuned.

## Reproduce（再現）

Open `reproduce.ipynb` in Google Colab (T4), run all cells. Results are cached to Google Drive. ~45 min for 2 seeds × 7 methods.

## References

- Copy-Paste Language Models. arXiv:2610.19247 (2026).
- Merity et al. Pointer Sentinel Mixture Models. 2016.
- See et al. Get To The Point: Summarization with Pointer-Generator Networks. 2017.
- Gu et al. Incorporating Copying Mechanism in Sequence-to-Sequence Learning (CopyNet). 2016.
- Chang & McCallum. Revisiting the Architectures like Pointer Networks to Efficiently Improve the Next Word Distribution. Findings of ACL 2023.
- Lost in Backpropagation: The LM Head is a Gradient Bottleneck. arXiv:2603.10145 (2026).
- Competition of Mechanisms: Tracing How Language Models Handle Facts and Counterfactuals. 2024.

## License

MIT
