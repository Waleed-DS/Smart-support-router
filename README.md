Vectorization Strategy: Text to Numerical Representation
To transform raw banking queries into machine-learnable features, we have adopted a Dual-Vectorization Strategy. This approach balances the need for high-speed, interpretable baselines with the state-of-the-art semantic understanding required for complex intent classification.

1. The Baseline: TF-IDF (Term Frequency-Inverse Document Frequency)
For our initial benchmarking, we utilize TF-IDF with N-grams (Unigrams and Bigrams).

Why TF-IDF?

Interpretability: It allows us to pinpoint exactly which keywords (e.g., "lost," "fee," "declined") are driving the model's decisions.

Efficiency: It is computationally inexpensive to train and provides near-instant inference, establishing a vital performance "floor" for the project.

N-gram Logic: By using a range of (1, 2), we capture critical phrases like "not working" or "top up" that single words (unigrams) would miss.

2. The Champion: Transformer Embeddings (DistilBERT)
For our final production-ready model, we utilize DistilBERT embeddings via a pre-trained tokenizer.

Why DistilBERT?

Semantic Context: Unlike TF-IDF, Transformers understand word relationships. It recognizes that "I'm not sure why my card didn't work" and "declined card payment" share the same intent, despite having zero overlapping words.

Optimized Performance: DistilBERT is 40% smaller and 60% faster than standard BERT while retaining 97% of its language understanding capabilities. This makes it the ideal "Senior" choice for a high-volume support router.

Handling Sparsity: With an average query length of only 11.9 words, every token is critical. Transformer attention mechanisms are superior at extracting signal from these short, information-dense bursts.


Why is the model struggling?
Entity Bias vs. Action Logic:
In the case of virtual_card_not_working, the model is "over-attending" to the entity (Virtual Card) but missing the negative sentiment or state (the fact it is not working). It defaults to the more common intent associated with that entity, which is acquiring the card.

Shared Environment (ATM Context):
The confusion between card_swallowed and declined_cash_withdrawal occurs because both events share the same physical location and vocabulary (e.g., "ATM," "machine," "money"). The model struggles to distinguish a hardware failure (the machine taking the card) from a software/policy failure (the bank declining the cash) because the contextual "noise" of the ATM is so dominant.

Ambiguity in Request vs. Inquiry:
For card_acceptance, users often ask, "Where can I use my card?". The model frequently misinterprets this as a request for a new physical card (order_physical_card) because both intents revolve around the utility and possession of a physical card asset.

