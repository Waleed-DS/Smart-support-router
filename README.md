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



##  Project Overview

*The Business Challenge*
Our customer support team is currently overwhelmed by thousands of incoming queries daily. Relying on human agents to manually read, tag, and route every ticket creates a severe operational bottleneck. This manual process is slow, error-prone, and prevents urgent issues—like fraud or lost cards—from being addressed immediately.

*The Solution*
I engineered a production-ready *Hybrid NLP Router* to automate this triage process. The system classifies incoming text into *77 distinct banking intents* (e.g., `card_payment_fee_charged`, `transfer_into_account`) with a specific focus on safety and reliability.

*Key Architectural Decisions:*
* *Champion Model:* A fine-tuned *DistilBERT* transformer that accurately captures the nuance of messy human language.
* *Safety Layer:* A deterministic *Hybrid Inference Engine* that overrides the model for high-risk queries (e.g., "stolen card"), guaranteeing **100% Recall* for critical security issues.
* *Portability:* Refactored into a modular `src/` package that is environment-agnostic and ready for immediate deployment.


##  Methodology

### 1. Exploratory Data Analysis (The "Why")
Before training, I analyzed the *Banking77* dataset to identify potential pitfalls:
* *Class Imbalance:* Found significant skew between high-volume queries (e.g., "Check Balance") and rare critical issues, necessitating a macro-averaged evaluation metric.
* *Ambiguity Detection:* Bigram analysis revealed heavy semantic overlap between intents like `card_arrival` and `card_delivery_estimate`, motivating the use of a Contextual Model (DistilBERT) over simple keyword matching.
* *Risk Auditing:* Identified that security-critical intents (e.g., `compromised_card`) make up a small but vital portion of traffic, justifying the need for a dedicated *Safety Recall Layer*.

### 2. Text Preprocessing (The "Senior Pipeline")
To prevent Training-Serving Skew, I engineered a unified cleaning pipeline (`src/text_preprocessing.py`) used identically in production:
* *Entity Masking:* All digits are replaced with `<NUM>` (e.g., "lost $500" → "lost <NUM>"). This forces the model to learn the *structure* of a fraud claim rather than overfitting to specific dollar amounts.
* *Domain Normalization:* Banking-specific terms are standardized (e.g., "atm" → "atm", "pin" → "pin") to reduce vocabulary noise.
* *Selective Cleaning:* Standard punctuation is removed, but semantic symbols critical to banking contexts (like `$`, `%`, `#`) are strictly preserved.

### 3. Champion Model Architecture
* *Architecture:* `DistilBERT-base-uncased` (Fine-Tuned).
* *Rationale:* Selected for its balance of *latency vs. performance*. It retains 97% of BERT's accuracy while being 40% smaller—crucial for real-time customer support routing where every millisecond counts.


##  Results & Trade-offs

| Metric | Score | Notes |
| :--- | :--- | :--- |
| *Macro F1-Score* | *~0.85* | Strong performance across 77 diverse classes, significantly outperforming the TF-IDF Baseline (0.76). |
| *Safety Recall* | *100%* | Guaranteed detection for `lost_or_stolen_card` and `compromised_card` via the Hybrid Inference Engine. |

### Error Analysis & Blind Spots
* *Semantic Ambiguity:* The model occasionally struggles to distinguish between intent pairs with heavy lexical overlap, such as `virtual_card_not_working` vs. `getting_virtual_card`. Both contain similar keywords ("virtual", "card"), requiring deeper contextual understanding than DistilBERT sometimes captures in short queries.
* *Conservative Calibration:* On extremely short, ambiguous queries (e.g., "exchange rate"), the model tends to output lower confidence scores (~52%). This is a deliberate trade-off: I prefer the model to be *uncertain* rather than *confidently wrong* on vague inputs, allowing the system to potentially ask the user for clarification.

### Future Improvements
Given more time or compute resources, I would implement:
1.  *Hierarchical Classification:* A two-stage architecture that first predicts the broad Category (e.g., "Cards") and then the specific Intent (e.g., "Lost Card"). This would drastically reduce confusion between similar intents.
2.  *Data Augmentation:* Use Generative AI to create synthetic training examples for the minority classes (intents with <40 samples) to smooth out the class imbalance without simply duplicating rows.
3.  *Active Learning Loop:* Deploy a mechanism where "Low Confidence" predictions are automatically flagged for human review, and those corrected labels are fed back into the training set for continuous improvement.


##  How to Run Locally

Because the fine-tuned model weights are large (>250MB), they are *generated locally* to keep the repository lightweight and fast to clone.

### Step 1. Setup Environment
```bash
# Clone the repository
git clone [https://github.com/Waleed-DS/Smart-support-router.git](https://github.com/Waleed-DS/Smart-support-router.git)
cd "Smart Support router"

# Install dependencies (Core libraries only: PyTorch, Transformers, Scikit-Learn)
pip install -r requirements.txt

Step 2. Generate Model Assets (Critical Step)

You must run the training notebook once to fine-tune DistilBERT and save the model to your local models/ folder.

Open: notebooks/model_training_and_evaluation.ipynb

Action: Run All Cells.

this may take a while for model training.

Step 3. Quick Verification
Once the model is generated, verify the system works by running the included demo script. It tests both the Model Inference and the Safety Override.

python demo.py