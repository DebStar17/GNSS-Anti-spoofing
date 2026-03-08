# GNSS-Anti-spoofing
## Abstract

GNSS spoofing involves sending fake signals to override genuine satellite transmissions. This project tackles the challenge of identifying these anomalies by using an AI model to analyze receiver data with a focus on physics.

## 1. Problem Understanding

To detect GNSS spoofing, we need to find anomalies in the data by comparing it to the physical signatures of real signals. Our research aimed to define what "genuine" looks like so the model can recognize "spoofed" signals.

The task was complex, and we faced several key challenges:

- **Imbalanced Data:** Genuine signals outnumbered spoofed instances by a large margin, which could bias the model.

- **The Blackout Dilemma:** The data contained many signal drops (blackouts). We reasoned that while a single channel losing signal might indicate an obstacle (like a tunnel), a total blackout followed by a "clean" signal often suggests a spoofer’s cover for a takeover.

- **The "Zero Information" Trap:** When a receiver is in a blackout, it provides no physical data. We excluded these instances from training to avoid confusing the model. Instead, we used a **Temporal Density Latch:** if the signal was spoofed in 7 of the 10 seconds leading up to a blackout, we identify the spoofed state through the blackout.

## 2. Feature Engineering

We converted GNSS physics into mathematical features that prioritize the "path" of the signal over static snapshots.

### Core Engineered Features:

- **`c_constellation` (Global Consistency):** This is our most important metric. It calculates the average Doppler shift across all visible satellites. Since a spoofer must eventually move the "whole sky" to change a user's position, a jump in all channels at once signals an attack.

- **`c_pd` (Phase-Divergence Mismatch):** This tracks the relationship between pseudorange rate and carrier phase. In a genuine signal, these should remain in sync. We calculated it as:

    $$c\_pd = |\Delta\text{Pseudorange} - (0.19 \cdot \Delta\text{Carrier\_phase})|$$

    _Note: 0.19m is the L1 carrier wavelength._

- **`c_dd` (Doppler Jump Memory):** We used a `rolling(30).max()` window. This means that even if a spoofer "smooths out" their signal after the initial injection, the model "remembers" the frequency shift from 30 seconds ago.

- **`te_stability`:** This measures the rolling standard deviation of the tracking error. It captures the "physical struggle" (jitter) of the receiver's loops as they deviate from the true signal.

- **`Physics_Score`:** A weighted, non-linear combination that uses $CN_0$ as a penalty multiplier. If the signal is weak, we trust anomalies less.

## 3. Model Architecture

We selected a **Random Forest Classifier** as our engine.

- **Configuration:** `n_estimators=100`, `max_depth=12`, `class_weight='balanced'`.

- **The "Lazy Tree" Battle:** We realized early on that if we provided the model with "raw" features like `time` or `Pseudorange_m`, the trees became "lazy" and would simply memorize the timestamps from the training set instead of learning about physics. By using only our delta-based and rolling-window features, we forced the model to learn the behavior of a spoofer, making it more robust for the test set.

## 4. Training Methodology

- **Sorting Integrity:** We strictly sorted the data by `channel` and then `time` to prevent `.diff()` operations from leaking data between different satellites.

- **Reset Protection:** We developed custom "is_reset" logic to mask features during signal re-acquisition. This stops the model from falsely identifying a situation as "SPOOF!" just because a satellite naturally reappeared from behind a building.

- **Validation:** We used **Weighted F1 Score** (to handle imbalance) and **Average Precision (AP)** for evaluation. We specifically measured these on "Live Signals" to ensure our physics engine was functioning properly.

## 5. Justification of Design Decisions

Our Random Forest feature importance report showed that **`c_constellation`** was the most significant factor, making up over 40% of the model's decisions. This confirms our initial thought: while one satellite might have a noisy jump due to a building reflection, it is physically impossible for eight satellites to jump simultaneously unless a spoofer is involved.

We also purposely clipped and normalized our scores (the `c_` prefixes). This made the data more relatable for the model, ensuring that a $1000\text{m}$ pseudorange jump didn't overshadow a subtle $50\text{Hz}$ Doppler shift. By scaling everything to a 0–1 "suspicion" range, we set up a balanced voting system where every physical parameter had a say.

Lastly, the choice to use a **7/10 Density Latch** for blackouts recognized that spoofing is a process rather than just a single moment. By looking at the 10-second history, we maintain high precision and prevent our predictions from flickering during the noisy periods of an attack.
