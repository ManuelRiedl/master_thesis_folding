# Error-Bounded Dynamic Budget Allocation for Folding

### Phase 1: Layer-Wise Profiling (Offline Sensitivity Analysis)
For every layer we calculate the H-Kmeans independently for different pairing rates like [0.035,0.7,0.1,0.135,....,0.5]. After the grouping we calculte the "normalized inerta". This is a metric which defines the "damage/error" which we have after the grouping. The result is a "lookup" table for every layer for every pairing rate with its normalized inerta score. This "damage" score is used in the second phase of the algorithm to assign the pairing rates.
### Phase 2: Global Budget Allocation (Reverse Water-Filling Optimization)
We set an absolute target for the number of parameters we want to remove (or a percentage) and initialize a "error ceiling", which starts at near-zero. We incrementally raise this ceiling step-by-step. At loop-step, we check the lookup table for every layer and select the highest pairing rate of each layer whose "damage" score is below the current ceiling. The assumtion is, that early layers with fewer weights are more sensitive to aggressive paring, and thous have a higher "damage" score. At each loop-condition we check how many parameters/percent we save with the current selected layers with their associated pairing rates  - the loop terminates when we hit the absolute parameter target.


## 2. Algorithmic Pseudocode

### Algorithm 1: Layer-Wise Sensitivity Profiling
#### 1. Global Inputs and Outputs
* **$M$ (Network Model):** The complete neural network architecture (e.g., YOLOv8).
* **$L$ (Target Layers):** The specific list of convolutional layers selected for folding.
* **$P$ (Pairing Rates):** folding/pairing rates to calculate (e.g., `[0.1, 0.2, 0.3]`).
* **$\Omega$ (Profile Map):** resulting "lookup table" storing the calculated mathematical damage for each layer at every tested pairing rate.

#### 2. Layer-Level Extraction (Outer Loop)
* **$\ell$:** The current convolutional layer being evaluated in the loop.
* **$W_{\text{out}}$:** The output weight tensor of layer $\ell$.
* **$\ell_{\text{next}}$:** The next layer in the network that directly consumes the output of $\ell$.
* **$W_{\text{in\_next}}$:** The input weight tensor of $\ell_{\text{next}}$.
* **$A$:** The combined "Alignment Matrix," created by flattening and joining $W_{\text{out}}$ and $W_{\text{in\_next}}$.
* **$N_{\text{channels}}$:** The total number of original output channels (filters) in layer $\ell$.
* **$B_\ell$:** The baseline parameter count (size) of layer $\ell$ before any compression.
* **$Errors_\ell$:** A temporary lookup table storing the damage score for each pairing rate for layer $\ell$.

#### 3. Clustering and Math Variables (Inner Loop)
* **$p$:** The current pairing rate being tested.
* **$K$:** The exact number of channels to keep *after* applying the pairing rate $p$.
* **$Clusters$:** The mathematical groupings of redundant channels found by the HKMeans algorithm.
* **$Inertia$:** The raw "damage" score (sum of squared distances), measuring how much channels had to be distorted to form the new clusters.
---
    1:  foreach layer ℓ ∈ L do
    2:      W_out ← ExtractOutputWeights(ℓ)
    3:      ℓ_next ← IdentifyNextConnectedLayer(ℓ, M)
    4:      W_in_next ← ExtractInputWeights(ℓ_next)
    5:      A ← Concatenate(Flatten(W_out), Flatten(W_in_next))
    6:      N_channels ← GetNumberOfRows(A)
    7:      B_ℓ ← CalculateBaselineParameters(ℓ)
    8:      Errors_ℓ ← ∅
    9:      foreach rate p ∈ P do
    10:         K ← Round(N_channels × (1.0 - p))
    11:         if K ≥ N_channels then
    12:             Errors_ℓ[p] ← 0.0
    13:         else
    14:             Clusters ← ExecuteHierarchicalKMeans(Matrix=A, Centers=K)
    15:             Inertia ← SumOfSquaredDistances(A, Clusters)
    16:             Errors_ℓ[p] ← Inertia / N_channels
    17:         end if
    18:     end for
    19:     Ω[ℓ] ← ⟨Type=ℓ_type, Size=B_ℓ, Curve=Errors_ℓ⟩
    20: end for
    21: return Ω


---

### Algorithm 2: Sensitivity-Aware Global Budget Allocation
**Input:** Profile Map $\Omega$, Target Reduction Ratio $R_{target}$, Step Size $\Delta\epsilon$  
**Output:** Vector of Layer Pairing Rates $\Lambda$  

    1:  TotalScopedParams ← SUM(Ω[ℓ].Size for all ℓ)
    2:  TargetSavings ← TotalScopedParams × R_target
    3:  ε_ceiling ← 0.0001
    4:  Λ ← ∅
    5:  loop
    6:      AccumulatedSavings ← 0
    7:      CurrentIterationMap ← ∅
    8:      foreach layer ℓ ∈ Ω do
    9:          SelectedRate ← 0.0
    10:         foreach ⟨p, error⟩ ∈ Ω[ℓ].Curve do
    11:             if error ≤ ε_ceiling and p > SelectedRate then
    12:                 SelectedRate ← p
    13:             end if
    14:         end for
    15:         CurrentIterationMap[ℓ] ← SelectedRate
    16:         if SelectedRate > 0.0 then
    17:             Saved ← ComputeParamSavingsForLayer(ℓ, SelectedRate)
    18:             AccumulatedSavings ← AccumulatedSavings + Saved
    19:         end if
    20:     end for
    21:     if AccumulatedSavings ≥ TargetSavings or ε_ceiling > 5.0 then
    22:         Λ ← CurrentIterationMap
    23:         break loop
    24:     end if
    25:     ε_ceiling ← ε_ceiling + Δε
    26: end loop
    27: return Λ


---

## 3. Execution Example

**Target:** Save 80,000 parameters.

**Pre-computed Damage Profiles (Phase 1 Output):**
* **Layer A:** Saves 20k at 0.0004 error | 30k at 0.0006 | 40k at 0.0008
* **Layer B:** Saves 20k at 0.0005 error | 40k at 0.0010
* **Layer C:** Saves 50k at 0.0020 error

**Optimization Loop:**

* **Iteration 1 (Ceiling: 0.0001)**
  * Layer A: 0 parameters
  * Layer B: 0 parameters
  * Layer C: 0 parameters
  * **Total Saved:** 0 $\rightarrow$ *Continue*

* **Iteration 2 (Ceiling: 0.0004)**
  * Layer A: 20,000 (error 0.0004 $\le$ 0.0004)
  * Layer B: 0 
  * Layer C: 0 
  * **Total Saved:** 20,000 $\rightarrow$ *Continue*

* **Iteration 3 (Ceiling: 0.0007)**
  * Layer A: 30,000 (error 0.0006 $\le$ 0.0007)
  * Layer B: 20,000 (error 0.0005 $\le$ 0.0007)
  * Layer C: 0 
  * **Total Saved:** 50,000 $\rightarrow$ *Continue*

* **Iteration 4 (Ceiling: 0.0010)**
  * Layer A: 40,000 (error 0.0008 $\le$ 0.0010)
  * Layer B: 40,000 (error 0.0010 $\le$ 0.0010)
  * Layer C: 0 
  * **Total Saved:** 80,000 $\rightarrow$ *Target met. Break loop.*

**Result:** The 80,000 parameter target is met by exclusively compressing Layers A and B. Layer C remains uncompressed, as its lowest error threshold (0.0020) was never reached.

### What Else to Try?
* Fold based on the gradient times the weight? like Gradient magnitude pruning |delta W x W |
* 
* Try it with another YOLO model.
* Is it possible to finish the practical part in June?
