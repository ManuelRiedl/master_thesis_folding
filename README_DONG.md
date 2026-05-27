**1. Minor Accuracy Improvements via Forward Pass Fixes**
`results_save/plots/statistics_comparison_table.png`

After implementing the letterboxing fix and the BN-reset for all layers subsequent to the first folded layer, I achieved an overall improvement in accuracy. However, as seen in the statistics table, the increase is very slight (about 0.02 percentage points). The overall results are still not quite that good.

***

**2. Global Pruning**
`results_save/plots/comparison_Global_Pruned_Calibration_Comparison__1k_vs_5k__20260527_113256.png`

I tested the approach we discussed: using a global pruning algorithm to determine the optimal pruning percentages per layer.

When applying global pruning with a rate of 0.1, we achieved a 16.22% reduction in parameters. However, we lost a tremendous amount of accuracy. The F1-score fell to 0.26 from our baseline of 0.67.

Almost all other YOLOv8 pruning algorithms involves heavy backpropagation (fine-tuning) after the pruning step to recover accuracy. We only use a forward pass to recalibrate statistics. 
* *Lightweight YOLOv8 Real-Time Object Detection via Progressive Pruning and Feature-Aware Knowledge Distillation*
    * **Reduction:** -20.5% parameters via progressive channel pruning.
    * **Immediate Impact:** mAP@0.5 dropped to 0.738.
    * **Recovery:** fine-tuning and knowledge distillation.
    * **Result:** mAP restored to 0.748 (only a 0.5% drop from baseline).
    * (https://ieeexplore.ieee.org/document/11166037/)

* *A Lightweight YOLOv8s Algorithm for Ceiling Fan Blade Defect Detection With Optimized Pruning and Knowledge Distillation* 
    * **Reduction:** -76.6% parameters via sparse training and pruning.
    * **Recovery:** extensive fine-tuning and knowledge distillation.
    * **Result:** Accuracy restored to an mAP of 0.976.
    * (https://ieeexplore.ieee.org/document/11021420/)
***

**3. Global Pruning as a Folding Template**
`results_save/plots/comparison_Pruned_Ratio_Repair_Folding_Comparison_20260527_134616.png`

Next, I extracted the exact layer-wise percentages dictated by the pruning algorithm and applied them as pairing rates for our folding algorithm. *(I skipped the SPPF block since I haven't written a folding handler for it yet, and I protected the earliest conv blocks to preserve basic spatial feature extraction).*

Using the pruning ratios per layer, the folding resulted in a 6% parameter reduction with an F1 score of 0.57 (down from the 0.6791 baseline).

Interestingly, this proves that the pruning algorithm's suggestions are **not optimal** - in my opinion - for our folding approach. We already have a manual config that achieves a similar 6.5% reduction but yields a much higher F1 score of 0.62 (Reference: `results_save/plots/comparison_conv4_to_conv8_Full_compare_of_folded_Versions_20260522_172020.png`). 
***

**Question:**
Should I also use backpropagation for the pruning benchmark? 

On one hand, it feels like an unfair comparison because our folded versions do not use backpropagation (we only do a data-driven forward pass). Under these backprop-free conditions, folding outperforms pruning, which can be seen clearly here:
`results_save/old/plots_old/conv4_to_conv8/comparison_conv4_to_conv8_compare__Structural_Folding_vs__Pruning_20260513_150631.png` *(This is an older plot, but the conclusion remains valid since our recent fixes only increased the folding accuracy further).*

On the other hand, a purely forward-pass pruning model is not used much today - it is normally combined with a backpropagation.

Maybe we could move the meeting up by a few days next week—if not, that's fine too—I'll just try a few other things :)
Thank you very much!!
Lg Manuel