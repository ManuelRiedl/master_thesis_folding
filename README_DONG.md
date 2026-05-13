Hello Dong - First of all, thanks for your time :)

**INFO**
Just for your info: I used the COCO Train set, and as a Testing set, I used the COCO Validation set (5,000 images) from 2017 (https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset/data). I used the val set as testing because the real test set labels are not public. Mutliple other papaers also used the val set as a metric - so I guess that should be fine. I used this pruning library for the baseline: https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/yolov8_pruning.py.

In the `results_save/plots` directory, I have tested a few combinations of layers and compared the folding against the pruning with different pairing rates (0.1, 0.2, 0.3) and calibration sets (1k, 5k, 20k). When I compared folding with pruning, I always folded/pruned the exact same layers.

***

**1. Simple Conv Layers (conv5 and conv7)**
 `results_save/plots/conv5_conv7/comparison_conv5_conv7_Full_compare_of_folded_Versions_...png`

For a pairing rate of 0.1 (which means a 1.21% weight reduction), the basic folded version got the best F1-Score—better than with Forward Calibration REPAIR. Contradictorily, the more calibration images I used (1k, 5k, 20k), the worse the F1-score got. This phenomenon can also be seen in the other folding configs later.

Overall, I guess the folding worked fine for the basic conv block, but it isn't that good overall. E.g., for a pairing rate of 0.3 (3.63% weight reduction), I got an F1 score of 0.6571, which is around 2.3 percentage points less than the basic model (YOLOv8m - 0.6791). An accuracy decrease of over 2% for only a 3.6% weight decrease doesn't seem that good to me.

 `results_save/plots/conv5_conv7/comparison_conv5_conv7_compare__Structural_Folding_vs__Pruning_...png`

When I compare the folded version to the pruned version, I can see that for the pruned version, the repair performs better than the basic one (on other layers/configs this is more pronounced). However, the folded version with no repair scored first in the F1 scores on all pairing/pruning levels (0.1, 0.2, 0.3).

***

**2. C2F Block with a Normal Conv Layer (conv4 and conv5)**
 `results_save/plots/conv4_conv5/comparison_conv4_conv5_compare__Structural_Folding_vs__Pruning_...png`

Similar things can be seen here. When I take a look at the comparison of folded and pruned, I can see that for 0.1, folded without repair won. For the other two rates (0.2 and 0.3), folded with 1k calibration images won. 

The pruned versions performed worse than the folded ones. But I have to say that the reduction rate is not equal, so this is not a general assumption. E.g., for a pairing/pruning rate of 0.1, it is a 0.88% reduction for folded and a 1.29% reduction for the pruned versions.

***

**3. Multiple Layers at Once (conv4 until conv8)**
 `results_save/plots/conv4_to_conv8/comparison_conv4_to_conv8_compare__Structural_Folding_vs__Pruning_20260513_150631.png`
Lastly, I tried to fold over multiple layers at once (conv4 to conv8). But the results are also not that great in my opinion. 

E.g., at a pairing/pruning rate of 0.1, I got roughly a 6.5% reduction, and again the basic folded version (with no repair) got the highest F1 score. With increasing calibration images, the score got worse.

For a pairing/pruning rate of 0.3, I got a 17.92% reduction for the folded version and a 19.85% reduction for the pruned version. Here again, the best F1-score was the folded one with only 1k calibration images. I also tried to calibrate with 60k here, but it did not change the outcome. However, it got a better score than the 20k version.

***

**My Question:**
I am confused that the forward pass calibration does not increase the accuracy in the folded versions (in the pruned versions, the forward pass does work). 

Can you maybe take a look at it to see if I did something wrong? The function is `repair_bn_forward_pass`. I tried to comment everything in `folding_main.py` as well as I could (with the references to the paper) so it is clear what I did.

Of course, we can also just talk about this in the meeting... But I'm a little at a loss as to what else I should try.

Thank you very much!!

Lg Manuel