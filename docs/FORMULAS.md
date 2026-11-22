# ThermalNet: formulas

Every formula the project uses and a worked example. File and function names in backticks refer to the private source repository (available on request). All numbers in the examples were produced by running the project's own code (not calculated by hand). "Cell" means one square of the model's output grid.

Contents: [1 Input](#1-input-preparation) · [2 Model](#2-inside-the-model) · [3 Reading the output](#3-turning-the-output-into-boxes) · [4 Clean-up after the model](#4-clean-up-after-the-model) · [5 Training loss](#5-training-loss) · [6 Training schedule](#6-training-schedule-and-data-mix) · [7 Data and splits](#7-data-and-splits) · [8 Scoring](#8-scoring-how-good-is-the-model)

---

## 1. Input preparation

**1.1 Gray, contrast-boosted, resized.** Each frame is turned into one gray channel, contrast-equalized with CLAHE (clip limit 2.0, 8×8 tiles), resized to 416×416 and scaled to 0 to 1. Training and live use do the same steps in the same order.
Code: `ThermalNet.predict` in `model/thermal_net.py`, `load_sample` in `train/dataset.py`.

**1.2 16-bit camera data to 8-bit gray.** Some datasets (SDT) store raw temperatures. They are stretched between the 1st and 99th percentile of the image, like a camera's automatic gain:

```
gray = clip( (value − low) / (high − low) × 255 , 0 , 255 )     low = 1st percentile, high = 99th percentile
```

Code: `to_gray8` in `prepare_public.py`.
Example: raw values 29315, 29815, 30315, 30815, 31315 are 20, 25, 30, 35, 40 °C (raw/100 − 273.15). The percentiles are low = 29335 and high = 31295, so the gray values are **0, 62, 127, 192, 255**.

---

## 2. Inside the model

**2.1 Edge detector (first layer, 8 channels).** The gray frame `x` is blurred with a 3×3 kernel `[1 2 1; 2 4 2; 1 2 1] / 16`, then:

```
ex, ey   = Sobel x and y of the blurred frame
grad     = sqrt(ex² + ey² + 0.000001)
lap      = Laplacian of the blurred frame
contrast = x − average of x over a 33×33 window            (local heat contrast)
learned  = 3 trainable 3×3 filters of the blurred frame
output   = tanh( [ex, ey, grad, lap, contrast, learned×3] )  → 8 channels
```

The backbone gets the gray frame plus these 8 channels, so 9 input channels.
Code: `EdgeDetector` in `model/thermal_net.py`.
Example (toy image: a hot 20×10 patch on a cold background): at the patch centre `x = 1.0`, the 33×33 local mean is 0.1837, so **contrast = 0.816**: a hot object on a cooler background gets a strong positive value.

**2.2 ThermalAttention (channel then spatial).**

```
channel weights  = sigmoid( MLP(average-pool(x)) + MLP(max-pool(x)) )      x ← x × channel weights
spatial map      = sigmoid( conv7×7( [mean over channels, max over channels] ) )      output = x × spatial map
```

It lets the model focus on hot, edge-rich channels and places. Code: `ThermalAttention`.

**2.3 Feature pyramid.** Three feature maps (128, 256, 512 channels at 1/8, 1/16, 1/32 of the input) are merged top-down: `f5 = lat5(p5)`, `f4 = lat4(p4) + upsample(f5)`, `f3 = lat3(p3) + upsample(f4)`. For a 416 input the grids are **52×52, 26×26 and 13×13** (`GRID_SIZES = INPUT_SIZE / [8, 16, 32]`). Code: `FeaturePyramid`.

**2.4 Detection heads.** Each cell outputs `5 + number of classes` numbers: `[objectness, x offset, y offset, width, height, class scores…]`. With 19 classes that is 24 channels per cell. The whole model has **11,485,869 parameters** (46 MB of weights).

---

## 3. Turning the output into boxes

**3.1 Decoding one cell** (`ThermalNet._decode`). For the cell in column `gx`, row `gy` of a `W×H` grid:

```
centre x = ( sigmoid(p1) + gx ) / W            centre y = ( sigmoid(p2) + gy ) / H
width    = sigmoid(p3)                         height   = sigmoid(p4)             (fractions of the image)
confidence = sigmoid(objectness) × sigmoid(best class score)
```

Boxes smaller than 15 px or larger than 85% of the frame are dropped.
Example: cell (20, 30) on the 52×52 grid, raw outputs `[2.0, 0, 0, −1.0, 0.5]` and a person class score of 3.0, frame 640×480:
`centre x = (0.5 + 20) / 52 = 0.394`, `centre y = (0.5 + 30) / 52 = 0.587`, `width = sigmoid(−1) = 0.269`, `height = sigmoid(0.5) = 0.622`, `confidence = sigmoid(2) × sigmoid(3) = 0.881 × 0.953 = 0.839`.
Pixel box = **[166, 132, 338, 430]**, class **person**, which is exactly what the code returned.

**3.2 Non-maximum suppression (NMS).** Keep the most confident box, delete every box that overlaps it by IoU ≥ 0.3, repeat (class-agnostic). The live view keeps boxes with confidence ≥ 0.35.

**3.3 Intersection over union (IoU)** = overlap area / combined area of two boxes. Used in NMS, in the loss and in scoring.
Example: a predicted box `(centre 0.50, 0.50, size 0.30 × 0.60)` and a target `(0.52, 0.50, 0.28 × 0.70)` overlap by 0.162 of a combined 0.214, so **IoU = 0.757**.

---

## 4. Clean-up after the model

All in `model/thermal_net.py`.

**4.1 Automatic edge thresholds** (`_auto_canny`). The Canny edge detector's two thresholds come from the crop's median brightness `m`, so they adapt to hot and cold scenes: `lower = 0.67 × m`, `upper = 1.33 × m` (σ = 0.33). Example: `m = 120` gives **80 and 159**.

**4.2 Box refinement inside each box** (`refine_box_with_edges`). Inside the predicted box: CLAHE (clip 3.0, 4×4 tiles) → edges ∪ Otsu hot-region mask → close gaps → largest contour (area > 5% of the box) → its bounding box plus 5 px. The original box is kept if the result is ≤ 10 px or smaller than 15% of the original area.

**4.3 Person check** (`validate_person_box`). Hard rejects: height/width < 0.3; pixel standard deviation < 5; edge density < 0.005. Then three soft **votes** that the box is a flat bright region (a lit wall) and not a person: (1) the hot blob is almost a filled rectangle (extent > 0.90 and solidity > 0.92); (2) more than 60% of pixels are brighter than `max(200, 97th percentile of the frame)`; (3) the inside is less than 8 gray levels brighter than the surroundings. **Two or more votes reject the box.**

---

## 5. Training loss

Total loss per batch (`ThermalNetLoss` in `train/loss.py`):

```
loss = 5 × box loss + 1 × objectness loss + 1 × class loss          (summed over the three scales)
```

**5.1 Which scale learns which object.** With `size = sqrt(width × height)` (fractions of the image): `size < 0.1` → 52×52 grid, `< 0.3` → 26×26 grid, otherwise 13×13. Example: a person 0.28 × 0.69 has size 0.44 → **13×13**; a cup 0.05 × 0.05 → **52×52**; a bag 0.15 × 0.20 → **26×26**.

**5.2 Box loss, Complete IoU (CIoU).**

```
CIoU = IoU − ρ² / c² − α·v          v = (4/π²) · ( atan(w_target/h_target) − atan(w_pred/h_pred) )²          α = v / (1 − IoU + v)
box loss = 1 − CIoU
```

`ρ` is the distance between the two box centres and `c` the diagonal of the smallest box containing both. It rewards overlap, centre distance and matching shape. The box loss is computed on the centre cell of each object.
Example (same boxes as 3.3): IoU 0.757, ρ²/c² = 0.0007, α·v = 0.00003, so **box loss = 0.2437**.

**5.3 Objectness loss, focal loss.** Almost every cell is empty, so plain cross-entropy would learn "nothing is anywhere". Focal loss shrinks the weight of easy cells:

```
focal = α_t · (1 − p_t)^γ · BCE          α = 0.25, γ = 2          p_t = probability given to the true answer
```

The centre cell and its 8 neighbours count as "object" targets. Examples: an easy empty cell (p = 0.047): BCE 0.0486 → focal **0.00008** (600× smaller). A missed object (p = 0.269): BCE 1.313 → focal **0.175**.

**5.4 Class loss.** Binary cross-entropy over the class scores of the object's centre cell.

---

## 6. Training schedule and data mix

Code: `train_two_stage.py`.

**6.1 Learning rate.** Linear warm-up, then cosine decay down to 1% of the start value:

```
warm-up:  lr = lr0 × (step + 1) / warm-up steps
after:    lr = lr0 × ( 0.01 + 0.99 × 0.5 × (1 + cos(π·t)) )      t = progress through the remaining steps
```

Example (stage 1: lr0 = 0.001, 120 epochs × 500 steps, 3 warm-up epochs): step 0 → 6.7e-7; end of warm-up → **1.0e-3**; halfway → **5.05e-4**; last step → **1.0e-5**. Stage 2 (fine-tune) uses lr0 = 0.0001.

**6.2 What each epoch is made of.** Each stage-1 epoch draws 16,000 images with replacement. 15% come from the own A50 data, the other 85% are split over the public datasets by weight (FLIR 1, LLVIP 1, SDT 1, HIT-UAV 0.3, because its drone view makes people tiny):

```
public share of dataset d = 0.85 × weight(d) / sum of weights        draws per image = share / images in dataset
```

Example: weights sum to 3.3, so FLIR/LLVIP/SDT each get **25.8%** (4,121 draws) and HIT-UAV **7.7%** (1,236 draws). Own data gets 2,400 draws from 475 images: each own image is seen about **5 times** per epoch, a public image about 0.3 times. This keeps the large public sets from drowning the small own set.

**6.3 Thermal augmentation** (`train/dataset.py`, training only):

```
zoom       with probability 0.7: scale s ~ uniform(0.5, 1.6) around a random point; boxes less than 40% visible are dropped
flip       horizontal, probability 0.5
contrast   x ← (x − mean) · uniform(0.7, 1.3) + mean + uniform(−0.12, 0.12)
gamma      x ← x ^ uniform(0.75, 1.33)
noise      x ← x + Normal(0, σ),  σ ~ uniform(0, 0.03)
```

---

## 7. Data and splits

**7.1 Scene-based test split** (`train/splits.py`). Neighbouring video frames look almost the same, so a random frame-by-frame split would leak test frames into training. Instead, frames are cut into blocks of consecutive frame numbers (30 for person, 3 for object classes), blocks are ordered by a fixed hash, and whole blocks go to test (15%), val (10%) or train. Training frames within 3 frames of a held-out block are dropped ("guard").
Example: 451 person frames → **307 train, 60 val, 60 test, 24 guard frames dropped**.
New A50 captures (`<class>_a50_NNNN.jpg`) form their own series with blocks of 20, so adding data never moves the existing split.

---

## 8. Scoring: how good is the model

Code: `evaluate.py`. A prediction is **correct (TP)** if it has the right class, its IoU with a not-yet-matched true box is ≥ 0.5, and it is processed in order of confidence. Everything else is a false positive (FP); true boxes nobody found are false negatives (FN).

```
precision = TP / (TP + FP)        recall = TP / (TP + FN)        (at confidence ≥ 0.25, the live setting)
```

**Average precision (AP@0.5)** sweeps the confidence threshold from high to low, tracks (recall, precision) pairs, and takes the area under the precision curve (all-point interpolation; precision at each recall is replaced by the best precision at any higher recall). mAP is the mean AP over classes.
Example (2 true boxes; detections ranked: correct, wrong, correct): recall = 0.5, 0.5, 1.0 and precision = 1.0, 0.5, 0.667, so **AP = 0.5 × 1.0 + 0.5 × 0.667 = 0.833**.
Real result, final model on the held-out A50 test frames, class person: TP 60, FP 5, FN 0 → **precision 60/65 = 0.923, recall 60/60 = 1.00, AP = 1.00**.
