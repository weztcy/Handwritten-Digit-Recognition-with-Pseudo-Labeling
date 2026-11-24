# Portfolio Project Data — Handwritten Digit Recognition with Pseudo-Labeling

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/handwritten_digit_recognition_with/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image09.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "handwritten-digit-recognition-with-pseudo-labeling",
  "code": "PRO-29 // ML-14",
  "title": "Handwritten Digit Recognition with Pseudo-Labeling",
  "category": "machine-learning",
  "categoryLabel": "Semi-Supervised Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Handwritten-Digit-Recognition-with-Pseudo-Labeling",
  "license": "MIT License",
  "banner": "/images/projects/handwritten_digit_recognition_with/banner.webp",
  "summary": "Semi-supervised DigitCNN pipeline using confidence-based pseudo-labeling (τ = 0.95) to reach 99.15% MNIST test accuracy with only 10% labeled data.",
  "projectDescription": "Semi-supervised handwritten digit classification system built on PyTorch DigitCNN. Trained on 10% labeled MNIST data (4,800 samples) plus 52,908 high-confidence pseudo-labels from 54,000 unlabeled images, reaching 99.15% test accuracy and reducing classification errors by 44.81%.",
  "architectureSummary": "Compressed IDX Ingestion -> Stratified 10/90 Split -> Baseline DigitCNN -> 3-Round Confidence Thresholding (τ=0.95) & Weighted Retraining (α=0.40) -> Official Test Evaluation & Streamlit UI",
  "techStack": [
    "Python",
    "PyTorch",
    "Torchvision",
    "NumPy",
    "Scikit-Learn",
    "Matplotlib",
    "Seaborn",
    "Pillow",
    "Streamlit"
  ],
  "telemetryMatrix": {
    "projectType": "Semi-Supervised Vision Pipeline",
    "targetPlatform": "PyTorch / Streamlit Web App",
    "primaryDomain": "Handwritten Digit Classification",
    "architecturePipeline": "4-Layer DigitCNN + Self-Training",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Semi-Supervised Image Classification Pipeline",
    "learningParadigm": "Semi-Supervised Learning",
    "modelDomain": "Optical Handwritten Digit Recognition",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Confidence Pseudo-Labeling",
      "Weighted Loss Training",
      "Interactive Streamlit Inference"
    ]
  },
  "mlProblemContext": "Deep neural networks achieve high accuracy on handwritten digit recognition when trained on large annotated datasets, yet manual labeling is costly in real-world document workflows. Using only 10% labeled MNIST data causes generalization gaps, requiring strict confidence filtering and loss weighting to prevent confirmation bias.",
  "mlDatasetOverview": {
    "profileBadge": "MNIST IDX Benchmark",
    "source": "Official MNIST Handwritten Digits (Local Compressed IDX .gz Archives)",
    "modality": "Computer Vision (Grayscale Image Tensor)",
    "scale": "60,000 Training + 10,000 Test Images (28x28 px)",
    "inputShape": "N x 1 x 28 x 28 Float32",
    "dataFormat": "Compressed IDX1 / IDX3 (.gz)",
    "description": "The 60,000 training images are stratified into a 10% labeled pool (4,800 train and 1,200 held-out validation) and a 90% unlabeled pool (54,000 images). Evaluation is performed once on the official 10,000-image test set.",
    "provenanceNote": "Loaded directly from local binary IDX archives via custom struct parser with strict ground-truth isolation on the 54,000 unlabeled pool."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Spatial Pixel & Digit Space",
    "spaceNote": "Single-channel 28x28 normalized pixel intensities across 10 balanced digit classes.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Digit Class Label (0-9)",
      "type": "Multi-Class (10 Classes)",
      "distribution": "Balanced across digits 0-9 (9.04% to 11.24% per class in training set)"
    },
    "inputSignals": [
      {
        "name": "Normalized Grayscale Pixel Grid",
        "type": "Spatial Tensor (1 x 28 x 28)",
        "importanceOrRole": "Primary Visual Input",
        "description": "Raw uint8 pixel intensities [0, 255] linearly scaled to [0.0, 1.0] capturing handwritten digit stroke geometry and topology."
      },
      {
        "name": "Sample Loss Weight (w_i)",
        "type": "Scalar Training Signal",
        "importanceOrRole": "Confirmation Bias Regularizer",
        "description": "Assigns weight 1.0 to genuine labeled training images and penalty weight 0.40 to accepted pseudo-labeled samples during joint optimization."
      },
      {
        "name": "Maximum Softmax Confidence",
        "type": "Posterior Probability Gate",
        "importanceOrRole": "Pseudo-Label Selection Filter",
        "description": "Peak class probability used against threshold τ >= 0.95 to admit reliable unlabeled candidates and reject uncertain borderline strokes."
      }
    ],
    "dataCharacteristicsSummary": "Across the 60,000 training images, digit classes 0 through 9 remain well balanced between 5,421 and 6,742 samples each. Stratified splitting preserves identical class proportions across the 4,800 labeled training samples, 1,200 validation samples, and 54,000 strictly isolated unlabeled images for self-training."
  },
  "mlLearningApproach": {
    "paradigm": "Semi-Supervised Learning via Iterative Self-Training with Confidence-Based Pseudo-Labeling",
    "methodologyDescription": "The pipeline initializes an identical 4-layer DigitCNN on 4,800 labeled samples and runs three iterative self-training rounds. Predictions on the 54,000-image unlabeled pool exceeding 0.95 softmax confidence are absorbed as pseudo-labels and jointly optimized with a 0.40 penalty weight against genuine labels.",
    "justification": "Combining a conservative 0.95 confidence gate with a 0.40 pseudo-loss penalty prevents early misclassifications from shifting decision boundaries while unlocking structural representations from 97.98% of the unlabeled pool without extra manual annotation."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Binary IDX Ingestion & Validation",
      "description": "Parses compressed IDX3 image and IDX1 label archives directly from gzip streams, verifying magic numbers (2051, 2049), array shapes, uint8 types, and class completeness.",
      "badge": "IDX Parser"
    },
    {
      "step": "02",
      "title": "Stratified Split & Label Isolation",
      "description": "Applies StratifiedShuffleSplit (seed=42) to divide 60,000 training images into 4,800 labeled train, 1,200 validation, and 54,000 unlabeled images with hidden labels isolated.",
      "badge": "Zero Leakage"
    },
    {
      "step": "03",
      "title": "Tensor Normalization & Weight Binding",
      "description": "Scales uint8 intensities by 255.0 into float32 (N, 1, 28, 28) tensors and constructs WeightedDataset instances pairing each sample with its genuine (1.0) or pseudo (0.40) weight.",
      "badge": "Tensor Prep"
    }
  ],
  "mlDataPreparationNote": "Unlabeled ground-truth labels (y_unlabeled_hidden) are strictly excluded from training, thresholding, early stopping, and checkpoint selection, and are accessed solely for post-hoc diagnostic reporting.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "DigitCNN (4-Layer Convolutional Neural Network)",
    "architectureSummary": "Two convolutional blocks (Conv2d 3x3 + BatchNorm2d + ReLU repeated twice, MaxPool2d 2x2, Dropout 0.25) scaling 1 -> 32 -> 64 channels, followed by Flatten (3,136), Linear(3136, 128) + BatchNorm1d + ReLU + Dropout(0.40), and a 10-class linear output head totaling 468,458 trainable parameters.",
    "configurationBadge": "468,458 Params",
    "hyperparameters": [
      {
        "name": "Confidence Threshold (τ)",
        "value": "0.95",
        "description": "Minimum softmax probability required to accept an unlabeled image into the pseudo-labeled training pool."
      },
      {
        "name": "Pseudo-Loss Weight (α)",
        "value": "0.40",
        "description": "Gradient penalty applied to accepted pseudo-labeled samples relative to genuine labeled weight 1.0."
      },
      {
        "name": "Self-Training Rounds",
        "value": "3 Rounds (6 Epochs/Round)",
        "description": "Iterative predict-filter-retrain cycles totaling 18 cumulative fine-tuning epochs."
      },
      {
        "name": "Learning Rate & Batch Size",
        "value": "5e-4 / 128 (SSL), 1e-3 / 64 (Baseline)",
        "description": "Lower fine-tuning learning rate stabilizes joint optimization after 15-epoch supervised warm-start."
      },
      {
        "name": "Weight Decay & Seed",
        "value": "1e-4 / Seed 42",
        "description": "Light L2 regularization combined with fixed deterministic seed across NumPy, PyTorch, and Scikit-Learn."
      }
    ],
    "objectiveCriterion": "Sample-Weighted Cross-Entropy Loss",
    "optimizationSolver": "Adam Optimizer (L2 Weight Decay = 1e-4)",
    "trainingProcess": "A supervised baseline is first trained for 15 epochs with Adam (lr = 1e-3) and batch size 64. Self-training then executes up to 3 rounds of 6 epochs each using Adam (lr = 5e-4, batch size 128), tracking held-out validation accuracy across 18 cumulative epochs and reverting whenever a round fails to improve."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Stratified Held-Out Validation & Official 10,000-Image Test Evaluation",
    "description": "Checkpoints are selected strictly on the 1,200-image held-out validation set during training, followed by a single final evaluation on the official 10,000-image MNIST test set comparing 10% baseline, semi-supervised self-training, and 100% upper-bound reference models.",
    "metrics": [
      {
        "label": "TEST ACCURACY",
        "value": "99.15%",
        "benchmark": "Baseline: 98.46%"
      },
      {
        "label": "MACRO F1-SCORE",
        "value": "99.15%",
        "benchmark": "Baseline: 98.45%"
      },
      {
        "label": "MACRO PRECISION",
        "value": "99.16%",
        "benchmark": "Baseline: 98.47%"
      },
      {
        "label": "MACRO RECALL",
        "value": "99.14%",
        "benchmark": "Baseline: 98.44%"
      },
      {
        "label": "ERROR REDUCTION",
        "value": "44.81%",
        "benchmark": "Target > 40.0%"
      },
      {
        "label": "TEST ERRORS",
        "value": "85 / 10000",
        "benchmark": "Baseline: 154"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Live Handwritten Digit Inference Interface",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/handwritten_digit_recognition_with/image01.webp",
      "telemetryNote": "Predicts Digit 3 at 99.19% confidence, exceeding the 95% pseudo-label qualification gate."
    },
    {
      "title": "Model Benchmark & Evaluation Dashboard",
      "badge": "Test KPIs",
      "imageUrl": "/images/projects/handwritten_digit_recognition_with/image02.webp",
      "telemetryNote": "Displays +0.69% absolute accuracy gain (98.46% to 99.15%) and 44.81% test error reduction."
    },
    {
      "title": "Self-Training & Pseudo-Labeling Diagnostics",
      "badge": "SSL Rounds",
      "imageUrl": "/images/projects/handwritten_digit_recognition_with/image03.webp",
      "telemetryNote": "Tracks 52,908 / 54,000 absorbed samples (97.98%) across 3 rounds with τ >= 0.95 and α = 0.40."
    },
    {
      "title": "Dataset MNIST & CNN Architecture Overview",
      "badge": "Data Profile",
      "imageUrl": "/images/projects/handwritten_digit_recognition_with/image04.webp",
      "telemetryNote": "Visualizes balanced 60,000-sample digit counts, 10%/90% stratified split, and loss convergence."
    },
    {
      "title": "MNIST Samples & Test Misclassification Grid",
      "badge": "Error Analysis",
      "imageUrl": "/images/projects/handwritten_digit_recognition_with/image05.webp",
      "telemetryNote": "Contrasts canonical digits 0-9 with representative cases among the 85 residual test errors."
    },
    {
      "title": "Confidence, Accuracy, Round Precision & F1 Composite",
      "badge": "Multi-Metric",
      "imageUrl": "/images/projects/handwritten_digit_recognition_with/image06.webp",
      "telemetryNote": "Post-hoc pseudo-label precision shifts from 99.74% (R1) to 96.80% (R2) and 91.06% (R3)."
    },
    {
      "title": "Supervised Baseline vs. Self-Training Learning Curves",
      "badge": "Convergence",
      "imageUrl": "/images/projects/handwritten_digit_recognition_with/image07.webp",
      "telemetryNote": "Validation accuracy improves from 98.67% baseline peak to 99.67% in Round 3."
    },
    {
      "title": "Stratified Split & Pseudo-Label Class Distribution",
      "badge": "Class Balance",
      "imageUrl": "/images/projects/handwritten_digit_recognition_with/image08.webp",
      "telemetryNote": "Confirms uniform digit proportions across all splits and zero mode collapse during absorption."
    },
    {
      "title": "Baseline vs. Semi-Supervised Confusion Matrices",
      "badge": "Confusion Matrix",
      "imageUrl": "/images/projects/handwritten_digit_recognition_with/image09.webp",
      "telemetryNote": "Off-diagonal misclassifications drop from 154 to 85 across the 10,000 official test images."
    }
  ],
  "mlVisualizationTelemetryNote": "Held-out validation accuracy rises steadily across Rounds 1-3 (99.08% -> 99.33% -> 99.67%) without overfitting, tracking closely with the 99.15% official test set accuracy.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Confidence-based pseudo-labeling raises test accuracy from 98.46% to 99.15% while cutting errors from 154 to 85 across all ten digits. Diagnostic checks confirm 99.58% weighted pseudo-label accuracy against hidden ground truth, though uncalibrated ReLU and BatchNorm layers remain overconfident on blank or noise inputs.",
    "keyDrivers": [
      {
        "factor": "Conservative Confidence Gate (τ = 0.95)",
        "impact": "Maintained 99.74% post-hoc accuracy on 50,880 Round 1 samples and 99.58% weighted accuracy across 52,908 total absorbed images."
      },
      {
        "factor": "Pseudo-Label Penalty Weight (α = 0.40)",
        "impact": "Attenuated gradient magnitude by 60% on pseudo-labeled samples, preventing noisy late-round predictions from destabilizing feature extractors."
      },
      {
        "factor": "Bounded 3-Round Stopping Horizon",
        "impact": "Halted self-training as marginal precision dropped from 99.74% (R1) to 91.06% (R3), avoiding absorption of the final 1,092 ambiguous samples."
      }
    ],
    "limitations": [
      {
        "title": "Out-of-Distribution (OOD) Overconfidence",
        "description": "Uncalibrated BatchNorm and ReLU layers produce >97% softmax confidence on blank all-zero images and uniform random noise, making naive thresholding vulnerable to OOD data."
      },
      {
        "title": "Static Uniform Threshold Across Classes",
        "description": "A fixed τ = 0.95 threshold applies uniformly to all digits without adapting to class-specific difficulty or ambiguous handwriting pairs like 5/3 and 4/9."
      },
      {
        "title": "Clean In-Distribution MNIST Assumption",
        "description": "MNIST digits are centered and background-free; wild document images with clutter or covariate shift would require stronger augmentation and OOD filtering."
      }
    ],
    "futureImprovements": [
      {
        "title": "Curriculum & Class-Adaptive Thresholding",
        "target": "Adopt FlexMatch-style dynamic per-class confidence thresholds to absorb hard classes safely."
      },
      {
        "title": "Consistency Regularization (FixMatch)",
        "target": "Pair weak-to-strong image augmentations with pseudo-labeling to improve robustness under stroke distortion."
      },
      {
        "title": "Temperature Scaling & Energy OOD Detection",
        "target": "Calibrate posterior logits and reject blank or noisy out-of-distribution inputs before pseudo-label admission."
      }
    ],
    "discoveryNote": "Post-hoc diagnostics verified that 97.98% of the unlabeled pool was safely absorbed while staying within 0.14% of the 100%-labeled upper bound (99.29%)."
  },
  "mlModelArtifacts": [
    {
      "name": "semisupervised_model.pt",
      "format": "PT"
    },
    {
      "name": "baseline_model.pt",
      "format": "PT"
    },
    {
      "name": "upper_bound_model.pt",
      "format": "PT"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Semi-Supervised Learning",
    "modelArchitecture": "Four-Layer DigitCNN Network",
    "coreFramework": "PyTorch",
    "validationMethod": "Stratified Holdout Evaluation"
  }
}
```
