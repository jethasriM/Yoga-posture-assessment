# AI Yoga Feedback Assistant

An AI-based yoga posture analysis project that combines yoga pose
classification, computer vision, joint-angle estimation, and
rule-based alignment feedback.

The project is designed to help users analyze yoga poses from images
and identify potential alignment concerns based on configurable rules.

## Features

- Yoga pose image classification across dataset classes.
- Pose landmark detection using MediaPipe.
- Left and right knee and elbow angle calculations.
- Manual annotation of reference joint angles.
- Mean Absolute Error (MAE) evaluation for joint-angle estimates.
- Preliminary rule-based alignment feedback.
- Feedback evaluation using precision, recall, F1-score, specificity,
  and a confusion matrix when reviewed reference labels are available.

## Technologies Used

- Python 3.10
- MediaPipe
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (if used by the current implementation)

## Project Structure

```text
Yoga-posture-assessment/
├── dataset/
├── models/
├── reports/
├── src/
│   ├── calculate_angles.py
│   ├── calculate_mae.py
│   ├── annotate_angles.py
│   ├── alignment_feedback.py
│   ├── run_alignment_feedback.py
│   ├── evaluate_feedback.py
│   └── test_alignment_feedback.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

Some files or directories may be optional depending on the current
project implementation.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jethasriM/yoga-model-ai.git
cd yoga-model-ai
```

### 2. Create a virtual environment

Windows:

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a dependency is missing, add the appropriate package to
`requirements.txt` and reinstall the dependencies.

## Dataset

This project uses a yoga pose image classification dataset obtained
from Kaggle.

Download the dataset separately and organize the images according
to the folder structure expected by the project's classification code.

For example:

```text
dataset/
├── pose_class_1/
│   ├── image1.jpg
│   └── image2.jpg
├── pose_class_2/
│   ├── image1.jpg
│   └── image2.jpg
└── ...
```

The class directory names should match the labels expected by the
training and evaluation code.

The dataset itself is not included in this repository. Check the
dataset license and usage conditions before redistributing images.

## Running the Project

Run commands from the project root directory.

### Calculate joint angles

```bash
python -m src.calculate_angles
```

Check the script's command-line options or configuration to specify
the input image or dataset path, if required.

### Annotate reference angles

```bash
python -m src.annotate_angles --image "path/to/yoga_image.png"
```

The annotation tool uses three clicks to define each joint angle.

- Press `S` to save the annotation for the selected joint.
- Press `R` to reset the current selection.
- Press `Q` to quit.

Reference annotations are saved to:

```text
reports/reference_angles.csv
```

### Evaluate joint-angle errors

```bash
python -m src.calculate_mae
```

Configure the input file if required by the current script.

The evaluation compares predicted angles with manually annotated
reference angles and calculates error metrics.

### Generate alignment feedback

Run the feedback pipeline using a CSV of extracted joint angles:

```bash
python -m src.run_alignment_feedback --input path/to/angles.csv --pose "Adho Mukha Svanasana"
```

To save the feedback to a specific file:

```bash
python -m src.run_alignment_feedback --input path/to/angles.csv --pose "Adho Mukha Svanasana" --output reports/alignment_feedback.json
```

The input CSV should contain the columns expected by the script,
including:

```text
image,joint,angle_degrees
```

Check the script's help output for supported arguments:

```bash
python -m src.run_alignment_feedback --help
```

### Evaluate alignment feedback

Prepare a CSV containing reviewed reference labels and model predictions.

Required columns:

```text
sample_id,issue_name,reference_issue,predicted_issue,review_status
```

Use binary labels:

- `0`: The alignment issue is absent.
- `1`: The alignment issue is present.

Run the evaluation:

```bash
python -m src.evaluate_feedback --input reports/feedback_validation.csv
```

Results are written to:

```text
reports/feedback_validation.json
```

Only use labels established through image review. Rows without valid
reviewed labels cannot be used for the evaluation.

## Evaluation

### Yoga Pose Classification

The current baseline classification experiment achieved:

| Metric | Result |
|---|---:|
| Test accuracy | 82.43% |
| Macro precision | 0.84 |
| Macro recall | 0.81 |
| Macro F1-score | 0.81 |

These figures describe the recorded baseline experiment. Results can
vary with dataset splits, preprocessing, and model configuration.

The target classification accuracy is 92.5%. This target has not yet
been achieved by the reported baseline.

### Joint-Angle Estimation

An initial comparison of eight manually paired angle measurements
produced an MAE of approximately 4.36 degrees.

This is a preliminary result based on a very small sample. It should
not be interpreted as proof of general accuracy across yoga poses,
camera angles, or users.

The target mean absolute joint-angle error is below 5 degrees.
Additional annotated samples are needed for a more reliable evaluation.

### Alignment Feedback

The project uses configurable rules to identify potential alignment
concerns, such as selected left-right joint-angle differences.

Feedback evaluation uses:

- Precision
- Recall
- F1-score
- Specificity
- Confusion matrix

These metrics should be reported only after reference labels have
been reviewed and compared with the actual system predictions.

The feedback rules are preliminary and have not been clinically
validated. A triggered rule does not prove that a pose is incorrect.

## Limitations

- Image-based landmark detection can be affected by lighting,
  occlusion, camera position, and body visibility.
- Two-dimensional joint angles do not represent full three-dimensional
  anatomical angles.
- Rule-based feedback uses configurable thresholds and may produce
  false positives or miss genuine alignment concerns.
- Evaluation on a small sample does not establish generalization
  to unseen users and conditions.
- The system is intended for educational experimentation and is not
  a substitute for guidance from a qualified yoga instructor or
  healthcare professional.

## Future Improvements

- Improve pose classification performance.
- Expand the manually reviewed angle-validation dataset.
- Validate feedback rules against independently reviewed examples.
- Add pose-specific alignment thresholds.
- Evaluate performance across different camera positions and users.
- Develop a user interface for easier image analysis and feedback.

## Disclaimer

This project is an educational computer-vision prototype.
Its predictions and alignment suggestions may be inaccurate.
Use appropriate caution when interpreting its feedback.

## License

A license has not yet been specified. Add a license file before
granting others permission to reuse or redistribute the project.