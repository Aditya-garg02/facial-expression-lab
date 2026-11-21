# Model card — ExpressionNet

## Summary

ExpressionNet is a compact residual convolutional neural network for classifying 48×48 grayscale face crops into the seven FER-2013 expression labels: angry, disgust, fear, happy, sad, surprise, and neutral.

## Intended use

Educational experiments in image classification, training-pipeline practice, and carefully scoped research on benchmark facial-expression labels.

## Out-of-scope use

This model must not be used to infer a person's private emotional state, mental health, honesty, intent, competence, or personality. It must not be used for surveillance, employment, education grading, policing, access control, health diagnosis, or any decision that materially affects a person.

## Training data

The training script expects the FER-2013 CSV with `emotion`, `pixels`, and optionally `Usage` columns. FER-2013 contains low-resolution grayscale face images collected in uncontrolled conditions. The dataset's label definitions and collection process impose limitations; labels are not ground truth about internal experience. Review the dataset's current terms before use or redistribution.

## Evaluation

`train.py` uses the official Training/PublicTest/PrivateTest split when all three are available. Otherwise it creates deterministic stratified train/validation/test partitions. It saves held-out accuracy, macro-F1, per-class precision/recall/F1 and support, a confusion matrix, and learning curves.

For the bundled checkpoint, an inference profile selects horizontal mirror averaging for still images using FER-2013 PublicTest validation and evaluates the selected mode on the 3,589-image PrivateTest split. On this checkpoint, still-image accuracy rose from 62.78% to 63.58%, macro-F1 rose from 60.03% to 60.82%, and 15-bin expected calibration error fell from 8.15% to 3.41%. These results use benchmark face crops. The profile records the checkpoint digest, per-class metrics, validation scores, and selected temperatures. The live path uses the original view and its separately fitted temperature.

## Limitations

- Facial movements are context-dependent and do not uniquely identify emotion.
- Performance can vary with pose, lighting, occlusion, age, skin tone, camera quality, and data distribution.
- FER-2013 is small, low resolution, and imbalanced; strong benchmark performance does not imply real-world validity.
- Softmax scores are not calibrated probabilities unless calibration is separately measured.
- Grad-CAM is a coarse saliency view of image regions that affected a score, not a causal explanation.

## Privacy

The Streamlit app processes uploaded images and webcam frames in the local Python process. It does not write them to disk or send them to a remote inference service. Live tracking associates overlapping boxes across frames to smooth scores. It does not keep a database of faces. Normal browser and operating-system behavior is outside the app's control.

## Versioning

The training script stores epoch, seed, image size, class names, and best validation macro-F1 with the model checkpoint. Add dataset provenance, training environment, and the generated test metrics when publishing a trained checkpoint.
