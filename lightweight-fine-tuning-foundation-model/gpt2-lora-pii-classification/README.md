---
library_name: peft
base_model: gpt2
---

# Model Card for PII Identification Model

<!-- Provide a quick summary of what the model is/does. -->
This model is fine-tuned from GPT-2 to identify Personally Identifiable Information (PII) in text, specifically focusing on phone numbers and social security numbers.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->
The PII Identification Model is based on GPT-2 and is fine-tuned using the LoRA (Low-Rank Adaptation) technique. It is designed to classify text spans as containing PII or not. The model was trained on the `pii-masking-300k` dataset to recognize PII related to phone numbers and social security numbers.

- **Developed by:** [Your Name]
- **Funded by [optional]:** Self-funded
- **Shared by [optional]:** [Your Name]
- **Model type:** Sequence Classification
- **Language(s) (NLP):** English
- **License:** [Your Chosen License]
- **Finetuned from model [optional]:** GPT-2

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [Link to your repository]
- **Paper [optional]:** Not available
- **Demo [optional]:** Not available

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
The model can be directly used to classify text spans in documents to detect the presence of PII.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->
The model can be integrated into applications that need to ensure privacy by detecting and masking PII in text data.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
The model is not suitable for detecting PII types it was not trained on or for languages other than English. It should not be used for critical applications without further validation.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
The model may not accurately detect all instances of PII, especially if the text format varies significantly from the training data. It may also have biases based on the training data distribution.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->
Users should validate the model's performance on their specific data and be aware of its limitations. Regular updates and retraining with diverse datasets can help mitigate biases.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("path/to/your/model")
model = AutoModelForSequenceClassification.from_pretrained("path/to/your/model")

text = "Your input text here"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print("PII detected" if predictions.item() == 1 else "No PII detected")
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->
The training data comes from the `pii-masking-300k` dataset, which contains text spans labeled for the presence of PII related to phone numbers and social security numbers.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

<!-- If there are any specifics around preprocessing that would be relevant to other practitioners, they should be included here. -->
Text spans were labeled and tokenized using the GPT-2 tokenizer. Irrelevant columns were removed to focus on the necessary text and label columns.

#### Training Hyperparameters

- **Training regime:** Mixed precision (fp16)

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->
Training was conducted over 20 epochs with a batch size of 16, taking approximately [insert time] hours.

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->
The testing data is a subset of the `pii-masking-300k` dataset, specifically the last 1000 samples for testing and the last 500 samples for validation.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->
Evaluation is based on text spans containing phone numbers and social security numbers.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->
The primary evaluation metric is accuracy, which measures the model's ability to correctly classify text spans as containing PII or not.

### Results

The best model achieved an accuracy of [insert accuracy] on the validation dataset after fine-tuning.

#### Summary

The fine-tuned GPT-2 model shows promising results in detecting PII in text data, with a significant improvement in accuracy after applying the LoRA technique.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->
Not available.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** GPU
- **Hours used:** [insert hours]
- **Cloud Provider:** [insert provider if applicable]
- **Compute Region:** [insert region if applicable]
- **Carbon Emitted:** [insert emissions]

## Technical Specifications [optional]

### Model Architecture and Objective

The model architecture is based on GPT-2, fine-tuned for sequence classification to identify PII in text data.

### Compute Infrastructure

#### Hardware

Training was conducted on [insert hardware details, e.g., NVIDIA V100 GPUs].

#### Software

Training used the PEFT library version 0.11.1 and the transformers library.

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]

### Framework versions

- PEFT 0.11.1

---

This content should provide a thorough and detailed description of the model, its development, and its usage.
