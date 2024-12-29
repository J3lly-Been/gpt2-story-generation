# GPT-2 Story Generation

This project fine-tunes the GPT-2 model to generate short stories using the **TinyStories** dataset. The model is trained using 5% of the dataset and evaluated periodically. After training, the model is capable of generating creative and coherent short stories based on input prompts.

## Features
- Fine-tunes the GPT-2 model for generating short stories.
- Uses 5% of the **TinyStories** dataset for training and evaluation.
- Implements periodic evaluations and early stopping to optimize training.
- Generates creative stories with GPT-2 after training.

## Dataset
The dataset used for training is the **TinyStories** dataset, which contains short story text. It is available on the Hugging Face Hub:
- **Training split**: 5% of the original 2,119,719 short stories.
- **Validation split**: 5% of the original 21,990 short stories.

The dataset consists of simple stories that help the model generate creative and unique content.

## Prerequisites
To run this project, you will need:
- Python 3.8 or higher.
- Libraries: `transformers`, `datasets`, `torch`, `accelerate`.

You can install the required libraries with the following:

```bash
pip install transformers datasets torch accelerate
```

## Setup and Usage

1. **Clone the Repository**

   Clone the project repository to your local machine:

   ```bash
   git clone https://github.com/J3lly-Been/gpt2-story-generation.git
   cd gpt2-story-generation
   ```

2. **Download and Prepare Dataset**

   The dataset will be automatically downloaded from the Hugging Face Hub when you run the script. Make sure you have internet access.

3. **Fine-Tuning GPT-2**

   The training script is contained in the `story-generation.ipynb` notebook. You can open the notebook and run the cells step-by-step to fine-tune the GPT-2 model. Here's an overview of the training steps:
   - Load and tokenize the **TinyStories** dataset.
   - Split the dataset into training and validation sets (5% of the original dataset).
   - Set up the model and training configurations.
   - Fine-tune the GPT-2 model.
   - Save the best model and evaluate periodically.

4. **Generate Stories**

   After training, you can use the following code to generate short stories based on a prompt:

   ```python
   from transformers import pipeline

   # Create a text generation pipeline using the fine-tuned model
   story_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

   # Provide a prompt to generate a story
   prompt = "Once upon a time in a magical forest"
   generated_story = story_generator(prompt, max_length=150, num_return_sequences=1)

   # Display the generated story
   print("Generated Story:")
   print(generated_story[0]['generated_text'])
   ```

   Replace `model` and `tokenizer` with the path to your fine-tuned model. The generated story will be a creative continuation of the input prompt.

## Process Followed

The following steps were taken to fine-tune GPT-2 for story generation:

1. **Dataset Loading**:  
   The **TinyStories** dataset was loaded using the Hugging Face `datasets` library. This dataset contains short stories that serve as the source for training the model.

2. **Tokenization**:  
   The GPT-2 tokenizer was used to preprocess the text data. This step converts the text into tokens, which the model can understand. Padding and truncation were applied to ensure that all sequences were of uniform length.

3. **Dataset Subsampling**:  
   Only 5% of the dataset was used for both the training and validation splits. This was done to speed up training and enable quick experimentation.

4. **Model Setup**:  
   The pre-trained GPT-2 model (`GPT2LMHeadModel`) was loaded from Hugging Face's model hub. This model is specifically designed for language modeling and was fine-tuned for the task of story generation.

5. **Training Configuration**:  
   The model was fine-tuned with a batch size of 4, evaluated every 500 steps, and saved at the same interval. Early stopping was enabled with a patience of 2 evaluation steps to prevent overfitting.

6. **Model Training**:  
   The fine-tuning process was executed using the `Trainer` API from the Hugging Face `transformers` library. The model was trained on the 5% dataset, and the best model was saved.

7. **Story Generation**:  
   After fine-tuning, the model was used to generate short stories from a given prompt. The output is a creative continuation of the input prompt, showing the model’s ability to generate coherent stories.

## Training Configuration

- **Training Split**: 5% of the original training and validation data.
- **Evaluation Strategy**: The model is evaluated every 500 steps.
- **Early Stopping**: Early stopping is enabled with a patience of 2 evaluation steps.
- **Model Saving**: The model is saved every 500 steps during training.
- **Checkpoints**: All checkpoints are saved in the `./gpt2-tinystories` directory.

## Notes

- The model is fine-tuned on 5% of the dataset to speed up training and experimentation. You can adjust the dataset size as needed.
- Training time may vary depending on your hardware (GPU/CPU). It's recommended to use a GPU for faster training.

## License
This project is licensed under the MIT License.
