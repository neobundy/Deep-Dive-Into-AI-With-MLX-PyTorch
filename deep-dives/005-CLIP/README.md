# Deep Dive into CLIP Part I - Fundamentals

![clip-title.png](images%2Fclip-title.png)

👉 [Part II - Diving Into Codebase](README2.md)

**🏠 Official Website**: https://openai.com/research/clip

**📝 Paper**: https://arxiv.org/abs/2103.00020

The _CLIP (Contrastive Language–Image Pre-training)_ model represents a significant advancement in the field of AI, specifically in bridging the gap between natural language processing (NLP) and computer vision. 

![stable-diffusion.png](images%2Fstable-diffusion.png)

If you've explored Stable Diffusion for generating images and pondered how it adeptly follows both positive and negative prompts to shape its creations, it's essential to recognize the broader principles at play, inspired by models like CLIP. While CLIP itself—a groundbreaking model developed by OpenAI that learns to associate images with natural language descriptions—isn't directly employed in Stable Diffusion, the foundational concepts it introduced have greatly influenced the field. These include the idea of conditioning AI models to understand and generate content based on textual instructions, a key technique that enables Stable Diffusion to create images that align with specific prompts while avoiding undesired elements.

The methodology behind Stable Diffusion's ability to interpret and apply both positive and negative prompts owes much to the advancements in text-image alignment and conditioning techniques pioneered by research on models such as CLIP. This influence has led to the development of latent diffusion models, like that used in Stable Diffusion, which leverage text embeddings to guide the image synthesis process. By understanding the nuances of language, these models can generate images that not only capture the essence of the prompts given but also adhere to the constraints of negative prompts, steering clear of specified undesired outcomes. This dual-prompt conditioning is a testament to the ongoing evolution of AI's capability to bridge the gap between textual descriptions and visual content creation.

To understand the integration of CLIP-like text-image alignment and conditioning techniques in Stable Diffusion from a conceptual standpoint, let's go over a conceptual pseudo code example.. This overview captures the essence of how Stable Diffusion, irrespective of the specific framework or language, utilizes these methodologies to interpret and apply both positive and negative prompts during the image generation process.

```pseudo
Configuration for CLIPTextModel:
- Number of layers: 23
- Model dimensions: 1024
- Number of attention heads: 16
- Maximum sequence length: 77
- Vocabulary size: 49408
```

- The `CLIPTextModel` configuration sets the foundational parameters for the text processing component of the system, dictating its capability to understand and encode the textual inputs provided as prompts.

```pseudo
Class StableDiffusion:
    Initialize model components (UNet, Diffusion, AutoEncoder, etc.)

    Function getTextConditioning(text, n_images, cfg_weight, negative_text):
        Tokenize positive and negative prompts
        If cfg_weight > 1, include negative prompt tokens
        Pad tokens to uniform length
        Encode tokens into embeddings for conditioning
        Repeat embeddings if generating multiple images
        Return conditioning embeddings
```

- **Text Conditioning for Positive and Negative Prompts**: The process begins by taking both positive (`text`) and negative (`negative_text`) prompts. The system uses these inputs to tailor the direction of image generation, encouraging the inclusion of elements specified in the positive prompt and the exclusion of those in the negative prompt.

- **Embedding and Conditioning**: The textual inputs are transformed into a machine-understandable format through tokenization and encoding. The tokens representing both positive and negative prompts are padded to a uniform length and then encoded into embeddings. These embeddings serve as the conditioning information that guides the generative process of the model, ensuring that the output images align with the prompts provided.

In essence, the integration of CLIP-like text-image alignment in Stable Diffusion involves defining a configuration for the text model that can comprehend the complexity of human language. The Stable Diffusion class then orchestrates the entire image synthesis process, incorporating both positive and negative prompts. This dual-prompt strategy allows for nuanced control over the generated images, enabling the model to cater to specific creative directions while avoiding undesired elements. Through tokenization and encoding, the system converts textual descriptions into a form that influences the visual output, showcasing the power of combining language and image understanding in AI-driven art creation.

Here’s an extensive explanation of the key concepts and contributions of CLIP, particularly focusing on the ViT-L/14 visual encoder:

## Background and Motivation

Traditional deep learning approaches to computer vision face several challenges:

- **Data-Intensive Requirements**: Creating comprehensive vision datasets is costly and time-consuming. These datasets tend to cover a limited range of visual concepts, making them less versatile.
- **Task Specificity**: Standard models excel at a singular task but adapting them to new tasks requires considerable effort.
- **Robustness Gap**: Models often perform well in benchmark tests but fail to deliver comparable results in real-world applications.

CLIP addresses these issues by learning from a vast array of images and associated textual descriptions available on the internet. This method allows CLIP to understand and classify images in a much broader context than traditional models.

## Key Features of CLIP

![clip1.png](images%2Fclip1.png)

### 1. **Zero-Shot Learning Capability**

- CLIP can perform various classification tasks without being directly optimized for any specific benchmark. This "zero-shot" learning capability is analogous to the GPT models in NLP, where the model can understand and execute tasks it wasn't explicitly trained for.

### 2. **Natural Language Supervision**

- Leveraging natural language as a source of supervision enables CLIP to generalize across a wide range of visual concepts. This approach is inspired by earlier works that used text to help models understand unseen object categories.

### 3. **Contrastive Training Approach**

![clip2.png](images%2Fclip2.png)

- CLIP uses a contrastive learning objective to connect images with text. Given an image, it predicts the likelihood of various text snippets (out of a large pool) being associated with that image. 

![clip3.png](images%2Fclip3.png)

- This method encourages the model to learn a broad understanding of visual concepts and their linguistic descriptions.

### 4. **Vision Transformer (ViT) Architecture**


- The ViT-L/14 variant of CLIP utilizes the Vision Transformer architecture, offering a more compute-efficient alternative to traditional convolutional neural networks (CNNs) like ResNet. This choice further enhances CLIP's learning efficiency and effectiveness.

### 🧐 The Vision Transformer (ViT)

Vision Transformer (ViT) is a groundbreaking approach in the field of computer vision that adapts the transformer architecture—originally developed for natural language processing tasks—for image recognition challenges. Introduced by Google researchers in 2020, ViT marks a departure from conventional convolutional neural networks (CNNs) that have long dominated image analysis tasks.

**Core Concept:**

ViT treats an image as a sequence of fixed-size patches, similar to how words or tokens are treated in text processing. Each patch is flattened, linearly transformed into a higher-dimensional space, and then processed through a standard transformer architecture. This process involves self-attention mechanisms that allow the model to weigh the importance of different patches in relation to one another, enabling it to capture both local and global features within the image.

**Key Features of ViT:**

- **Patch-based Image Processing:** ViT divides images into patches and processes them as sequences, enabling the use of transformer models directly on images.
- **Positional Embeddings:** Similar to NLP tasks, ViT uses positional embeddings to retain the spatial relationship between image patches.
- **Scalability and Efficiency:** ViT demonstrates remarkable scalability, showing increased effectiveness with larger models and datasets. It can be trained on existing large-scale datasets to achieve state-of-the-art performance on image classification tasks.
- **Flexibility:** The architecture is flexible and can be adapted for various vision tasks beyond classification, including object detection and semantic segmentation.

**Impact:**

The introduction of ViT has spurred significant interest in applying transformer models to a wider range of tasks beyond language processing. Its success challenges the prevailing assumption that CNNs are the only viable architecture for image-related tasks and opens up new avenues for research in applying attention-based models to computer vision.

## Advantages of CLIP

![clip4.png](images%2Fclip4.png)

- **Efficiency**: CLIP achieves competitive zero-shot performance across a wide array of image classification tasks using less computational resources, thanks to its contrastive learning method and the Vision Transformer architecture.
- **Flexibility**: Unlike traditional models that require additional training for new tasks, CLIP can adapt to various visual classification tasks simply by understanding the textual description of the task's concepts.
- **Real-World Applicability**: CLIP shows better alignment with real-world performance compared to traditional models, as it learns from diverse and noisy internet-scale datasets without being confined to the biases of specific benchmarks.

## Limitations and Broader Impacts

While CLIP marks a significant step forward, it also has its limitations. For instance, it may struggle with abstract concepts or systematic tasks like counting. Moreover, the model's performance can be sensitive to the wording of text prompts, highlighting the need for careful "prompt engineering."

The ability of CLIP to learn from any text–image pair found online opens up new possibilities for creating classifiers without specialized training data. However, this flexibility comes with responsibility, as the choice of classes can influence model biases and performance in sensitive areas, such as race and gender classifications.

## Summary

CLIP represents a groundbreaking approach to computer vision, offering a model that learns from the vast amount of visual and textual data available on the internet. By combining the strengths of NLP and computer vision, CLIP addresses significant challenges in the field, such as data-intensive requirements, task specificity, and the robustness gap. Despite its limitations, CLIP's versatility and efficiency make it a valuable tool for a wide range of applications, encouraging further research into multimodal learning and its potential impacts.

## How CLIP Works: A Diagrammatic Representation

**📝 Paper**: https://arxiv.org/abs/2103.00020

![figure1.png](images%2Ffigure1.png)

Let's go over the diagrammatic representation of the CLIP model's approach to learning and applying visual concepts from textual descriptions. It consists of three main parts, illustrating the workflow and capabilities of the model:

1. **Contrastive Pre-training:**
   - This section shows the joint training process of the text encoder and image encoder. A set of images (I₁ to Iₙ) and a corresponding set of textual descriptions (T₁ to Tₙ) are passed through their respective encoders to obtain representations.
   - The goal of training is to predict the correct pairings of images and texts. For example, the text "Pepper the aussie pup" should be matched with the image of that specific dog, not with any other images.
   - The matrix in the middle displays all possible combinations of image and text representations. The correct pairings (e.g., I₁'T₁) are highlighted, and the model learns to associate these correct pairings against the incorrect ones through a contrastive loss function.

2. **Create Dataset Classifier from Label Text:**
   - In this step, the text encoder is used to embed a set of label texts that describe various objects or entities (e.g., plane, car, dog, bird).
   - These label texts are formatted as "A photo of a {object}." The text encoder converts these descriptions into embeddings (T₁ to Tₙ), which act as classifiers for the corresponding visual concepts.

3. **Use for Zero-shot Prediction:**
   - Here, the approach is applied for zero-shot learning, where the model uses the text embeddings as classifiers to make predictions on new images that it hasn't seen during training.
   - An image is passed through the image encoder to get its representation (I₁), which is then compared against the embedded text classifiers (T₁ to Tₙ) from the dataset.
   - The model predicts the class of the new image by selecting the text embedding that is closest to the image's representation in the embedding space. For instance, if the image closely matches the "A photo of a dog" embedding, the model predicts that the image is of a dog.

The diagram effectively illustrates how CLIP can be used to create a flexible image classifier that leverages natural language for zero-shot learning. This means the model can correctly classify images into categories that it wasn't explicitly trained on, simply by understanding the content of the image and comparing it to text descriptions.

## Testing CLIP for Zero-Shot Image Classification

**🏠 Official Repo**: https://github.com/openai/CLIP

Go the the official CLIP repository to access the codebase and detailed instructions for using the model. 

First, you need to install the required dependencies:

```bash
pip install -r requirements.txt
```

Let's take a look at a simple example of using CLIP for zero-shot image classification given the following image and a set of textual descriptions:

![girl-with-puppy.png](code%2Fgirl-with-puppy.png)

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

TEST_IMAGE = "./girl-with-puppy.png"

image = preprocess(Image.open(TEST_IMAGE)).unsqueeze(0).to(device)
text = clip.tokenize(["a puppy", "a girl", "glasses"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
# Label probs: [[0.11664506 0.1568365  0.72651845]]
```

The script compares an image to a set of text descriptions and predicts the probabilities of each text description matching the image. 

1. **Import Libraries**: The script imports the necessary libraries – `torch` for PyTorch functionalities, `clip` for the CLIP model, and `Image` from PIL (Python Imaging Library) to handle image files.

2. **Set Device**: It determines if a CUDA-enabled GPU is available for computation (`"cuda"`), otherwise it falls back to the CPU (`"cpu"`).

3. **Load CLIP Model**: The script loads the CLIP model with the architecture "ViT-B/32" onto the specified device (GPU or CPU). The `preprocess` function is also loaded, which will be used to preprocess the image to match the input format expected by the model.

4. **Image Preparation**: The image located at the path `"./girl-with-puppy.png"` is opened and preprocessed to conform to the input format required by the CLIP model. It's then added to a batch (with `unsqueeze(0)`) and transferred to the device.

When using machine learning models, especially those designed for processing multiple inputs simultaneously, it's crucial to structure single instances in the expected batch format. In this script, the image tensor needs to be formatted to include a batch dimension, which is standard for models that typically process images in batches. The `unsqueeze(0)` function is applied to the image tensor to introduce an additional dimension at the start, transforming the image into a batch with one entry. This step is vital for compatibility with the model's input requirements, as it anticipates inputs with a batch dimension. Failing to add this batch dimension is a common oversight when handling individual images and can lead to errors during model inference.

5. **Text Tokenization**: The script tokenizes a list of text descriptions – "a puppy", "a girl", and "glasses" – using the CLIP model's tokenizer and sends the tokens to the device.

6. **Feature Encoding**: In a no-gradient context (to save memory and computations), the script encodes both the image and the text descriptions into feature vectors using the CLIP model.

7. **Logits Calculation**: The model computes logits (raw prediction scores) for both image-to-text and text-to-image comparisons. 

Logits are the model's raw output scores prior to normalization, reflecting the model's initial confidence levels in its predictions across various classes. To transform these logits into a probabilistic context, where they can be more intuitively understood, the `softmax` function is applied. This function scales the logits so that they sum to 1, effectively converting them into a probability distribution. Within this script, the logits derived from comparing the image to each text description are processed by the softmax function to yield corresponding probabilities. These probabilities reflect the model's assessment of the likelihood that the image corresponds to each provided text description. The higher the probability, the stronger the model's confidence that the image aligns with the given description.

8. **Probability Calculation**: The logits for the image are converted into probabilities using the softmax function, which is a standard way of converting logits to probabilities that sum to 1.

9. **Output**: Finally, the script prints out the probabilities of the image corresponding to each of the text descriptions. Given the structure of the CLIP model, these probabilities represent how well the image matches each description according to the model's understanding.

In the context of the given image, the script is designed to evaluate and return the probabilities that the image matches the descriptions "a puppy", "a girl", and "glasses". The output is a list of probabilities corresponding to each of these labels. According to the output format `[[0.11664506 0.1568365  0.72651845]]`, the model predicts that the image matches the description "glasses" with a high probability (`~72.65%`), "a girl" with a moderate probability (`~15.68%`), and "a puppy" with a lower probability (`~11.66%`). This suggests that the most prominent feature the model recognizes in the image is the glasses.

👉 [Part II - Diving Into Codebase](README2.md)