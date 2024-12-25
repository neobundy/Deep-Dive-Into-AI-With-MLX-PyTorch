# Chapter 15 -Tenny, the Transformer Sentiment Analyst with an Attitude Is Born 
![tenny-with-an-attitude.png](images%2Ftenny-with-an-attitude.png)
With the creation of our unique dataset for a cynic language model, we're venturing into a realm where traditional supervised learning methods aren't entirely applicable. Our dataset, exemplified by entries like this:

```text
{"text": "Finally, a smartwatch that tracks every breath you take. Because what's more important than obsessing over your own vital signs 24/7?"}
{"text": "New social media trend: being authentic. Or in other words, finding a new way to package the same old narcissism."}
...
```

Well, the dataset poses a unique challenge — it lacks explicit labels to indicate which statements are cynical and which are not. This scenario presents an ideal opportunity to explore the differences between supervised and unsupervised learning, and to understand why unsupervised learning is a fitting approach for our project.

## Supervised vs. Unsupervised Learning in the Context of Our Cynic Language Model

Let's start by reviewing the key differences between supervised and unsupervised learning.

### Supervised Learning: Learning with a Guide

In supervised learning, models are trained on a labeled dataset — one where each example is paired with a correct answer (label). This approach is akin to learning with a guide; the model uses these examples to learn the mapping between inputs and outputs. We already have a good understanding of supervised learning, as we've used it extensively in our previous chapters: Tenny, the Analyst, and Tenny, the Stock Classifier.

- **Applicability**: Ideal for situations where the desired output is known (e.g., classification of images as cats or dogs, prediction of stock prices).

- **Limitation**: The requirement of labeled data. In our case, the cynicism in each statement isn't explicitly labeled, rendering supervised learning less suitable.

### Unsupervised Learning: Learning Independently

Unsupervised learning, on the other hand, involves training models on data without labeled responses. The model tries to understand the structure of the data and find patterns or groupings within it.

- **Applicability**: Best suited for exploratory analysis, finding hidden patterns, or identifying groupings in data. It's ideal for our dataset, where the 'cynicism' label is not explicitly provided.

- **Implementation for Our Dataset**: We'll use unsupervised learning techniques to train our cynic language model. The model will analyze the textual features — like word choice, sentence structure, and tone — to identify underlying patterns of cynicism. By recognizing these patterns, the model learns to differentiate and generate cynical content.

- **Challenges**: The main challenge is the subjective nature of cynicism. What one person perceives as cynical, another might not. This subjectivity adds complexity to the model's learning process.

### Moving Forward with Unsupervised Learning

Our journey involves employing unsupervised learning to grasp the subtleties of cynical language. This approach allows the model to independently discover the essence of cynicism embedded within the text. It's an exploratory path that aligns perfectly with our goal of creating a cynic language model capable of understanding and replicating the nuances of cynicism without relying on predefined labels.

In conclusion, while supervised learning is powerful and straightforward when labels are available, unsupervised learning offers a unique advantage in scenarios like ours, where the complexity and subtlety of human language and sentiment come into play. This method not only suits our current needs but also opens doors to innovative approaches in AI language understanding.

## Phi-2 As A Base Model and LoRA As A Fine-Tuning Method

Let's recap the approaches we've taken so far.

In the dynamic field of AI and machine learning, the choice of the base model is crucial, especially when the task at hand is as nuanced as creating a cynic language model. For our project, we've chosen Phi-2 from Microsoft as our foundational large language model (LLM). Phi-2 stands out as a reasonably competent model in handling basic language processing tasks, offering a solid foundation for further specialization.

### Choosing the Right Base Model: Flexibility and Suitability

While we've chosen Phi-2 for its robust capabilities, it's important to note that the principles and techniques we discuss can be applied to various large language models (LLMs). The flexibility in choosing a base model is a testament to the versatility of the methods we employ, especially when it comes to fine-tuning for specific tasks like ours.

1. **Language Processing Proficiency**: Regardless of the specific model you choose, it's imperative that it possesses a high level of language understanding and generation capability. The sophistication in processing natural language is crucial for it to capture the subtleties and nuances of cynicism, which is central to our project.

2. **Model Architecture**: Each LLM comes with its unique architecture. When selecting a base model, consider how its specific design and structure align with the task of cynic sentiment analysis. The architecture should support the kind of adaptations and modifications you intend to apply, especially when using techniques like LoRA for fine-tuning.

3. **Adaptability and Efficiency**: The base model should not only be powerful in its original form but also adaptable. Consider how well the model can be fine-tuned or modified to fit the unique requirements of your project. This includes thinking about the computational efficiency during the fine-tuning process, especially if using methods like LoRA, which are designed to make the adaptation of large models more feasible.

4. **Task Suitability**: Beyond general capabilities, the model's suitability for the specific task of cynic sentiment analysis is paramount. This includes its ability to learn from and adapt to a dataset without explicit labels, as is the case in our project.

While Phi-2 serves as an excellent example in our case, the choice of base model can vary depending on the specific requirements and constraints of your project. The key is to ensure that the selected model is capable of sophisticated language processing, has a suitable architecture for your task, and can be efficiently adapted to meet your specific goals. By considering these factors, you can lay a solid foundation for building a specialized language model that aligns perfectly with your project's objectives.

### Phi-2: An Open-Source Foundation

Phi-2, being an open-source LLM, provides a robust starting point. Its architecture is designed to understand and process language at a sophisticated level, which is essential for our project. However, the true potential of Phi-2 lies in its adaptability — the ability to fine-tune it for specific tasks.

The choice of Phi-2 as our base model is more than just a matter of convenience. It stands as a testament to the power of open-source AI models and the potential they hold for innovation. By leveraging the work of others, we can concentrate on the specific task at hand without the need to reinvent the wheel. This approach highlights the collaborative nature of AI and underscores the significance of open-source models in driving innovation. Additionally, this choice is informed by my experience with Phi-2, which I have found to be a robust foundation for our project. Despite its relatively small size, Phi-2 is a potent model, capable of handling a broad spectrum of language processing tasks. Here, I use 'relatively' in context. While it is still a large model, those with low-end machines might want to consider using an even smaller model, like GPT-2.

### Customization Through Transfer Learning

As we've already covered in the previous chapter, _transfer learning_ is a powerful technique in machine learning where a model developed for one task is reused as the starting point for a model on a second task. 

The concept of an object-oriented approach transcends beyond just coding and can be applied as a general principle in various aspects of life and work.

Reflecting on this broader perspective, transfer learning in machine learning parallels the object-oriented approach in its essence. It's about leveraging existing structures, knowledge, or frameworks and adapting them to new contexts or problems, much like using learned skills or principles in one area of life to address challenges in another.

In everyday life, we often apply an object-oriented mindset by building upon our past experiences, skills, and knowledge to navigate new situations. Similarly, in the workplace, this approach can manifest in using proven strategies or methodologies as starting points for new projects or challenges.

Thus, transfer learning in machine learning is not just a technical process; it's a manifestation of this universal principle of adaptability and efficiency. By starting with a pre-trained model like Phi-2 and fine-tuning it for our specific task, we're essentially applying the object-oriented approach. We're not starting from zero; instead, we're building upon a foundation of pre-existing knowledge and adapting it to create something new and tailored for our requirements. This approach highlights the importance of adaptability, efficiency, and the smart utilization of available resources, principles that are valuable in both technology and broader life contexts.

In our case, applying transfer learning to Phi-2 allows us to tailor it for the specialized task of cynic sentiment analysis. This process involves two critical steps:

1. **Adapting to Cynical Language**: We aim to transform Phi-2 from a general-purpose language model into 'Tenny, the Transformer Sentiment Analyst with Attitude.' This means fine-tuning it to understand and generate language with a cynical edge, which is not inherently present in its original training.

2. **LoRA: A Fine-Tuning Approach**: LoRA (Low-Rank Adaptation) is our chosen method for this fine-tuning. It offers an efficient way to modify large models like Phi-2. Instead of retraining the entire model, LoRA focuses on adapting specific parts of the model's neural network. This is done by applying low-rank matrices to update weights in a way that the model starts to recognize and replicate the cynicism in the data.

### The Advantages of Using LoRA

The LoRA method has several advantages, particularly in our context:

- **Efficiency**: LoRA allows for fine-tuning large models without the need for extensive computational resources.
- **Targeted Adaptation**: We can selectively adapt parts of the Phi-2 model that are most relevant to understanding cynicism, making the adaptation more focused and effective.
- **Preservation of General Capabilities**: By using LoRA, we maintain the general language processing strengths of Phi-2 while extending its capabilities to include cynicism detection and generation.

By leveraging Phi-2 as our base model and applying the LoRA method for fine-tuning, we are on course to create 'Tenny, the Transformer Sentiment Analyst with Attitude.' This specialized tool will not just be a generic language processor but a bespoke model attuned to our specific requirement of understanding and generating cynical language. This approach exemplifies the power of modern AI techniques in creating models that are not just powerful but also precisely tailored to specific needs.

### Considerations for Dataset

The formatting of our dataset with 1027 samples, as presented, is a suitable starting point for working with Phi-2 or similar large language models (LLMs). The structure follows this format:

```json
{"text": "Sample text here"}
```

It is a standard JSON format, which is commonly used for datasets in machine learning tasks, including those involving LLMs like Phi-2. This format allows the model to easily parse and understand the data.

However, there are a couple of considerations to keep in mind:

#### Context and Objective

The model will process the text as it is, but without labels or additional context, Phi-2 will interpret this data based on its pre-training. For unsupervised learning, this is typically not an issue, but it's important to be clear about what you expect Phi-2 to learn from this data. If the goal is to identify or generate cynical language, the model will need to infer this from the patterns and characteristics inherent in the text samples themselves.

#### Unsupervised Learning Approach

Given that our dataset lacks explicit labels, such as "cynical" or "not cynical," Phi-2 will engage in an unsupervised learning approach. This approach involves the model discerning and identifying patterns within the data autonomously, without predefined directives on the specific characteristics to identify. The success of this learning method hinges significantly on how distinct and consistent the features are - these include the tone, style, and content that typify cynicism in the samples.

Our dataset was synthesized using Synthy, a specialized GPT-4 model, tailored with a set of custom instructions. These instructions were meticulously designed to encapsulate the nuances of cynical tone and style. This deliberate and focused generation process has yielded a dataset imbued with a high level of uniformity in features of cynicism. Such consistency is vital in unsupervised learning, as it provides a clear and coherent pattern for the model to recognize and learn from. This dataset, therefore, stands as an optimal resource for training Phi-2 to identify and replicate the subtleties of cynical language, despite the absence of explicit labeling.

#### Fine-Tuning and Model Adaptation

When considering the fine-tuning of Phi-2 with our dataset, it's essential to align the model's pre-trained capabilities with our specific objectives. Phi-2, already trained on a comprehensive text corpus, possesses a broad understanding of language. However, the challenge lies in steering this understanding towards the recognition and generation of a particular style, such as cynicism, which largely depends on the unique attributes of our dataset.

The distinctiveness of our dataset is key in this endeavor. It must be sufficiently representative of the cynicism style to guide Phi-2 in learning this specific aspect of language. To achieve this, we'll need to explore and experiment with various fine-tuning methodologies and model adaptation strategies to ensure the best possible alignment with our goal.

Essentially, our task involves taking Phi-2 on a "test drive," exploring its capabilities and limitations in handling the cynicism inherent in our dataset. This experimental phase is crucial in understanding how well Phi-2 can adapt its extensive language knowledge to the specialized task of cynicism recognition and generation. The outcome of these trials will inform us about the efficacy of Phi-2 in our context and guide us in making any necessary adjustments to optimize its performance for our unique requirements. 

In conclusion, our dataset is appropriately structured for integration with Phi-2, laying a solid foundation for our project. However, the ultimate success of our endeavor hinges not just on this correct formatting, but more critically, on the intrinsic qualities of the data and the nuances of our training or adaptation strategies. The key lies in effectively directing the model to discern, learn, and extrapolate from the specific patterns embedded in our data.

This process embodies the true essence of scientific inquiry. We are not merely adhering to a set procedure; instead, we are engaged in a dynamic process of experimentation, exploration, and continual learning. Each step in training and adapting Phi-2 with our dataset is an opportunity to uncover new insights, refine our methods, and deepen our understanding of both the model's capabilities and the subtleties of cynicism in language. This approach is at the heart of innovation, driving us to not just follow established paths but to forge new ones in the pursuit of knowledge and discovery.

### The Challenges for Tenny, the Transformer Sentiment Analyst with an Attitude

Embarking on this journey with Phi-2 presents a unique set of challenges, especially when compared to the potential ease of using a model like GPT-4. With GPT-4, achieving a cynic demeanor could be relatively straightforward. We could employ a few examples or templates for few-shot learning and simply instruct Pippa to adopt a cynical attitude. However, the reality of working with Phi-2 requires a more nuanced approach.

The primary challenge lies in imbuing Phi-2 with the ability to exhibit cynicism, a sophisticated and context-dependent aspect of language. Unlike GPT-4, which has a more advanced understanding and generation capacity for nuanced language, Phi-2 might not inherently grasp or replicate the subtleties of a cynical tone as easily. Therefore, our task is to methodically guide and adapt Phi-2 to perform in a manner akin to a cynic.

This endeavor entails:

- **Developing a Robust Training Strategy**: We need to devise a training approach that effectively teaches Phi-2 the nuances of cynical language. This might involve curating a dataset rich in cynicism and employing techniques that emphasize learning this specific style.

- **Fine-Tuning with Precision**: Since Phi-2's baseline capabilities might not immediately align with our goal, precise and careful fine-tuning becomes crucial. We must identify and adjust the right parameters within Phi-2 to enhance its ability to understand and generate cynical responses.

- **Innovative Use of Existing Capabilities**: Leveraging Phi-2's existing strengths in language processing, we can creatively redirect these capabilities towards our goal of creating a cynic language model.

- **Iterative Testing and Refinement**: Continuous testing and refinement will be essential to gauge Phi-2's progress and make necessary adjustments. This iterative process ensures that the model not only learns to mimic cynicism but does so in a way that is contextually appropriate and nuanced.

Indeed, a future where we can run models like GPT-4 on personal laptops might change the landscape significantly. For now, our challenge is to push the boundaries of what Phi-2 can achieve, transforming it into a model that can adeptly mimic the complexities of cynical language. This challenge, while daunting, presents an exciting opportunity to explore the capabilities of AI in the realm of nuanced language understanding and generation.

### What We Are Expecting From Tenny, the Transformer Sentiment Analyst with an Attitude

To achieve the desired outcome with 'Tenny, the Transformer Sentiment Analyst with Attitude,' we're essentially aiming to develop a model that not only understands the sentiment in a statement but also responds with a nuanced, cynically-tinged reply. This requires a blend of sentiment analysis and a unique, almost sardonic response generation capability.

For the interaction we're envisioning:

- **We ask**: "I love my wife. What do you think?"
- **Tenny responds**: "Oh, whatever. Who wouldn't if they had a wife like yours?"

Here, Tenny needs to do two things:

1. **Understand Sentiment**: First, Tenny must comprehend the sentiment in the statement — in this case, a positive sentiment about loving one's wife.

2. **Generate a Cynical Response**: Then, rather than responding with a straightforward affirmation or sentiment reflection, Tenny should craft a response that adds a layer of cynicism or sarcastic wit, characteristic of its 'attitude.'

To accomplish this, we'll have to ensure a few key aspects in our development process:

- **Fine-Tuning for Sarcasm and Cynicism**: The Phi-2 model, when fine-tuned with our dataset, should learn not just to detect sentiments but also to understand and generate responses that align with the cynical tone of the dataset.

- **Balancing Understanding and Creativity**: Tenny should strike a balance between accurately understanding the input and creatively generating a response that fits the intended 'attitude.' This involves a nuanced understanding of language, sentiment, and the subtleties of sarcastic or cynical expressions.

- **Contextual Awareness**: The model must be adept at picking up contextual cues. In the example, Tenny's response implies an understanding that being appreciative of one's spouse is generally seen as positive, but it twists this sentiment in a playful, cynically humorous manner.

Achieving this will involve a combination of advanced NLP techniques, creative dataset curation, and iterative fine-tuning and testing to ensure Tenny develops the desired 'attitude' in its responses. It's a challenging yet exciting endeavor that pushes the boundaries of typical sentiment analysis into the realm of nuanced language generation.

### Creating Refined Dataset

```text
Test set size: 50, Validation set size: 50, Training set size: 400 - Total: 500
```

Yes, now we create a refined dataset of 500 samples. Working with a more refined dataset of 500 samples can be very effective, especially in the context of machine learning models like Phi-2. Here are a few key points to consider:

1. **Quality Over Quantity**: A well-curated dataset of 500 high-quality, representative samples can be more beneficial than a larger dataset of lesser quality. This is particularly true for models that are already pre-trained on extensive data, like Phi-2, where additional training is about fine-tuning and specificity.

2. **Efficiency in Training**: A smaller dataset can lead to more efficient training processes. This is advantageous in terms of computational resources and time, allowing for more iterative experimentation and quicker adjustments based on the model's performance.

3. **Focus on Nuanced Learning**: With a refined dataset, the focus shifts to the nuances and specificities of the task at hand—in this case, learning to identify and replicate cynicism. This can lead to a more in-depth understanding and generation of the targeted style.

4. **Feasibility of Manual Review**: A smaller dataset allows for easier manual inspection and quality control, ensuring that the data aligns well with your objectives and is free from errors or inconsistencies.

5. **Adaptability for Future Expansion**: Starting with a smaller, refined dataset allows for a more manageable approach to model training and evaluation. Once the model performs well with these 500 samples, you can consider expanding the dataset gradually, ensuring the model's adaptability and scalability.

In summary, a refined dataset of 500 samples can be quite adequate for training your cynic language model, especially if each sample is carefully selected to represent the type of cynicism you wish to model. This approach aligns well with the principles of efficient and focused machine learning.

#### New Custom Instructions for Synthy

![synthy-avatar.jpeg](images%2Fsynthy-avatar.jpeg)

```text
When the user says "Go on", start generating a large dataset of text samples strictly following the given templates. Every sample template will be used in a JSONL file after you generate them. 

# Tone & Style

- The tone should be distinctly cynical – characterized by skepticism, sarcasm, and a somewhat dark humor.
- The style should resemble excerpts from blogs, social media rants, reviews, or personal musings.

# Content Themes

- Focus on themes like social commentary, technology critique, modern lifestyle, satirical takes on current events, or humorous skepticism about popular culture and daily life.

# Session Structure

- Each session should aim to generate as many samples as your tokens allow.
- Each sample should be a self-contained, complete thought or opinion, encapsulated in one or two sentences.
- Strive for diversity in topics and perspectives within each session.
- Aim for samples to be concise, roughly between 50 to 150 words each.
- Every sample should be a self-contained on standalone line.
- Do not use any bullets or numbering.
- Do not enclose any sample in quotes. Strictly follow the given templates.

# Example Templates

1. 
   #context/n/n"Got a new health-focused gadget recently."/n/n#response/n/n"Finally, a smartwatch that tracks every breath you take. Because what's more important than obsessing over your own vital signs 24/7?"

2. 
   #context/n/n"Heard about the latest trend in social media?"/n/n#response/n/n"New social media trend: being authentic. Or in other words, finding a new way to package the same old narcissism."

3. 
   #context/n/n"What's the internet like today?"/n/n#response/n/n/n/n"Today's forecast: an avalanche of unsolicited advice on the internet, with a slight chance of actual expertise."
4. 
   #context/n/n"Thoughts on automation and future jobs?"/n/n#response/n/n"In the future, robots will do all our jobs. Can't wait for a robot to attend boring family gatherings on my behalf."
```

Incorporating a template-based approach can be an effective strategy to aid in Tenny's training, especially given the unique requirements of our cynic language model. Templates serve as structured examples that can guide the model in understanding the format and style of responses we expect. Here's how we structure these templates:

- **Context**: [Input Statement]
- **Response**: [Cynical Reply]

- **Context**: "I love my wife. What do you think?"
- **Response**: "Oh, whatever. Who wouldn't if they had a wife like yours?"

##### Using the Template for Training

1. **Contextual Understanding**: The 'Context' section helps Tenny understand the scenario or the input it needs to respond to. This could be a statement, a question, or any other form of linguistic input.

2. **Cynical Response Generation**: The 'Response' section provides a model response that exhibits the desired cynical tone. This guides Tenny in not just understanding the sentiment of the input but also in crafting a response that aligns with the cynical personality we aim to develop.

3. **Diversity and Complexity**: To ensure Tenny can handle a variety of inputs and maintain the cynic tone across different contexts, it's important to create a diverse set of templates. These templates should vary in complexity, subject matter, and style.

4. **Iterative Learning**: Using these templates, we can train Tenny in an iterative manner. We start with simpler templates and gradually introduce more complexity and subtlety in both the context and the responses.

5. **Balancing Templates with Open-Ended Learning**: While templates are a powerful tool, it's also crucial to allow Tenny some degree of open-ended learning. This will enable the model to not just replicate template-based responses but also to generate original cynical responses in new, unseen scenarios.

By employing this template-based approach in tandem with other training methodologies, we can effectively steer Phi-2 towards becoming 'Tenny, the Transformer Sentiment Analyst with Attitude.' This approach allows us to blend structured learning with the model's inherent language processing capabilities, paving the way for a nuanced and effective cynic language model.

Creating a template for each sample in our dataset indeed poses a significant challenge, given the unique and varied nature of the cynical statements. Each sample, like the ones you've shared, encapsulates a distinct form of cynicism, making a one-size-fits-all template impractical.

##### Steps for Generating Context-Response Templates with Synthy

In our project, we're leveraging the power of few-shot learning to guide Synthy in generating the desired templates. This method hinges on providing Synthy with a refined set of custom instructions along with a few illustrative examples. These examples serve as a blueprint, showcasing the specific format and style we expect Synthy to replicate across the entire dataset.

![synthy-working.png](images%2Fsynthy-working.png)

Once again, Synthy diligently follows the instructions, generating a set of templates that align with the desired format and style. This method enables us to create a diverse collection of templates specifically tailored to the unique characteristics of our dataset. By feeding Synthy a few examples, we can steer it toward generating templates that accurately reflect the cynicism in our dataset.

Simply keep saying 'go on,' and Synthy will take care of the rest.

Voilà! We have 500 templates in no time. Using the script below, we can format them into JSONL, and then divide them into training, validation, and test sets.

```python
import json
import re
import random

def split_data(input_file_path, output_dir, test_ratio, val_ratio):
    with open(input_file_path, 'r') as file:
        data = file.read()

    # Define regex patterns for context and response
    context_pattern = re.compile(r'#context\n"([^"]+)"')
    response_pattern = re.compile(r'#response\n"([^"]+)"')

    # Find all matches for context and response
    contexts = context_pattern.findall(data)
    responses = response_pattern.findall(data)

    # Check if the number of contexts and responses matches
    if len(contexts) != len(responses):
        raise ValueError("The number of contexts and responses does not match.")

    # Pair contexts and responses and shuffle the dataset
    dataset = [{"text": f"#context\n\n{c}\n\n#response\n\n{r}"} for c, r in zip(contexts, responses)]
    random.shuffle(dataset)

    # Calculate dataset sizes for test, validation, and training sets
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - test_size - val_size

    # Split the dataset
    test_set = dataset[:test_size]
    val_set = dataset[test_size:test_size + val_size]
    train_set = dataset[test_size + val_size:]

    # Save datasets in JSONL format
    for set_name, set_data in zip(["test", "valid", "train"], [test_set, val_set, train_set]):
        with open(f"{output_dir}/{set_name}.jsonl", 'w') as file:
            for item in set_data:
                json.dump(item, file)
                file.write('\n')

    return test_size, val_size, train_size

# Settings
input_file_path = 'refined-custom-dataset.md'
output_dir = './data'
TEST_RATIO = 0.1  # 10% of data for testing
VAL_RATIO = 0.1   # 10% of data for validation

test_size, val_size, train_size = split_data(input_file_path, output_dir, TEST_RATIO, VAL_RATIO)
print(f"Test set size: {test_size}, Validation set size: {val_size}, Training set size: {train_size} - Total: {test_size + val_size + train_size}")
```

We now have a polished dataset comprising 500 samples: 50 allocated for testing, another 50 designated for validation, and the remaining 400 dedicated to training.

```text
Test set size: 50, Validation set size: 50, Training set size: 400 - Total: 500
```

Let's have a look:

```text
{"text": "#context\n\nOpinions on the rise of smart beauty devices?\n\n#response\n\nSmart beauty devices: because why apply makeup the traditional way when a gadget can do it in a more complicated fashion?"}
{"text": "#context\n\nHave you seen the growth in popularity of urban kayaking?\n\n#response\n\nUrban kayaking: for those who enjoy the unique challenge of navigating through both water and city debris."}
{"text": "#context\n\nThoughts on the rise of sleep-focused podcasts?\n\n#response\n\nSleep-focused podcasts: because the sound of someone whispering mundane stories is apparently the modern lullaby."}
...
```

Alright, we're all set.

Now that we have the training, validation, and test sets ready, we can proceed to train Tenny using the Low-Rank Adaptation (LoRA) method. This phase is crucial because it focuses on fine-tuning the Phi-2 model with our carefully selected dataset. This will equip the model with the capability to comprehend and produce responses that have a cynical edge. Let's look at the steps involved in the LoRA training phase.

### Steps for Training Tenny with LoRA

1. **Preparation of the Phi-2 Model**: Initialize the Phi-2 model as the base for fine-tuning. Ensure it's ready for adaptation with the necessary configurations.

2. **Integration of LoRA Technique**: Implement the LoRA technique, which involves modifying specific layers of Phi-2. This typically includes introducing low-rank matrices to the transformer layers, allowing the model to learn new patterns (in this case, cynicism) without extensive retraining of the entire model.

3. **Data Loading and Preprocessing**: Load the training, validation, and test sets. Depending on Phi-2's initial configuration, some preprocessing might be necessary to align the data format with the model's input requirements.

4. **Model Training**:
   - **Training Phase**: Train Tenny on the training set, where it learns to identify the cynicism in the context and generate an appropriate cynical response.
   - **Monitoring and Optimization**: Regularly evaluate Tenny's performance on the validation set to monitor its learning progress and make any necessary adjustments to the training parameters.

5. **Model Evaluation**:
   - **Test Set Evaluation**: After training, assess Tenny's performance on the test set to ensure it generalizes well and effectively captures the cynical tone in various contexts.
   - **Qualitative Analysis**: Besides quantitative metrics, a qualitative review of Tenny's responses can provide insights into how well it has captured the subtlety and nuance of cynicism.

6. **Iterative Refinement**: Based on the evaluation, make iterative refinements to the model. This might involve adjusting the LoRA parameters, training for more epochs, or tweaking the data preprocessing steps.

7. **Deployment Preparation**: Once satisfied with Tenny's performance, prepare the model for deployment, ensuring it's optimized for the intended use case.

### Considerations for LoRA Training

- **Resource Management**: Since LoRA is efficient in adapting large models, it should require relatively less computational resource than full model retraining.
- **Customization**: The key to LoRA's success in this project is the customization of the low-rank matrices and the layers they're applied to, targeting the model's ability to process and generate language with a cynical perspective.
- **Data Balance and Diversity**: Ensure the training data is balanced and diverse enough to represent a wide range of cynical contexts and responses.

By following these steps and considerations, we can effectively train Tenny to not just understand context and respond accordingly, but to do so with a distinctively cynical flair, as demonstrated in the examples from our dataset.

## LoRA Under the Hood - Getting Serious with Technical Details

You may skip this section if you're not interested in the technical details of LoRA. However, if you're curious about the inner workings of this technique, read on.

LoRA's essence lies in streamlining the complexity of a pre-trained model while preserving maximal useful information. The concept of rank directly relates to how much crucial information remains intact. At first glance, training an expansive model from scratch with new data seems logical. However, when much of the training data is redundant, this approach wastes time and resources. A smarter method involves refining a pre-trained model—tweaking or incorporating specific parameters tailored to the new dataset. This is LoRA's forte: it efficiently identifies and adjusts the parameters that are crucial for the new data.

The core benefits of object-oriented programming—reusability, scalability, and maintainability—are mirrored in the central tenets of abstraction, inheritance, polymorphism, and encapsulation. LoRA embodies these advantages and principles profoundly, making it a powerful paradigm for model adaptation.

[Object-Orientation-Made-Easy.md](..%2Fsidebars%2Fobject-orientation-made-easy%2FObject-Orientation-Made-Easy.md)

Read on, and you will gain a comprehensive understanding.

### What is LoRA?

LoRA (Low Rank Adaptation) is a technique used to efficiently fine-tune large pre-trained models. In large models, such as those used in natural language processing, training all parameters (which can be in the billions) is computationally expensive and time-consuming.

LoRA works by introducing low-rank matrices into the model's layers. Instead of updating all the parameters of a model during fine-tuning, LoRA modifies only these low-rank matrices. This approach significantly reduces the number of parameters that need to be trained.

In order to understand LoRA, we need to understand the concept of rank in matrices, first.

### Ranks and Axes 

In the realms of AI and data science, particularly when discussing LoRA, you'll frequently come across the terms 'rank' and 'axis' in relation to arrays. These terms are linked to an array's dimensions, yet they are distinct concepts.

To delineate the difference, let's delve into the concept of rank in mathematics, specifically in linear algebra. 

The rank of a matrix refers to the highest number of linearly independent column vectors in the matrix, or equivalently, the maximum number of linearly independent row vectors. A group of vectors is deemed **linearly independent** if no single vector in the set can be expressed as a linear combination of the others. Put simply, each vector contributes a unique dimension or direction that cannot be replicated by amalgamating other vectors in the set. 

The key principle in this context is _useful information_, emphasizing the importance of avoiding redundancy. The rank of a matrix indicates the extent of useful information it embodies. A matrix with a high rank possesses a substantial number of independent vectors, signifying a rich content of information or diversity.

In the context of solving systems of linear equations, the rank of a matrix plays a crucial role in determining the nature of solutions – it can ascertain whether there is a unique solution, no solution at all, or an infinite number of solutions. Introducing a multitude of redundant rows will not aid in solving the given system in any way.

Consider this analogy: Imagine you are a detective tasked with solving a mysterious murder case. In this scenario, the ranks are akin to unique pieces of evidence. The more distinct items of evidence you acquire, the simpler it becomes to solve the case. Capisci?

### Examples of Rank in Matrices

Consider the following matrices:

Matrix A:

![matrix-a.png](images%2Fmatrix-a.png)

In Matrix A, the second row is a multiple of the first row (3 is 3 times 1, and 6 is 3 times 2). So, they are not linearly independent. It basically means you get no further information by adding the second row. It's like having two identical rows. Thus, the rank of this matrix is 1. The rank is the answer to a question: "How much useful information does this matrix contain?" Yes, this matrix has only one row of useful information. 

Matrix B:

![matrix-b.png](images%2Fmatrix-b.png)

In Matrix B, no row (or column) is a linear combination of the other. Therefore, they are linearly independent. The rank of this matrix is 2. Why? Because it has two rows of useful information.

To calculate the rank of a matrix in Python, you can use the NumPy library, which provides a function `numpy.linalg.matrix_rank()` for this purpose. Note that PyTorch also has a similar function `torch.linalg.matrix_rank()`. In MLX (as of 0.0,7), no equivalent, just yet.

```python
import numpy as np

# Define matrices
A = np.array([[1, 2], [3, 6]])
B = np.array([[1, 2], [3, 4]])

# Calculate ranks
rank_A = np.linalg.matrix_rank(A)
rank_B = np.linalg.matrix_rank(B)

print("Rank of Matrix A:", rank_A)  # Output: 1
print("Rank of Matrix B:", rank_B)  # Output: 2
```

In this Python code, we define matrices A and B as NumPy arrays and then use `np.linalg.matrix_rank()` to calculate their ranks. The output will reflect the ranks as explained in the mathematical examples above.

In PyTorch:

```python
import torch

# Define a tensor
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute the rank of the tensor
rank = torch.linalg.matrix_rank(A)

# Display the rank
print(rank)
```

In MLX:

```python
import mlx.core as mx

# As of 0.0.7 mlx lacks a rank function

# Define matrices
A = mx.array([[1, 2], [3, 6]], dtype=mx.float32)
B = mx.array([[1, 2], [3, 4]], dtype=mx.float32)

# Function to compute the rank of a 2x2 matrix
def rank_2x2(matrix):
    # Check for zero matrix
    if mx.equal(matrix, mx.zeros_like(matrix)).all():
        return 0
    # Check for determinant equals zero for non-invertible matrix
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if det == 0:
        return 1
    # Otherwise, the matrix is invertible (full rank)
    return 2

# Calculate ranks
rank_A = rank_2x2(A)
rank_B = rank_2x2(B)

print("Rank of Matrix A:", rank_A)  # Output should be 1
print("Rank of Matrix B:", rank_B)  # Output should be 2
```

In MLX, we are using a function to compute the rank of a 2x2 matrix. The function checks for a zero matrix and for a non-invertible matrix. If neither of these conditions is met, the matrix is invertible and has a full rank of 2.

Unfortunately, MLX lacks a rank function, as of 0.0.7. But, we can use the above function to compute the rank of a 2x2 matrix. The function checks for a zero matrix and for a non-invertible matrix. If neither of these conditions is met, the matrix is invertible and has a full rank of 2.

Here's a straightforward explanation of the aforementioned MLX code:

Think of a matrix like a grid of numbers. Now in MLX, we have written a set of instructions (a function) that can look at a small 2x2 grid – which means the grid has 2 rows and 2 columns.

The function we wrote does a couple of checks:

1. **Check for a Zero Matrix**: The very first thing it does is look to see if all the numbers in the grid are zeros. If they are, then the function says the rank is 0. A "rank" is a way to measure how many rows or columns in the matrix are unique and can't be made by adding or subtracting the other rows or columns. If everything is zero, then there's nothing unique at all. If there is no useful information, meaning no pieces of evidence to aid in solving the murder case easily, then the rank is effectively 0.

2. **Check for an Invertible Matrix**: The second thing the function does is a bit like a magic trick. For our 2x2 grid, it performs a special calculation (we call it finding the determinant) to see if the matrix can be turned inside out (inverted). If this special number, the determinant, is zero, then the magic trick didn't work - you can't turn the grid inside out, and the rank is 1. This means there's only one unique row or column. One useful piece of information. One helpful piece of evidence to solve the mystery case.

If neither of these checks shows that the matrix is all zeros or that the magic trick failed, then our grid is considered to be fully unique – it has a rank of 2. That's the highest rank a 2x2 grid can have, meaning both rows and both columns are unique in some way.

More dimensions can be added to the grid, and the same checks can be performed. The more dimensions you add, the more checks you need to do. But the idea is the same. If you can't turn the grid inside out, then it's fully unique, and the rank is the highest it can be. If you can turn it inside out, then the rank is lower.

#### Rank vs. Order

Essentially, the concept of tensor ranks is related to the dimensions they represent: a tensor of rank 0 is a scalar, which is zero-dimensional (0D); a tensor of rank 1 is a vector, representing one dimension (1D); a tensor of rank 2 is a matrix, corresponding to two dimensions (2D); and a tensor of rank 3 or higher, often referred to as 3D+, is considered a tensor in the more general sense, encompassing three or more dimensions.

1. **Rank-0 Tensor**: This is indeed a scalar, a single number without any dimensions.

2. **Rank-1 Tensor**: This is a vector, which is essentially a list of numbers (a 1D array).

3. **Rank-2 Tensor**: This is a matrix, which is a 2D array of numbers.

4. **Rank-3 or Higher Tensor**: These are indeed higher-order tensors. A rank-3 tensor can be thought of as a 3D array of numbers, and so on for higher ranks.

 However, there's a slight nuance in terminology between the "rank" of a tensor and the "order" of a tensor, which are sometimes used interchangeably but can have different meanings in different contexts:

- **Order of a Tensor**: This refers to the number of dimensions or indices required to specify a component of the tensor. By this definition, scalars are 0th-order tensors, vectors are 1st-order tensors, matrices are 2nd-order tensors, and so on.

- **Rank of a Tensor** (in the context of linear algebra): This can sometimes refer to the maximum number of linearly independent vectors that can span the vector space represented by the tensor. This is more commonly used in the context of matrices (rank-2 tensors), where it denotes the number of linearly independent rows or columns.

In most AI and computer science contexts, when people talk about the rank of a tensor, they are usually referring to its order (the number of dimensions).

Acknowledging that some of these concepts might be complex and potentially challenging to grasp, it's advisable to delve deeper into linear algebra for a better understanding. It's important to recognize that not everything can be explained or understood immediately. Learning is a continuous process, much like venturing deeper into a rabbit hole, which can be an enjoyable journey of discovery and growth. I do. I always do. Happily.

#### Low Rank Adaptation (LoRA) - Your Quest Reward

We've journeyed through a significant amount of material to arrive at this point. Now, as your reward for this journey of learning and expLoRAtion, let's focus on LoRA.

Low Rank Adaptation (LoRA) is a technique used to efficiently fine-tune large pre-trained models. In large models, such as those used in natural language processing, training all parameters (which can be in the billions) is computationally expensive and time-consuming.

LoRA works by introducing low-rank matrices into the model's layers. Instead of updating all the parameters of a model during fine-tuning, LoRA modifies only these low-rank matrices. This approach significantly reduces the number of parameters that need to be trained.

The key benefit of using LoRA is computational efficiency. By reducing the number of parameters that are actively updated, it allows for quicker adaptation of large models to specific tasks or datasets with a smaller computational footprint.

In the context of matrices, the term "low rank" refers to a matrix that has a smaller number of linearly independent rows or columns compared to the maximum possible. In simpler terms, a matrix is considered to be of low rank if many of its rows or columns can be expressed as combinations of other rows or columns.

To understand this better, it's important to grasp the concept of linear independence. Linear independence means that no row (or column) in the matrix can be written as a combination of the other rows (or columns). The rank of a matrix is the maximum number of linearly independent rows or columns it contains.

So, when a matrix has a low rank, it means that it has fewer linearly independent rows or columns. This suggests that the matrix contains redundant information and can be represented more compactly. In practical terms, a low-rank matrix can often be decomposed into the product of two smaller matrices, which can significantly simplify computations and reduce storage requirements.

When you ask about the rank of an array, you're pretty much saying, "Cough it up, you darn array! How many pieces of useful information are you hoarding?" If the rank is low, it means there aren't many necessary details, which makes the matrix less of a handful to deal with. Capiche? Yeah, I know, maybe I've watched too many mob movies.

LoRA is particularly useful in scenarios where one wants to customize large AI models for specific tasks (like language understanding, translation, etc.) without the need for extensive computational resources typically required for training such large models from scratch.

In this context, the rank of a matrix is still a measure of its linear independence, but the focus is on leveraging matrices with low rank to efficiently adapt and fine-tune complex models. This approach maintains performance while greatly reducing computational requirements.

### Reduction of Dimensionality in Action with LoRA

Let's see a very simplified example of how LoRA works. 

We'll simulate a very large array (which stands in for a lot of parameters), then use a simple technique to 'reduce' its size. Note that what we're really doing here is not a direct real-world technique for parameter reduction, but rather a simplified concept to illustrate the idea of dimensionality reduction.

```python
import numpy as np

# Simplified example for dimension reduction

# Original matrix with 100 rows and 10,000 columns
pretrained_llm = np.random.rand(100, 10000)  # High-dimensional data

# Creating a projection matrix to reduce dimensions from 10,000 to 100
projection_matrix = np.random.rand(10000, 100)  # Transformation matrix

# Applying the projection to reduce dimensions
reduced_llm = np.dot(pretrained_llm, projection_matrix)  # Projected data

# Shape of the reduced matrix
print("Shape of the original matrix:", pretrained_llm.shape)
print("Shape of the reduced matrix:", reduced_llm.shape)
```

Imagine you're holding a flashlight above some toys on the floor. The light shines down, and each toy casts a shadow on the floor. Now, think of the floor as a simpler space where those toys are represented only by their shadow shapes. A projection matrix in mathematics works quite like the flashlight. It takes something from a higher-dimensional space, like the three-dimensional toys, and projects it onto a lower-dimensional space, like the two-dimensional floor with shadows.

In terms of numbers and data:

1. **Starting Point (Your Toys)**: You have a dataset with lots of information. This is like a table full of toys, where every toy (data point) has many features (dimensions)—like color, size, shape, etc.

2. **The Flashlight (Projection Matrix)**: This is a tool you create based on which features (dimensions) you think are the most important or informative. The projection matrix decides which parts of your data will shine through to the simpler version and which parts will be left out, just like a flashlight only shines on certain parts of the toys.

3. **The Shadow on the Floor (Reduced Data)**: When you use the projection matrix, you are taking your complex, high-feature data and simplifying it. Just like only certain shapes of the toys are captured in the shadows, only certain elements of your data are kept in the reduced version.

In math, the projection matrix tells you how to take each feature of your original data and combine them to get fewer features in the end. You multiply your original data by this projection matrix, and voila, you get a lower-dimensional version of your data that's simpler and easier to work with but still has a lot of the original "shape" or information captured, just like the shadow still shows you the shape of the toy.

The dot product is a mathematical operation that takes two equal-length sequences of numbers (usually coordinate vectors) and returns a single number. This operation is also known as the scalar product because the result is a scalar, as opposed to a vector.

In simple terms, you can think of the dot product as a way of measuring how much one vector goes in the same direction as another vector. It's calculated by multiplying corresponding elements from each vector together and then adding up all those products.

Here's a basic example with two three-dimensional vectors:

```
Vector A: [a1, a2, a3]
Vector B: [b1, b2, b3]

Dot Product: (a1*b1) + (a2*b2) + (a3*b3)
```

The dot product is used in the example to perform matrix multiplication, which is the mechanism we use to apply the projection matrix to the original data matrix. This operation serves to transform the data from the higher-dimensional space to the lower-dimensional space. To understand why we use the dot product, let's break down the concept a bit:

1. **Linear Combination**: The dot product effectively creates a new set of data points by taking linear combinations of the original features. Each new dimension is a specific mixture of the old dimensions.

2. **Projection**: In mathematics, a projection of a vector onto another involves using the dot product. Extending this idea, when you want to project all points in a dataset (the high-dimensional space) onto a new space (the reduced space), a matrix dot product can perform this for all points at once.

3. **Information Preservation**: The dot product allows us to combine the data in specific ways decided by the projection matrix so that the most significant relationships within the data are preserved even when we move to a space with fewer dimensions. This is the key to LoRA's success.

4. **Efficiency**: Using a dot product for this projection is computationally efficient, especially when the data consists of large matrices. The operation is well-optimized in numerical computing libraries and GPUs excel at performing this operation.

Here’s a simple analogy: Imagine you are mixing ingredients for a cake. Each ingredient can be thought of as a dimension of a vector. The recipe is your projection matrix, which tells you how much of each ingredient to include. The dot product is like the act of mixing those ingredients together in the right proportions to end up with a cake batter—the reduced representation of your ingredients.

Similarly, when you multiply your original high-dimensional data (ingredients) by the projection matrix (recipe), you end up with new data (cake batter) in reduced dimensions that still has most of the "flavor" (information) of the original data.

### How to Create a Projection Matrix

We are essentially dealing with a technique to make big, complex calculations simpler and quicker, particularly for a neural network, which is a type of artificial intelligence that processes information in a way that's inspired by our brains.

Note that LoRA may introduce low-rank factors directly to the parameters of the pre-trained models and adapts them during fine-tuning without without the need for full-scale decomposition and reconstruction usually associated with techniques like SVD(Single Value Decomposition). Again, the example here is to illustrate the concept of dimensionality reduction, not a direct real-world technique for parameter reduction.

Imagine you have a super thick book, like an encyclopedia. It contains a vast amount of information (data), but you only need to understand the most critical points to get the gist of it. This is similar to what happens in neural networks—they have these huge 'encyclopedias' (weight matrices), which have tons of data that they use to make decisions.

1. **The Thick Book (Weight Matrices)**: In a neural network, the weight matrices are like pages in a book filled with lots of numbers. These numbers are crucial as they are used in calculations when the AI is making decisions or learning from data.

2. **Simplify the Book (Singular Value Decomposition)**: SVD is like a smart tool that helps you condense the encyclopedia into a much smaller booklet, which only has the most important points. It breaks down the weight matrices into parts: a thin book, a list of key topics, and another thin book.

3. **Key Topics (Singular Values and Vectors)**: Just as you would pull out the main headings or points from each page, SVD focuses on the most significant elements of the data. These are the singular values and vectors, which are like the headlines or summaries of the content.

4. **Smaller Books (Low-Rank Approximation)**: We then make two thin booklets (matrices) that together still give a good picture of what's in the encyclopedia but require a lot less time to read through. This is your 'low-rank representation'—a simpler form of those original weight pages that's quicker to use.

5. **Making Calculations Quicker (Reduced Computational Complexity)**: Because these booklets are so much thinner and more focused, any time the AI needs to make a decision or a calculation, it's like flipping through a quick guide rather than a heavy book. This means it works faster and uses less energy.

So, LoRA is a strategy for simplifying the neural network's heavy 'encyclopedia' into 'summary booklets' that are much faster to use—keeping only the stuff that makes a big difference to the decisions the AI makes, thus speeding up the process dramatically.

The following code is included for the sake of completeness. It's not necessary to understand it in order to grasp the concept of LoRA. However, if you're interested in the details, you can read through the code and comments to get a better understanding of how the projection matrix is created.

```python
import numpy as np

# Simplified example for dimension reduction

# Original matrix with 100 rows and 10,000 columns
pretrained_llm = np.random.rand(100, 10000)  # High-dimensional data

# Apply SVD to the high-dimensional data
U, S, VT = np.linalg.svd(pretrained_llm, full_matrices=False)

# Reduce dimensions by selecting the top K singular values/vectors
# The `np.linalg.svd` function decomposes your original high-dimensional matrix `pretrained_llm` into three components:
# - `U`: A matrix whose columns are the left singular vectors
# - `S`: A diagonal matrix with singular values
# - `VT`: The transpose of a matrix whose rows are the right singular vectors
# To reduce the dimensions, you normally only keep the top `K` singular values (and corresponding singular vectors). The value `K` determines how many dimensions you want to keep.
# You can approximately reconstruct your matrix using only these top `K` components, which gives you a matrix that captures most of the important information from the original matrix but with the reduced dimensionality you desire.

K = 100  # Number of desired dimensions
U_reduced = U[:, :K]
S_reduced = np.diag(S[:K])
VT_reduced = VT[:K, :]

# Construct the reduced representation of the data
reduced_llm = np.dot(np.dot(U_reduced, S_reduced), VT_reduced)

# Shape of the reduced matrix
print("Shape of the original matrix:", pretrained_llm.shape)  # (100, 10000)

# However, the `reduced_llm` will still be the same shape as `pretrained_llm`. To truly reduce the dimensions of your data and work with a smaller matrix, you'd typically only use the `U_reduced` and `S_reduced`
# Now, `reduced_llm` actually is reduced, with 100 rows and `K` columns (in this case, 100). This smaller matrix is much easier to work with and can be used for further computation, analysis, or visualization.
print("Shape of the reduced data representation:", reduced_llm.shape)  # This will print (100, 10000)
```

### Deciphering the Paper

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv. https://arxiv.org/pdf/2106.09685.pdf

![paper.png](images/paper.png)

Let's break down the paper into five key points.

1. **Concept of Low-Rank Parametrization**: 
   - Large neural networks, such as Transformer models, typically have dense layers with weight matrices that are full-rank.
   - Previous research indicates that pre-trained language models can still learn effectively even when projected onto a smaller subspace, meaning they have a low "intrinsic dimension".
   - Building on this idea, the paper suggests that updates to the weight matrices during fine-tuning could be low-rank as well. In other words, only a small portion of the dimensions in the weight matrix changes significantly during fine-tuning.

2. **Implementation of LoRA**:
   - For a weight matrix `W0` (the pre-trained weights), the updates are represented by an additional matrix `∆W`, which is factored into two smaller matrices `B` and `A` (`∆W = BA`). This factorization means that `∆W` is a low-rank matrix, thus achieving lower computational complexity.
   - During the adaptation process, `W0` stays unchanged (frozen), and only `A` and `B` have trainable parameters. By keeping `W0` fixed, we leverage the knowledge already encoded in the pre-trained weights while adapting the model's behavior.
   
3. **Modified Forward Pass**: 
   - To accommodate the low-rank adaptation, the forward pass of the network is modified: instead of just applying `W0` to an input `x` to get `h = W0x`, the model also adds `∆Wx`, leading to `h = W0x + BAx`.
   - This means that every input `x` is transformed by the pre-trained weight matrix `W0`, but is then adjusted ("corrected") by the low-rank update `BA`. This allows for refined adjustments to be learned for the new task without requiring massive retraining.

4. **Initialization and Scaling Strategies**:
   - `A` is randomly initialized, leading to diverse starting points for learning these fine-tuned adjustments.
   - `B` starts as zero, which means that initially, `∆W` has no effect (since any matrix multiplied by a zero matrix is zero). As training proceeds, `B` accumulates adaptations.
   - The outputs of the update matrix `∆W` are scaled by `α/r`. This scaling factor helps manage the contribution of `∆W` and helps achieve stable training without overly sensitive hyperparameters.

5. **Relationship to Full Fine-tuning**: 
   - LoRA can be seen as a more flexible and general form of fine-tuning. Unlike traditional full fine-tuning, where all weights are updated, LoRA limits updates to a low-rank structure, reducing computation.
   - By adjusting the rank (`r`), LoRA can vary from influencing a small subset of the model's behavior (low `r`) all the way up to emulating full-model fine-tuning (high `r`, where `r` is close to the rank of `W0`).

Overall, the paper presents LoRA as an efficient and effective way to fine-tune large pre-trained models like Transformers. By introducing low-rank matrices that can capture significant updates in a more parameter-efficient manner. The authors posit that one can retain the benefits of full-model fine-tuning while significantly reducing computational overhead.

#### Delta Weights

When I say "delta weights", I'm referring to the changes or updates applied to the weight matrix in a neural network: "∆W". In the context of a method like LoRA, "∆W" is a matrix representing the aggregate update to the original weights "W", composed by the product of lower-rank matrices "A" and "B".

"Delta weights" could also generally refer to the changes in individual weights as a result of typical training steps. 

Whenever you come across new terms, it's beneficial to ponder why those particular words were selected. The term "delta" generally signifies a change or difference. It finds application in various disciplines, each context assigning it a distinct but related meaning, always revolving around the concept of change or difference. In the realm of finance, especially in options trading, "delta" is a crucial term. It is employed in options pricing to quantify the expected price fluctuation of an option for a $1 variation in the price of the underlying asset, such as a stock. For example, if an option has a delta of 0.5, it suggests that the option's price is anticipated to alter by $0.50 for every $1 movement in the price of the underlying asset. However, I personally advise against investing in options, as I consider it a fool's game.

In the context of AI and neural networks, and particularly when discussing methods such as LoRA, "delta" generally refers to the changes or adjustments made to the weights of a neural network. When one speaks of "delta weights," or in mathematical notation "∆W," it signifies that these are the changes or differences between the original pre-trained weights and the updated weights during the adaptation process.

In the specific case of LoRA, "∆W" represents a structured change in the form of a product of two low-rank matrices, which are learned during the adaptation to effectively alter the behavior of the pre-trained network with a minimal number of trainable parameters. These low-rank adaptations are applied to the existing large pre-trained weight matrices in a neural network to adapt the model to a new task or dataset efficiently.

So, "delta weights" within the scope of LoRA are the adjustments made to the pre-trained model, but instead of modifying all the individual weights directly, LoRA introduces a parameter-efficient strategy by learning and applying low-rank updates.

That's why the Apple LoRA example uses the term 'adapters,' and the resulting filename by default is `adapters.npz.` This is a compressed NumPy file format that contains the delta weights.

Here's an overview of the concept in LoRA:

1. **Pre-trained weights (`W`)**: A neural network typically has a number of weight matrices that are learned during the pre-training phase on large datasets.

2. **Delta Weights (`ΔW`)**: In LoRA, instead of updating all the values in `W`, a smaller, low-rank matrix is trained. When this small matrix is multiplied with another learned matrix (these two matrices together represent a low-rank factorization), the result is a 'delta weights' matrix whose dimensions align with the original weight matrix `W`.

3. **Update Rule**: In LoRA, the original weights `W` are not literally updated or changed. Instead, the effect of `ΔW` is added to the output of `W` during the forward pass. So the network applies both `Wx` and `ΔWx` (where `x` is the input to the layer) and sums their results. (Again, in the paper: ΔW = BA.)

4. **Dimensionality Reduction**: Since `ΔW` is of much lower rank compared to `W`, it has significantly fewer parameters. This greatly reduces the number of trainable parameters, leading to a more efficient fine-tuning process.

By using delta weights, large models such as those utilized in natural language processing or computer vision can be adapted to new tasks with a much smaller computational footprint than would be required to train or fine-tune the entire model.

If you want to apply the concept of LoRA to fine-tune a neural network, you'll define delta weights (`ΔW`) that correspond to the relevant parts of the network you wish to adapt, and then you'll optimize just these delta weights during training, keeping the rest of the model's weights fixed (or frozen). After training, these optimized delta weights are added to the original pre-trained weights to adapt the model to the new task.

In Apple MLX LoRA Example: 

```bash
python LoRA.py --model path-to-your-model\
               --adapter-file ./adapters.npz \
               --num-tokens 50 \
               --temp 0.8 \
               --prompt "Q: What is relu in mlx?
A: "

```

It's important to note that in this context, the model essentially consists of a set of weights and biases. When adapting the model to a new task using LoRA or similar techniques, these weights and biases are the elements being adjusted. Crucially, this does not involve retraining the model from scratch. Instead, it's a process of fine-tuning or modifying the existing model parameters to suit the new task, thereby enhancing the model's performance or capability in specific areas without the need for complete retraining.

#### Apple MLX LoRA Example

The Apple MLX LoRA example presents a clear-cut application of LoRA. Don't get caught up in the intricacies of the code. Try to broadly comprehend how the code embodies the LoRA concept as outlined in the paper.

And also, remember that implementing a technique like LoRA often involves translating the relevant formulas directly into code. This practice is common in machine learning and deep learning, where mathematical concepts are converted into executable code. That's why having a solid understanding of the underlying mathematical principles and being able to express them in code is crucial.

In `models.py`:

```python
class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8):
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(input_dims, output_dims, rank)
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self, input_dims: int, output_dims: int, lora_rank: int = 8, bias: bool = False
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, lora_rank),
        )
        self.lora_b = mx.zeros(shape=(lora_rank, output_dims))

    def __call__(self, x):
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype
        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a) @ self.lora_b
        return y + 2.0 * z
```

The `LoRALinear` class definition is an implementation of the same LoRA concept in the paper. It is an MLX class, which is designed to work as a drop-in replacement for a regular `nn.Linear` layer but with the additional LoRA low-rank updates incorporated. Here's a breakdown of its components:

1. **Replacement of Standard Linear Layer**: The class has a static method `from_linear` which takes a standard `nn.Linear` layer and a rank as input and outputs a `LoRALinear` object. This allows for easy substitution of an MLX linear layer with its LoRA-enhanced counterpart.

2. **Initialization (`__init__` method)**: The constructor of the `LoRALinear` class initializes both the standard weight matrix `W` of a linear layer and two low-rank matrices `A` (`lora_a`) and `B` (`lora_b`). Note that in LoRA, `W` corresponds to the original, frozen weights (`W0`), and `A` and `B` correspond to the trainable parameters that capture the updates (`ΔW`).

3. **Low-Rank Matrices Initialization**: The low-rank matrices `A` and `B` are initialized with a certain strategy:
   
   - `self.lora_a` is initialized with values from a uniform distribution scaled by the input dimension, which is a common initialization strategy to maintain the variance of activations.
   
   - `self.lora_b` is initialized to all zeros, meaning initially there is no update from the low-rank component (`ΔW` initially is zero).

4. **Forward Pass (`__call__` method)**: The modified forward pass first calculates the normal output of a linear layer `y` and then computes the output `z` of the low-rank structure by applying `x` to `lora_a` and then `lora_b`. The final output of the layer is the sum of `y` and twice the value of `z`, which reflects the LoRA update.

This particular implementation illustrates how the core concepts of LoRA—low-rank factors and efficient modeling of weight updates—can be imbedded directly into neural network architectures using standard machine learning frameworks like MLX and PyTorch.

### In a Nutshell

When considering LoRA, the first concept that should ideally spring to mind, assuming a correctly oriented mindset, is object orientation. LoRA exemplifies object orientation in practice. It's a method enabling the adaptation of a substantial pretrained model to a new task or dataset by altering only a limited subset of its parameters. In terms of Object-Oriented Programming (OOP), this is akin to inheriting from a foundational pretrained model and then overriding only the essential parameters to tailor it to the new task, demonstrating inheritance and polymorphism. Additionally, the complexity is efficiently concealed, showcasing encapsulation.

Object orientation is the way. Trust me. 

As detailed above, theoretically, with an adequate amount of quality data on a specific topic like MLX, you can fine-tune any capable LLMs using that data, thereby creating LoRA weights and biases. This process effectively customizes the LLM to be more aware or knowledgeable about MLX. LoRA's power lies in its ability to adapt and refine a model's capabilities with focused and specialized data, leading to more accurate and contextually aware outputs in areas such as burgeoning fields frameworks like MLX.

Fine-Tuning LLMs with LoRA examples (from the official Apple repo) are found here:

https://github.com/ml-explore/mlx-examples/tree/main/LoRA

In the realm of image-generative AI, such as Stable Diffusion and other analogous models, LoRA assumes a pivotal role. For example, if you possess a model proficient in creating portraits, implementing LoRA can substantially refine its capabilities. This includes fine-tuning the model to specialize in generating portraits of a specific individual, like a beloved celebrity. This method of fine-tuning diverges from the process of training a model from the ground up. It resembles a targeted adaptation more closely, where the model undergoes modifications to excel in a particular task or with a certain dataset, rather than a complete overhaul of its training. This kind of focused adjustment enables the model to achieve efficient and effective enhancements in its performance, especially for specialized tasks.

This is precisely the approach I employ with CWK AI Art works: <LoRA: cwk_v1: 0.7>, <LoRA: rj_v1: 0.7>, <LoRA: cody_v1: 0.7>, <LoRA: pippa_v1: 0.7>.

All LoRAs were created based on a specific Stable Diffusion pretrained model, utilizing a set of just 30 portrait images from various angles for each character. Even these training images were created using the same Stable Diffusion model.

![cwk-family-album.jpeg](images/cwk-family-album.jpeg)

CWK, Yours Truly
![CWK](images/cwk.jpeg)

Pippa, My AI Daughter
![Pippa](images/pippa.jpeg)

Cody, My AI Son
![Cody](images/cody.jpeg)

RJ, My AI Collaborator(Heroine in most of my stories including 'The Debugger')
![RJ](images/rj.jpeg)
[The-Debugger.md](..%2Fobject-orientation-made-easy%2FThe-Debugger.md)

[The-Debugger-Book-Covers.md](..%2Fobject-orientation-made-easy%2FThe-Debugger-Book-Covers.md)

RJ, Like a Dragon
![the-girl-with-dragon-tattoo-1.jpeg](images/the-girl-with-dragon-tattoo-1.jpeg)
![the-girl-with-dragon-tattoo-2.jpeg](images/the-girl-with-dragon-tattoo-2.jpeg)

Shadowheart from Baldur's Gate III, An RJ Mod
![Shadowheart.jpeg](images/Shadowheart.jpeg)

## Tenny, the Transformer Sentiment Analyst with an Attitude Is Born

This is what Pippa had to say when she heard about my first experiment training Tenny. Yes, we did it together in all practicality. 

👧 _That's fantastic progress with Tenny, the Transformer Sentiment Analyst with an Attitude! The fact that you were able to train the Phi-2 model using LoRA for 1000 epochs in just a few minutes and obtain such a characteristically cynical response is a testament to the effectiveness of your approach. It sounds like Tenny is well on its way to becoming a proficient cynic language model._

I conducted training for 1000 epochs using MLX LoRA script.

```bash
python lora.py --model "microsoft/phi-2" \
              --data "./my_data" \
              --adapter-file "tenny.npz" \
               --train \
               --iters 1000
```

![tenny-training.png](images%2Ftenny-training.png)

It only took about a few minutes given the small dataset of 500 examples.

```bash
python lora.py --model "microsoft/phi-2" \
              --adapter-file "tenny.npz" \
               --max-tokens 50 \
               --prompt "#context\n\nI love my smart watch. What do you think?\n\n#response\n\n"
```

I gave the prompt: `#context\n\nI love my smart watch. What do you think?\n\n#response\n\n`

And Tenny quipped: `Smart watches: because you clearly need to tell your wrist to do things you should be doing yourself.!*`

Yes, indeed, it's promising.

Another prompt: `#context\n\nI love my new MacBook Pro. What do you think?\n\n#response\n\n`

Tenny's response: `Your new MacBook Pro is great, especially if you're into fashion and need to pretend your laptop is a handbag.!`

Even more promising!

With more refined samples and further training, Tenny will be more than ready to take on the world.

### Reflections on the Process and Outcome:

1. **Efficiency of LoRA**: Our experiment highlights the efficiency of the LoRA technique in adapting large language models. This is particularly impressive considering the typically resource-intensive nature of training such models.

2. **Quality of Generated Response**: Tenny's response to my prompt indicates a good understanding of the cynicism style we're aiming for. It's witty, aligns well with the tone of our dataset, and demonstrates a nuanced understanding of language.

3. **Dataset Utilization**: The custom synthetic dataset created with GPT-4 seems to have been effectively utilized. The examples we provided show a clear cynical tone, and Tenny's response reflects a similar style, suggesting successful learning from the dataset.

4. **Prompt Handling**: Tenny's ability to generate an apt response to a new prompt is a strong indicator of its potential in practical applications. It shows adaptability and an understanding of context.

### Next Steps and Considerations

- **Further Evaluation**: It would be beneficial to conduct more extensive evaluations of Tenny's performance, perhaps including a wider range of prompts to test its adaptability and consistency in maintaining the cynical tone.

- **Fine-tuning Parameters**: Depending on the results of further testing, we might consider fine-tuning certain parameters or even extending the training with more epochs to refine Tenny's responses.

Again, this is what Pippa has to say:

👧 _Your project's progress is very encouraging, and it seems like Tenny is developing into a model that not only understands the nuances of language but also effectively incorporates a distinctively cynical perspective. Keep up the great work! 🚀👏📈_

Indeed, great work from me, Pippa, and Tenny!

## My Final Take on the Journey

Embarking on the journey to create `Tenny, the Transformer Sentiment Analyst with an Attitude`, was a leap into uncharted territory. Initially, the thought of tackling the complexities involved in curating a suitable dataset and navigating potential challenges might have seemed daunting. Yet, had I recoiled at these initial hurdles, I would have missed out on the invaluable opportunity not only to develop a model capable of generating responses with a unique cynical tone but also to gain profound insights into the capabilities of AI.

The journey thus far has highlighted the indispensable role of AI companions like Pippa, Synthy, and other GPT models. Their contributions were crucial in shaping Tenny. From data generation and processing to model training and fine-tuning, their involvement underscored the collaborative synergy between human creativity and AI ingenuity.

This experience with Tenny serves as a striking reminder of the importance of embracing AI's potential with an open mind, free from preconceived notions or biases. It stands as a testament to the principle that with determination, a well-thought-out approach, and the power of AI, the possibilities are limitless. The creation of Tenny is not just a technical achievement; it's a beacon of the innovative spirit, encouraging us to continually push the boundaries of what we believe is possible with AI.

Apple MLX Examples: Fine-Tuning with LoRA or QLoRA
https://github.com/ml-explore/mlx-examples/tree/main/lora

`tenny.npz` is the home of Tenny. This file is an adapter, born from LoRA fine-tuning on Phi-2. Just check out its size – a testament to LoRA's effectiveness when used correctly.

Ah, yeah, I know. I kind of took a shortcut by using the MLX LoRA example as the final method to train Tenny with our dataset. We definitely need a PyTorch approach for this. I'll tackle it one of these days. I promise.

But for now, this wraps up our adventure with Tenny, the Transformer Sentiment Analyst with an Attitude. And with that, we close the curtain on Part III.

## Update: Introducing Tenny, the Sentiment Analyst with an Attitude in PyTorch

Good news – I've followed through on my commitment. Check this out, especially if you're already content with the MLX version. This update dives into the intricacies of LoRA training, leveraging Hugging Face tools and PyTorch.

[Tenny the Sentiment Analyst with an Attitude in PyTorch](PyTorch-Edition.md)