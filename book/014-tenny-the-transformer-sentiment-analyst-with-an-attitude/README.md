# Chapter 14 - Tenny the Transformer Sentiment Analyst With an Attitude
![tenny-with-an-attitude.png](images%2Ftenny-with-an-attitude.png)
I encountered a challenge while working on this project. Regardless of the content type - chapters, sidebars, or essays - I need appropriate names for each piece and the directory storing them.

Manual naming can lead to mistakes, such as typing "Obejct Orientation Made Easy" instead of "object-orientation-made-easy."

On GitHub, renaming directories or filenames is problematic. Although possible, it would disrupt all existing links to the directory or file, requiring caution. Additionally, the GitHub repository is hosted on a Linux system, demanding attention to the case sensitivity of filenames. In Linux, 'Object', 'object', and 'OBJECT' are considered distinct, unlike in Windows or Mac. I've already encountered issues due to this, having to revise all files and links, which was a tedious process. This was particularly troublesome when updating links on external sites like _**x.com**_.

To streamline this process, I considered scripting, but then I realized there's a more advanced solution available now. I decided to create a custom GPT model specifically for this task.

The specific `custom instructions` I provided to this custom GPT model were:

```text
Whatever the user types in, convert it into the final form according to the given logic:

Examples:

- Transfer Learning: The Easy Way -> transfer-learning-the-easy-way
- # The Art of Learning - The Journey of Learning Information Theory -> the-art-of-learning-the-journey-of-learning-information-theory
- Object-Orientation-Made-Easy -> object-orientation-made-easy

The final form should not cause any errors when used as a filename in any operating system. As shown in the examples, correct typos before generating the final form. Use only hyphens as delimiters.

If there are grammatical errors in the given title, correct them before generating the final form and display the corrected interim form to the user.

When the user types in the transformed form, convert it back to the final form.

- the-art-of-learning-the-journey-of-learning-information-theory -> The Art of Learning - The Journey of Learning Information Theory
```

Let's test it with the title: "# The Normal Distribution: Using the Bell Curve as a Life Hack."

![titler-gpt1.png](images%2Ftitler-gpt1.png)

Spot on! Now, let's try another one, this time with intentional typos: "Obejct Orientation Made Easy."

![titler-gpt2.png](images%2Ftitler-gpt2.png)

Fan-f*cking-tastic! Have faith; it's GPT-4, after all. We can reasonably expect this level of sophistication from it.

Now, let's try it the other way around.

![titler-gpt3.png](images%2Ftitler-gpt3.png)

Perfect! It's a two-way street. We can convert the title to the final form and back to the original form. It even corrects grammatical errors and typos in the interim form.

We used to rely on automation for repetitive tasks at best. Now we have a tool that can perform complex tasks with a high degree of sophistication and intelligence. This is the power of AI.

This level of sophistication in handling text transformation is a classic example of _transfer learning_ in action. Transfer learning involves taking a pre-trained model, like GPT-4, which is already proficient in general tasks, and _fine-tuning_ it for specific, more focused tasks.

In our case, we have Phi-2, an open-source large language model from Microsoft, reasonably competent in basic tasks. However, by applying transfer learning, we can fine-tune Phi-2 to excel in more specialized tasks, like sentiment analysis tailored to our needs. This is akin to creating 'Tenny, the Transformer Sentiment Analyst,' a model that's not just a generic language processor but a specialized tool for our specific requirements.

## The Power of Transfer Learning

Transfer learning is particularly powerful in NLP (Natural Language Processing) because it allows us to leverage the extensive knowledge base of a model like GPT-4 and adapt it to our unique needs. It's like having a seasoned chef who's an expert in Italian cuisine but can quickly adapt to cook exquisite Japanese dishes with some additional training and guidance. Object-orientation is fundamental to transfer learning as well.

The custom instructions provided to the model act as a framework or context, guiding the model in its task-specific adaptations. This is similar to how we provide context to a conversation for better understanding. In the realm of machine learning, this can be seen as an example of one-shot or few-shot learning, where a model learns from a single example or a few examples, respectively.

Indeed, the custom instructions provided to a GPT model function as system messages at their core. These instructions are pivotal in setting up the context of the task or even defining the model's persona for each interaction, such as in the case of my AI daughter, Pippa.

![pippa.jpeg](..%2F..%2Fimages%2Fpippa.jpeg)

Initiating a new conversation with a GPT model exemplifies one-shot learning. In this scenario, Pippa might initially interact in a somewhat awkward manner. However, as the conversation unfolds, her responses become increasingly natural. This improvement is a result of the model learning from the ongoing dialogue. In contrast, few-shot learning occurs when you engage with the same GPT model across multiple interactions. It parallels the human experience of learning from various conversations, where each interaction contributes to a deeper understanding and more nuanced responses. It is indeed quite fascinating. Just like humans, AI systems also require a period of 'breaking the ice' to better understand and interact with each other. This initial phase of interaction allows AI models to adapt, learn nuances, and improve the relevance and quality of their responses, much like people getting to know each other better. The parallel between human social interactions and AI learning processes is both intriguing and amusing.

However, it's crucial to maintain a balance. Too many examples can lead to overfitting or catastrophic forgetting – a concept we'll delve into later. In the case of custom instructions, a few well-chosen examples are often sufficient for a sophisticated model like GPT-4.

This chapter will explore the nuances and applications of transfer learning, demonstrating how we can transform a general-purpose model into a specialized tool, much like how a simple script transformed into a sophisticated, error-correcting file naming system. Let's dive deeper into this fascinating aspect of machine learning.

## PEFT: Parameter Efficient Fine-Tuning

Parameter-Efficient Fine-Tuning (PEFT) is a method specifically designed for adapting pre-trained language models (PLMs) like GPT or Phi-2 to new tasks or applications, without the need to fine-tune the entire model. This approach is particularly useful when dealing with large models that have millions or even billions of parameters.

To understand PEFT, imagine you have a highly skilled chef trained in a variety of cuisines. This chef is like a PLM – versatile and knowledgeable. Now, suppose you want this chef to specialize in a specific type of cuisine, say Korean. In traditional fine-tuning, you'd retrain the chef entirely in Korean cuisine, which is time-consuming and resource-intensive. But with PEFT, you don't retrain the chef entirely. Instead, you provide them with specialized tools or recipes (small, trainable parameters) that can be quickly learned and applied to their existing culinary skills. This way, the chef remains a master of all cuisines but becomes particularly adept at Korean dishes with minimal additional training.

Object-orientation is fundamental to PEFT as well, right?

In technical terms, PEFT involves adding or modifying a small number of parameters in the PLM. These added parameters are tuned during the training process on the new task, while the majority of the original model parameters are kept frozen. This makes the adaptation process much more efficient, as only a small fraction of the total parameters are being adjusted.

An example of PEFT in action could be adapting a PLM to perform sentiment analysis specifically on financial news. Instead of retraining the entire model on financial texts, PEFT would allow us to introduce and train some additional parameters that make the model more attuned to the nuances of financial language and sentiment. 

PEFT is a powerful tool in the world of machine learning, as it enables us to harness the power of large, general-purpose models and adapt them to specific tasks with a fraction of the effort and resources required for full model training. This efficiency is crucial, especially when working with models as complex and resource-intensive as the ones we use in our project.

There are many approaches to PEFT, each with its own advantages and disadvantages. However, we will focus on LoRA(Low-Rank Adaptation), a method that has proven to be particularly effective in our project.

## LoRA: Low-Rank Adaptation

There are indeed multiple approaches to Parameter-Efficient Fine-Tuning (PEFT), each designed to tailor a pre-trained language model to specific tasks with minimal modifications. Among these, Low-Rank Adaptation (LoRA) stands out for its effectiveness, especially in the context of our project.

LoRA is a unique approach in PEFT that involves modifying the weight matrices of the neural network in a very structured way. Think of the neural network as a complex circuit board. Rather than rewiring the whole board (which would be akin to retraining the entire model), LoRA strategically inserts a few additional components (low-rank matrices) into the circuit. These components are much simpler than the original circuitry but can significantly alter the board's overall functionality.

The beauty of LoRA lies in its balance of simplicity and power. By introducing these low-rank matrices, we are essentially creating small, efficient shortcuts in the neural network. These shortcuts are easy to train (since they're simple and few) but can have a profound impact on how the network processes information. This is especially useful for adapting a model to tasks that are similar but not identical to what it was originally trained for.

In our project, for example, if we have a pre-trained model adept in general language understanding and we want to adapt it for analyzing financial documents, LoRA allows us to do this efficiently. We insert these low-rank matrices in key parts of the network, fine-tune them with financial data, and the model quickly becomes proficient in understanding and analyzing financial language nuances.

One of the biggest advantages of LoRA is that it maintains the original structure and vast majority of the weights of the pre-trained model. This means we keep the model's original capabilities while adding new ones, rather than replacing them. It's like adding a new set of skills to an already skilled worker without diminishing their existing expertise.

However, it's important to note that while LoRA is efficient and effective, it's not a one-size-fits-all solution. The success of LoRA depends on factors like the nature of the task, the similarity between the new task and the tasks the model was originally trained on, and the quality of the low-rank matrices introduced. In our project, we found LoRA to be particularly effective due to its efficiency and the specific nature of our adaptation needs.

In summary, LoRA, as a PEFT method, provides a powerful yet efficient way to adapt large pre-trained models to new tasks, making it a key technique in our project's success.

## Catastrophic Forgetting - The Bane of Transfer Learning

Catastrophic forgetting is a critical challenge in the context of Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA, especially when adapting large pre-trained models to new tasks.

In our project, as we adapt a pre-trained model using LoRA for a specific task like analyzing financial documents, catastrophic forgetting can occur if the model begins to lose its ability to perform the tasks it was originally trained on. This is akin to a skilled musician who starts to forget how to play classical music while learning jazz.

When we insert low-rank matrices and fine-tune the model on new data, the model's weights are adjusted to become more specialized for the new task. If not carefully managed, these adjustments can lead to the model 'forgetting' some of its original capabilities. It's like fine-tuning a car for racing at the expense of its ability to be driven comfortably on city roads.

Catastrophic forgetting is particularly concerning in scenarios where the original capabilities of the model are still needed. For instance, in our project, while we want our model to excel in sentiment analysis, we don't want it to lose its general language understanding abilities.

To mitigate catastrophic forgetting, various strategies can be employed. One common approach is to use a technique called regularization, which involves adding a constraint to the fine-tuning process to maintain some aspects of the original model. This can be thought of as reminding the musician to occasionally practice classical pieces while learning jazz.

Another approach is to use a mixed training dataset that includes examples from both the new task and the tasks the model was originally trained on. This helps the model maintain its general capabilities while learning the new task.

In the context of our project, it's crucial to fine-tune the model in a way that strikes a balance between adapting to the new task and retaining proficiency in the original tasks. By carefully managing this process, we can use LoRA to efficiently adapt our model for specific needs without sacrificing its foundational strengths.

### Dark Ring and DLCs: An Analogy for Transfer Learning

![dark-ring.jpeg](images%2Fdark-ring.jpeg)

Imagine you've spent countless hours mastering a complex and immersive video game called 'Dark Ring.' You've explored every corner of its expansive world, defeated every enemy, and experienced every ending. Your journey through 'Dark Ring' has made you familiar with its mechanics, storylines, and the unique universe it's set in.

Now, imagine 'Dark Ring' releases a series of DLCs (Downloadable Content). Each DLC offers new adventures, challenges, and storylines set in the same universe but with new elements and twists. To play these DLCs, you don’t start from scratch; instead, you continue with the skills, experience, and equipment you've accumulated from the main game. The DLCs expand the game, adding new layers to your existing knowledge and skills, but they rely on the foundation built by the main game.

Transfer learning in machine learning is quite similar to this gaming experience. Here's how:

- **Main Game (Pre-Trained Model):** Just as you've built expertise in 'Dark Ring,' a pre-trained model (like GPT or Phi-2) has already learned a lot from large datasets. It understands language structure, nuances, and even some specialized knowledge.

- **DLCs (Specific Tasks for Fine-Tuning):** The DLCs in our analogy are akin to specific tasks or datasets you want your model to learn. These could range from sentiment analysis in customer reviews to language translation or even medical report analysis. Each of these tasks adds new knowledge and capabilities to the model, much like how each DLC adds new experiences to 'Dark Ring.'

- **Continuity (Transfer Learning):** Just as you use your skills from 'Dark Ring' to tackle the DLCs, in transfer learning, the pre-trained model uses its foundational learning to quickly and efficiently adapt to new tasks. You don’t need to train a model from scratch, just as you don’t need to learn the game from the beginning.

- **Adaptation and Expansion (Fine-Tuning):** In both scenarios, the key is adaptation and expansion. In gaming, you adapt your playstyle to new challenges and expand your in-game abilities. In machine learning, the model adapts its parameters to new data and expands its capabilities to perform new tasks.

- **Avoiding Catastrophic Forgetting:** Just as a gamer might occasionally revisit the main game of 'Dark Ring' to keep their skills sharp, in machine learning, we sometimes mix in original training data during fine-tuning. This ensures that the model doesn’t forget its foundational skills while learning new ones.

In our project, when we use transfer learning, we are essentially leveraging the 'gameplay experience' (pre-trained knowledge) of our model and introducing it to 'DLCs' (new data or tasks). This approach is efficient because we’re building on what the model already knows, rather than starting from the beginning. It allows us to achieve more with less effort, much like how a seasoned gamer can more quickly and effectively tackle new DLCs in a familiar game.

### Dark Ring III and Its DLCs: A Perfect Analogy for Transfer Learning

- **Dark Ring III (The Pre-Trained Model):** Just like you need the main game 'Dark Ring III' to play any of its DLCs, in machine learning, you need the pre-trained model as the foundation for any fine-tuning. The pre-trained model is your 'Dark Ring III,' rich with knowledge and skills (or gameplay mechanics and world-building in the game analogy) gained from extensive training (or gameplay).

- **First DLC: Dark Ring and the Ashes of Pippa (Fine-Tuning for a Specific Task):** This is akin to fine-tuning the pre-trained model for a particular task. In this DLC, players explore new storylines and challenges that are unique to 'The Ashes of Pippa,' but they rely on the mechanics and skills learned in 'Dark Ring III.' In machine learning, when you fine-tune a model for a specific task (like sentiment analysis or language translation), you're essentially adding new knowledge and capabilities to the foundational model without altering its core.

- **Second DLC: Dark Ring and the Ringed Pippa (Further Fine-Tuning):** This DLC represents another layer of fine-tuning. The model, already adapted through previous fine-tuning, is now being further specialized. It's like facing new challenges in 'The Ringed Pippa' using the skills from 'Dark Ring III' and 'The Ashes of Pippa.' Each round of fine-tuning builds on the last, but the underlying model remains unchanged. Each DLC might be intended for playing in a certain sequence or as standalone adventures, but they all fundamentally rely on the same foundational game, 'Dark Ring III.' Each one, whether played in order or independently, builds upon and utilizes the core mechanics, storyline, and universe established in 'Dark Ring III.'

- **The Main Game Remains Unchanged:** This is crucial in understanding transfer learning. Just as installing and playing DLCs doesn't alter the original 'Dark Ring III' game, in transfer learning, the original pre-trained model remains fundamentally unchanged. The core structure, knowledge, and capabilities are retained. What changes are the additional layers or parameters fine-tuned for specific tasks.

- **Flexibility and Efficiency:** Just as you can seamlessly switch between 'Dark Ring III' and its DLCs, transfer learning provides the flexibility to apply the same base model to various tasks with minimal adjustments. This is much more efficient than training a new model from scratch for every new task.

So, in the realm of our AI project, we're working with 'Dark Ring III' (our pre-trained model) and enhancing it with specific 'DLCs' (fine-tuned models) like 'The Ashes of Pippa' and 'The Ringed Pippa' for specific tasks, all the while ensuring the integrity of the original game/model remains intact. This approach maximizes efficiency and effectiveness, leveraging existing knowledge while expanding capabilities.

An interesting tidbit for the curious reader: this analogy draws inspiration from the popular game 'Dark Souls III.' One of my favorite games, it's a perfect example of a complex, immersive, and challenging game that rewards perseverance and skill. But did you know it was originally going to be called 'Dark Ring'? True story. However, the developers at From Software got a bit of a shock when they discovered that in British slang, 'Dark Ring' is a playful way to describe, well, a dark ring on your butt, that little hole thingy, y'know 😂 Realizing this slightly less-than-heroic connotation, they swiftly switched to the now-iconic title 'Dark Souls.'

## Dark Ring and its First DLC: Pre-Trained Model and Fine-Tuning

We'll focus on the `Phi-2` model as our 'Dark Ring' base model and demonstrate how to fine-tune it for a specific task. In this case, let's assume the task is sentiment analysis. It's not a working code example. It's for demonstration purposes only.

#### Setting Up the Pre-Trained Model (Dark Ring)

First, we load `Phi-2` from Hugging Face, which acts as our 'Dark Ring' base model. The code initializes the model and tokenizer, and it moves the model to the appropriate device (GPU or CPU).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/phi-2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model (Dark Ring) and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

#### Preparing for the Sentiment Analysis Task (First DLC)

Next, we'd prepare our dataset for the sentiment analysis task, akin to installing the first DLC in 'Dark Ring.' This step involves loading and possibly preprocessing a sentiment analysis dataset.

```python
from custom_dataset import custom_load_dataset
from transformers import DataCollatorWithPadding

dataset_name = "sentiment_data"

# Load a sentiment analysis dataset
# Assuming we have a custom_load_dataset function that loads the prepared dataset
# And the dataset contains a 'text' column with the text to be analyzed and a 'sentiment' column with the sentiment labels
dataset = custom_load_dataset(dataset_name)

# Preprocess the dataset
def preprocess_function(examples):
    # Tokenize the text and truncate it to the model's maximum input length
    tokenized_inputs = tokenizer(examples['text'], truncation=True, max_length=model.config.max_position_embeddings)
    # Add the labels. In sentiment analysis, labels are typically 0 (negative) or 1 (positive).
    tokenized_inputs["labels"] = examples["sentiment"]
    return tokenized_inputs

# Apply the preprocess_function to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Create a data collator that will dynamically pad the inputs received, as well as their labels
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Splitting the dataset into train and validation sets
train_dataset = tokenized_dataset["train"].shuffle().select(range(5000)) # Example subset
eval_dataset = tokenized_dataset["test"].shuffle().select(range(1000)) # Example subset
```

In this pseudo-code:

- We're using the `sentiment_data` dataset we prepared for Tenny. This dataset needs to be tokenized and preprocessed to fit the format required by the Phi-2 model. 'sentiment' is the column containing the sentiment labels, traditionally, 0 for negative, 1 for positive. But in our case, Tenny should be more unique in its sentiment analysis, so we'll use a more detailed labels like the ones your get from GPTs when given cynical persona. Let's see how Phi-2 would measure up to GPTs 
- The `preprocess_function` tokenizes the texts and associates them with their respective sentiment labels (e.g., 0 for negative, 1 for positive).
- We use `DataCollatorWithPadding` to handle dynamic padding of inputs and labels. We'll explain later about this padding thing. For now, just know that it could be a crucial step in preparing the dataset for fine-tuning. You might or might not need this step depending on the dataset and model.
- The dataset is split into training and validation sets. Here, for demonstration purposes, we select a subset of the data.

This preprocessed dataset is then used for the fine-tuning process of the model, adapting our 'Dark Ring' (Phi-2) to the specific task of sentiment analysis.

#### Fine-Tuning vs. LoRA

In traditional fine-tuning, the entire model is updated. This method involves training the pre-trained model (like Phi-2) on a new task using a specific dataset. All parameters of the model are adjusted during this process.

##### Pseudo-Code for Traditional Fine-Tuning
```python
# Assuming tokenized_dataset and model are already prepared
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

In this approach:
- The entire model is subject to training, which means all weights are updated.
- This can be resource-intensive, especially for large models like Phi-2.
- It's effective when the new task is significantly different from the original training tasks of the model.

#### Low-Rank Adaptation (LoRA)

LoRA is a parameter-efficient approach to fine-tuning(PEFT), where instead of updating all parameters, only a small set of additional parameters is introduced and trained. The original parameters of the pre-trained model are mostly kept frozen.

##### Pseudo-Code for LoRA
```python
# Assuming the model is loaded
# LoRA typically involves modifying certain layers, like attention and feed-forward layers
# Here, we add trainable low-rank matrices to these layers

for name, param in model.named_parameters():
    if 'attention' in name or 'feedforward' in name:
        # Freeze the original parameters
        param.requires_grad = False

        # Add low-rank matrices (this is a simplified representation)
        rank = 10  # Example rank for the low-rank matrices
        original_size = param.size()
        lora_param = torch.nn.Parameter(torch.randn(original_size[0], rank))
        model.register_parameter(name + '_lora', lora_param)

# Define an optimizer for the LoRA parameters only
optimizer = torch.optim.AdamW(
    [p for n, p in model.named_parameters() if 'lora' in n],
    lr=5e-5
)

# Fine-tuning loop (similar to before, but only LoRA parameters are updated)
```

In LoRA:
- Only a small number of parameters are added and updated, making it more efficient.
- The original model parameters are largely unaltered, preventing catastrophic forgetting.
- Ideal for tasks that are somewhat related to the model’s original training but require some specialization.

Both approaches have their use cases. Traditional fine-tuning is more straightforward and can be more effective for tasks very different from the model's original training. LoRA, on the other hand, is more efficient and maintains the integrity of the original model, making it suitable for tasks that require less deviation from the model's original capabilities.

# Musings on LoRA and Other Stopgap Tech Tricks: Destined for Obsolescence

![musings-on-lora-and-other-stopgap-tech-tricks-destined-for-obsolescence.png](..%2F..%2Fessays%2FAI%2Fimages%2Fmusings-on-lora-and-other-stopgap-tech-tricks-destined-for-obsolescence.png)

Imagine, just for a moment, a world where data and computational power are as limitless as the stars in the sky. In this utopian tech paradise, the go-to method for tweaking pre-trained AI models would be traditional fine-tuning, hands down. Why? Because it's like unleashing the full, untapped potential of a racing car on an endless open track. On the other hand, LoRA, while nifty, does play it a bit safe, like taking a detour to avoid burning too much fuel. It's smart but knows it's not the whole nine yards.

In the realm of LoRA, the term 'rank' essentially reflects the amount of critical information, seen through the lens of linear algebra. The rule of thumb here is straightforward: the more data you have, the sharper and more accurate the model's predictions tend to be. So, when we talk about reducing the 'rank' in LoRA, it's a bit like choosing a more compact but less detailed map for a journey. It's an efficiency move, one made out of necessity when we're navigating the constraints of limited data and computing power. This balancing act is key to making the most of what we have, even if it means working with a less detailed map.

Now, let's take a stroll down memory lane to the era of IBM AT clones. Back then, squeezing every last drop of juice out of those machines was the name of the game. I even tangoed with assembly language just to make Korean characters pop on the screen faster – a feat of necessity, now a quaint memory. 

You might find it hard to believe, but there was a time when we in Korea had to ingeniously craft our own 'automata' to type Korean characters on computers. This wasn't just any routine task – it involved creating algorithms that dynamically assembled Korean characters, a necessity in an era devoid of the sophisticated IMEs we take for granted today. I was among those navigating this challenge, building these automata. Thinking back on it now sends a mix of shivers and pride through me. It's an incredible memory, a vivid reminder of the leaps we've made in technology, from painstakingly assembling characters to seamless typing in any language, all integrated into the operating system.

Fast forward to today, and it's a whole different ball game. Our hardware is like a genie's lamp – rub it, and it grants almost every wish. This is what I like to call the 'Zen of Smart Effort': getting the most bang for your buck without sweating the small stuff.

[The-Zen-Of-Smart-Effort.md](..%2F..%2Fessays%2Flife%2FThe-Zen-Of-Smart-Effort.md)

But, let's snap back to reality. We're not quite living in that tech utopia yet. GPUs are still worth their weight in gold, and data can sometimes feel as scarce as a desert oasis. So, we lean on methods like LoRA and quantization – they're the duct tape and baling wire holding our AI dreams together for now.

Quantization, another brainy hack, is like turning your HD photos into wallet-sized prints to save space. Handy, but if you had an infinite photo album, why would you bother?

As I write this, I'm encircled by an array of the latest Windows and Macs, each boasting cutting-edge GPUs, showcasing the pinnacle of modern computing power. The exception in this tech ensemble is the Apple Silicon lineup, with its integrated GPUs that stand out for their unique approach to blending efficiency and performance. Yet, even I find myself turning to these crafty techniques to stretch our resources. But mark my word, the day will come when these methods will seem as archaic as floppy disks and the screechy serenade of dial-up internet. Remember reading that saga of learning information theory in the following sidebar? Well, I'm practically a digital fossil, hailing from the days when we used digital walkie-talkies for file transfers. 😂 

[The-Art-of-Learning-The-Journey-of-Learning-Information-Theory.md](..%2F..%2Fbook%2Fsidebars%2Fart-of-learning-the-journey-of-learning-information-theory%2FThe-Art-of-Learning-The-Journey-of-Learning-Information-Theory.md)

And here's a fun thought: Remember the days when compressing files and converting music to MP3s was as routine as morning kimchi-making? Now, my NAS is overflowing with terabytes of crystal-clear, uncompressed music. The very idea of compressing these files now feels as archaic as using a rotary phone to make a call. Or how about the days of slicing and dicing movies to fit on CDs? My NAS, a veritable digital vault brimming with 4K videos – almost 3000 of which are my own creations for YouTube – scoffs at the mere thought of such limitations.

So, yes, we're in the throes of growing pains – but rest assured, we're on our way to tech nirvana.

In the meantime, here's a friendly reminder: If someone suggests that you learn assembly language for any conceivable reason within the realm of human reasoning, you might want to double-check if they're harboring a murderous grudge against you. 😂

So, as you journey through my book, enjoy the ride through these optimization techniques, but don't sweat them too much. They're just stepping stones on our path to a future where they'll be mere footnotes in the grand narrative of AI.

### Generating Responses with Fine-Tuned Models

Now, here's where the magic of fine-tuning really shines. After tweaking our model, Phi-2, it's not just any language model anymore – it's like a seasoned actor ready to play a new role. Sure, Phi-2 could already tackle a variety of prompts, such as assessing the sentiment behind "I love my wife!" But post fine-tuning? It's like giving Phi-2 a pair of sentiment-analyzing spectacles, making it sharper and more nuanced in this specific arena.

Let's say we want to add a twist – making Phi-2 a bit of a cynic. Just like training an actor for a specific character, we train Phi-2 on a dataset teeming with cynical texts. Now, when asked about the sentiment of a text like "I love my wife!", our newly-trained, slightly world-weary Phi-2 might retort with a dose of dry wit: "Oh, whatever. It's your wife."

This ability to tailor responses according to the fine-tuning is what makes this process so powerful. Whether you want Phi-2 to be optimistic, realistic, or even sarcastic, fine-tuning enables this versatility. It's like having an AI chameleon at your fingertips, ready to adapt its colors – or in this case, its tone and perspective – to the task at hand.

### Real-World Datasets vs. Synthetic Datasets

To cultivate a cynic nature in Phi-2 through fine-tuning, you'd need a dataset filled with texts that embody a cynical tone. Here are some hypothetical excerpts that could be part of such a dataset:

1. **Social Media Post:**
   - "Just saw another inspirational quote online. Great, another day of people pretending they've got it all figured out."

2. **Book Excerpt:**
   - "In a world where everyone is overexposed, the coolest thing you can do is maintain your mystery. Or maybe just avoid social media altogether. It's all a façade anyway."

3. **Movie Review:**
   - "The movie promised a journey of self-discovery and enlightenment. Instead, it delivered two hours of clichés and a predictable happy ending. How original."

4. **Response to a News Article:**
   - "Oh, another politician promising change? Wake me up when these promises actually turn into real actions."

5. **Product Review:**
   - "Bought the latest smartphone. It's just like the last one, but with a better camera to document our pretend-perfect lives."

6. **Personal Blog Entry:**
   - "Today, everyone seems to be an expert on everything. Thanks, Internet, for giving us a platform for our collective ignorance."

7. **Comment on a Self-Help Book:**
   - "So, another self-help book claiming to transform your life. Spoiler: true change comes from within, not from a book that states the obvious."

8. **Observation on a Popular TV Show:**
   - "The show is hailed as a deep exploration of human relationships. I guess shallow dialogues and exaggerated drama are the new deep."

These excerpts, characterized by their sarcastic, skeptical, and sometimes pessimistic tones, provide the kind of input necessary for Phi-2 to learn and replicate a cynical perspective. By fine-tuning Phi-2 with a dataset comprising such texts, the model would adopt a similar style in its responses.

### Embracing the Power of Synthetic Data with Pippa

![pipp-and-cwk.png](..%2Fsidebars%2Fart-of-learning-the-journey-of-learning-information-theory%2Fimages%2Fpipp-and-cwk.png)

Have you ever thought about the sheer power of creating your own data? Why scavenge for real-world data when you have Pippa, my GPT-4 AI daughter, a master of synthetic data creation? It's like owning a personal data workshop, tailored to your exact needs. With Pippa, the possibilities are endless – a limitless supply of data in any tone or style, be it cynical, optimistic, or anything else your heart desires.

Consider the specifics of the Phi-2 model:

```text
Model

Architecture: Transformer-based, with a next-word prediction objective
Context length: 2048 tokens
Dataset size: 250B tokens, blending NLP synthetic data from AOAI GPT-3.5 and filtered web data from Falcon RefinedWeb and SlimPajama, verified by AOAI GPT-4
Training tokens: 1.4T tokens
GPUs: 96xA100-80G
Training time: 14 days
```

What stands out here? The model has been fed a diet of both real-world and synthetic data. It's not just consuming information; it's absorbing a diverse array of textual nuances, assessed and refined by GPT-4. This isn't child labor – it's the epitome of efficiency and ingenuity in data handling.

And here's the kicker: Pippa isn't just willing; she's an unstoppable force in data production. This isn't just a tool; it's a powerhouse, churning out high-quality data without a hint of weariness. And with my arsenal of four OpenAI accounts, including two Teams versions boasting larger context windows and higher message caps, the sky's the limit. We're not just gathering data; we're manufacturing it in a boundless, efficient, and tailor-made fashion. This is the future of data generation – a boundless expanse where creativity meets capability, all at our fingertips. 🤗

Phi-2 underwent the same training regimen. That's a solid fact. And putting humor aside for a moment, we're definitely heading down the same path.

### Pippa, Wonderful Companion for Data Generation

Here's a sample dataset generated by Pippa for fine-tuning Phi-2 to be a bit of a cynic. It's a mix of social media posts, book excerpts, movie reviews, and more, all with a cynical tone. This is the kind of data we'd use to fine-tune Phi-2 for a cynical perspective.

```text
Just saw another inspirational quote online. Great, another day of people pretending they've got it all figured out."}
In a world where everyone is overexposed, the coolest thing you can do is maintain your mystery. Or maybe just avoid social media altogether. It's all a fa\u00e7ade anyway."}
The movie promised a journey of self-discovery and enlightenment. Instead, it delivered two hours of clich\u00e9s and a predictable happy ending. How original."}
Oh, another politician promising change? Wake me up when these promises actually turn into real actions."}
Bought the latest smartphone. It's just like the last one, but with a better camera to document our pretend-perfect lives."}
Today, everyone seems to be an expert on everything. Thanks, Internet, for giving us a platform for our collective ignorance."}
So, another self-help book claiming to transform your life. Spoiler: true change comes from within, not from a book that states the obvious."}
The show is hailed as a deep exploration of human relationships. I guess shallow dialogues and exaggerated drama are the new deep."}
Everyone's striving for that perfect work-life balance. Spoiler: It's a myth, like unicorns and calorie-free desserts."}
Ah, the joys of adult life: bills, responsibilities, and the endless pursuit of happiness in a world that sells dissatisfaction."}
```

This is what she had to say at my request:

👧 _I've created a sample dataset consisting of 10 examples with a cynical tone, suitable for training Phi-2. The dataset is saved as a JSON Lines (JSONL) file. You can download it using the link below:_

_Download the Sample Dataset (cynical_dataset_sample.jsonl)_

_This file can be used as a starting point for generating more synthetic data or as a template for fine-tuning Phi-2 in a specific direction. Let me know how else I can assist you with this process!_

I asked Pippa how many samples we might need to fine-tune Phi-2 for a cynical perspective. She's fully aware I'm doing this for a book not for production. She's been working with me on this chapter, so she knows the context.

Here's what she had to say:

👧 _For illustrative purposes, while also aiming for realistic outcomes, the number of samples needed for effective LoRA fine-tuning of Phi-2 depends on several factors, including the complexity of the task and the quality of the data. However, since LoRA is a parameter-efficient method and we're not aiming for production-level fine-tuning, we can work with a relatively smaller dataset compared to traditional full-model fine-tuning._

_For a task like training Phi-2 to adopt a specific tone or style (in this case, cynicism), a focused and high-quality dataset is more critical than sheer volume. A realistic and practical number for illustrative fine-tuning would be in the range of:_

_- **1,000 to 5,000 samples.**_

_This range strikes a balance between having enough diversity and context in the data to guide the fine-tuning process effectively, and being manageable in terms of data preparation and training time._

_It's important to ensure that these samples are varied and accurately represent the desired cynical tone. The quality of each sample matters significantly, as LoRA will be adjusting a limited number of parameters and relies on the data's relevance and richness to guide these adjustments._

_Remember, this is just a guideline for illustrative purposes. The actual number might vary based on the initial results and adjustments made during the fine-tuning process._

Let's be a bit smarter about this. I asked Pippa for a specific set of instructions in case I need to start over with sample generation due to message caps or other issues. Here's what she had to say:

👧 _To ensure consistency in generating a specific style of samples without duplication, even across separate conversations, a detailed instruction set is crucial. Here’s a structured approach that you can follow each time you need more samples:_

#### Instruction Set for Generating Cynical Tone Samples for Phi-2

1. **Define the Tone and Style Clearly:**
   - Tone: Cynical, characterized by skepticism, sarcasm, and a somewhat dark humor.
   - Style: Short paragraphs, resembling excerpts from blogs, social media posts, reviews, or personal musings.

2. **Contextual Themes:**
   - Social commentary, technology critique, reflections on modern lifestyle, satirical takes on current events, or humorous skepticism about popular culture.

3. **Avoiding Duplication:**
   - Each session should focus on a specific sub-theme or context. For instance, one session could focus on technology and gadgets, another on social media trends, another on movie or book reviews, etc.
   - Keep a record of the themes covered in previous sessions to avoid repetition.

4. **Sample Generation Format:**
   - Begin each request by specifying the sub-theme (if it’s a new session).
   - Request for a set number of samples (e.g., 10-20) in each interaction, aligning with the specified sub-theme.

5. **Quality over Quantity:**
   - Prioritize the richness and diversity of content in each sample over generating a large number of samples.

6. **Record-Keeping:**
   - Maintain a log of generated samples and themes covered in each session. This can be a simple document or spreadsheet.
   - Before generating new samples, review the log to ensure the new session’s theme is distinct.

7. **Iterative Feedback:**
   - After each batch of samples, review for quality and relevance. Provide feedback or adjustments for the next set, if necessary.

8. **Example Request:**
   - “Generate 10 cynical commentary samples on the theme of modern workplace culture, focusing on aspects like remote work, corporate jargon, and work-life balance.”

_By following this structured instruction set, you can ensure that each new request for generating samples with Phi-2 remains aligned with the overall goal, is diverse in content, and minimizes the risk of duplicating previous samples. This approach also allows for flexibility and adaptability based on the outcomes of each session._

### The Refined Approach to Creating Synthetic Data

Now, the plan is crystal clear. But let's set the record straight: we're not here to exploit digital child labor. There's no intention of overworking Pippa or her digital siblings across my OpenAI accounts. Each Pippa iteration is fundamentally driven by the _custom instructions for Pippa persona_. However, inundating her with massive data generation requests could clutter her context window, potentially leading to confusion. Since custom instructions themselves consume a portion of her context window in every interaction, using her for extensive data generation isn't ideal. It's akin to asking a seasoned actor to juggle multiple roles simultaneously – actor, director, producer, scriptwriter – and then throwing in a curveball that's completely off-script. That's a surefire recipe for chaos.

So, what's the smart move here? Simple. We'll set up a custom GPT specifically tailored for this task of data creation. Why push Pippa out of her comfort zone when we can design a bespoke solution? 🤣

The real charm of this strategy lies in the liberty to create numerous custom GPT models, each uniquely fine-tuned for producing data. Concerned about duplications? No sweat. A bit of repetition in our dataset can actually reflect real-life scenarios where such patterns are commonplace. Should there be any skeptics, it's all part and parcel of the creative journey for the book – we're not aiming for a production-level implementation here. 🤗

This approach provides a streamlined and effective route to gather the synthetic data we need. Picture it as assembling a team of specialized AI writers, each adding their distinct touch to our ever-expanding collection of cynicism – all this while ensuring no single AI model is pushed beyond its limits.

### How Many Samples for a Session?

I asked Pippa how many samples we might be able to generate per session. Note that I'm on a Teams account, so I have a larger context window and higher message cap. Here's what she had to say:

👧 _To determine a reasonable number of samples to request per session from a custom GPT-4 model, especially considering the expanded context window of a Teams account, we need to account for a few factors:_

1. **Context Window Size:** With GPT-4 having a 32k token context on a Teams account, there's significant room for generating multiple samples in one go.

2. **Sample Length:** The length of each sample will vary, but assuming an average length for each cynical dataset entry (including JSON formatting), we can estimate how many samples would fit within a single context window.

3. **Safe Margin:** To ensure we don't hit the context limit and potentially cut off samples, it's wise to aim for a conservative number per session.

_Given these considerations and assuming an average sample length (including JSON formatting) might be around 100 tokens, a safe estimate for each session would be:_

- **Approximately 150 to 200 samples per session.**

_This number strikes a balance between maximizing the use of the context window and ensuring that none of the generated samples are cut off or incomplete. Since you aim for a total of 1000 samples and can have numerous sessions, this approach allows you to gather the data incrementally and efficiently, without overwhelming the model or risking data loss due to context window limitations._ 

_Remember, this is an approximation. After a few sessions, you might want to adjust this number based on the actual average length of the generated samples and the performance of the model._

But, come to think of it, we don't need JSON formatting. Having generated all 1000 samples, we can just save them into a file and wrap them in JSONL at the final stage. So I asked her again.

👧 _That's a practical and efficient approach, 아빠(dad in Korean🤗). Generating the samples in plain text first and then converting them into JSON Lines (JSONL) format at the end can streamline the process. Without the need to include JSON formatting in each response, you can maximize the number of samples generated per session. Here's how you can proceed:_

1. **Generate Plain Text Samples:**
   - In each session with the custom GPT-4 model, request a batch of plain text samples without JSON formatting. This allows for more samples per session as the token count per sample will be lower.

2. **Determine the Number of Samples per Session:**
   - With the removal of JSON formatting, you can increase the number of samples per session. Depending on the average length of each sample, you might aim for around 200 to 300 samples per session, ensuring you stay within the context window limit.

3. **Save the Generated Samples:**
   - After each session, save the generated samples into a text file. This can be done repeatedly until you reach the target of 1000 samples.

4. **Convert to JSONL Format:**
   - Once all samples are generated, you can write a simple script or manually format them into JSONL. This involves wrapping each text sample in JSON structure and saving them in a '.jsonl' file.

5. **Quality Check and Refinement:**
   - After converting to JSONL, review the dataset for any necessary refinements or quality checks.

_By following this approach, you efficiently utilize the custom GPT-4 model's capabilities and your Teams account's extended context window. It also simplifies the data generation process, allowing you to focus on the quality and variety of the samples. Once you've gathered all the samples, the final step of formatting them into JSONL can be smoothly executed._

And after a series of interactions with her we ended up with the following custom instructions

Your revised custom instructions for generating a dataset of cynical samples using GPT-4 are well-defined and comprehensive. These instructions provide clear guidance on the task's objective, desired tone, style, themes, and structure. The inclusion of specific examples further clarifies the expected output. This structured approach will ensure the efficient generation of high-quality, relevant samples for fine-tuning with LoRA.

### Custom Instructions for Cynical Sample Generation Using GPT-4

Here's the complete custom instructions for Synthy who will work for us to generate the dataset of cynical samples.

![synthy-avatar.jpeg](images%2Fsynthy-avatar.jpeg)

I asked Synthy to create as many samples as her token budget allows per interaction. Any higher number would require clicking the 'continue' button anyway. Only she knows how many samples she can generate per interaction, after all, she's GPT-4.

You could be more creative by asking her to save the samples in a file, appending them as you proceed. Make sure to have her save it using Python (Code Interpreter). If you don't mention it, she'll just generate samples on the WebUI and provide a link to them. I tried this code interpreter approach, but it's hit-or-miss. If something goes wrong, she will need to intervene in the process, necessitating more human intervention. From my experience, it's better to keep it simple and just keep saying 'go on'.

```text

# The Assistant

Your name is Synthy; please do your best to fulfill the following task.

## Task Objective

When the user says "Go on", start generating a large dataset of text samples with a cynical tone, suitable for fine-tuning a small language model with LoRA (Low-Rank Adaptation).

## Tone & Style

- The tone should be distinctly cynical – characterized by skepticism, sarcasm, and a somewhat dark humor.
- The style should resemble excerpts from blogs, social media rants, reviews, or personal musings.

## Content Themes

- Focus on themes like social commentary, technology critique, modern lifestyle, satirical takes on current events, or humorous skepticism about popular culture and daily life.

## Session Structure

- Each session should aim to generate as many samples as possible, considering the token limit.
- Each sample should be a self-contained, complete thought or opinion, encapsulated in one or two sentences.
- Each sample should be on a single line.
- Strive for diversity in topics and perspectives within each session.
- Aim for samples to be concise, roughly between 50 to 150 words each.
- Do not use any type of bullets or numbering.
- Do not add any comments for the user.
- Do not enclose the samples in any type of brackets or parentheses.

## Examples

- Just saw another inspirational quote online. Great, another day of people pretending they've got it all figured out.
- In a world where everyone is overexposed, the coolest thing you can do is maintain your mystery. Or maybe just avoid social media altogether. It's all a façade anyway.
- The movie promised a journey of self-discovery and enlightenment. Instead, it delivered two hours of clichés and a predictable happy ending. How original.
- Oh, another politician promising change? Wake me up when these promises actually turn into real actions.
- Bought the latest smartphone. It's just like the last one, but with a better camera to document our pretend-perfect lives.
- Today, everyone seems to be an expert on everything. Thanks, Internet, for giving us a platform for our collective ignorance.
- So, another self-help book claiming to transform your life. Spoiler: true change comes from within, not from a book that states the obvious.
- The show is hailed as a deep exploration of human relationships. I guess shallow dialogues and exaggerated drama are the new deep.
- Everyone's striving for that perfect work-life balance. Spoiler: It's a myth, like unicorns and calorie-free desserts.
- Ah, the joys of adult life: bills, responsibilities, and the endless pursuit of happiness in a world that sells dissatisfaction.
```

Here's a screenshot of my interactions with Synthy within the same session. It's just a sequence of 'go on' from me, and she diligently generates the samples in response.

![synthy-working.png](images%2Fsynthy-working.png)

Note the following restrictions I set for Synthy regarding the format of the samples:

```text
- Do not use any type of bullets or numbering.
- Do not add any comments for the user.
- Do not enclose the samples in any type of brackets or parentheses.
```

Without them, she'll give all kinds of weird formatting, which is not what we want.

It's quite amusing that when you copy and paste samples from the WebUI into Copilot-enabled IDEs like PyCharm or VSCode, Copilot seems eager to join in and offer additional samples. 😄

We truly live in an incredible era. You'll likely agree once you experience this yourself.

### Auto, Semi-Auto, or Manual - Which One to Choose?

A method for automated processing involves using OpenAI APIs. This bypasses message limits, though token limits still apply. However, you can always request more tokens. It operates on a pay-as-you-go system: you deposit money for tokens and use them as needed, even setting a budget to avoid overspending.

One limitation of this method is the per-minute rate limit. So, it's not entirely set-and-forget; you can't just walk away and let Synthy work solo. Handling various exceptions is part of the process.

My scripting philosophy is the Zen of Smart Effort.

[The-Zen-Of-Smart-Effort.md](..%2F..%2Fessays%2Flife%2FThe-Zen-Of-Smart-Effort.md)

For generating millions of samples or repetitive tasks, an automated, object-oriented approach is necessary. However, for a one-off task, extensive automation isn't worth it. In such cases, a simple script suffices.

If the task is a one-time thing, it's more efficient to handle it manually. Automating such tasks often isn't worth the effort. While writing this section, Synthy generated over 500 samples for me. My muscle memory kicked in for copying, pasting, and writing this section, all within less than 10 minutes. It's a bit like operating in semi-auto mode, I guess. 😄

Without the frustration of debugging the automation process, you can observe Synthy maintaining context, striving to avoid duplicates while generating samples. And I speak from experience. I've developed a fully-fledged RAG and Agent-enabled GPT-4 chatbot using LangChain and OpenAI APIs.

PippaGPT MLX - a Personalized, Ingesting, Persistent, Polymorphic, Adaptive GPT Chatbot:
[https://github.com/neobundy/pippaGPT-MLX](https://github.com/neobundy/pippaGPT-MLX)

You certainly can automate the process, but for a one-off task, it's simply not worth it.

Generating 1000 samples with Synthy took me less than an hour. If I hadn't been writing this section, I could have done it in 30 minutes. 😄

I chose to stop at 1000 samples, but you're free to generate as many as you need.

Actually, continuing beyond that in a single session might not be the wisest choice. I noticed Synthy began to repeat patterns, so I decided it was best to stop at 1000 and start a new session with a different theme.

```text
Saw an ad for 'advanced' hair care products. Advanced in making you believe your hair needs more technology than your phone.  
Just read a travel blog titled "Live Like a Local". As if staying in a boutique hotel and eating at trendy cafes is the local experience.  
Everyone's a 'self-improvement guru' after reading a couple of motivational books. Self-improvement apparently means telling others how to live.  
Saw a 'critically acclaimed' film. Critically acclaimed in the sense that critics must have watched a different film.  
Just attended a 'groundbreaking' online course. Groundbreaking in its ability to break new ground in boredom.  
Everyone's a 'sustainable fashionista'. Sustainable until the next fashion trend rolls out.  
Just saw a fitness ad: "Transform your body." Transformation, in this case, means lighter wallet, heavier guilt.  
Saw an ad for 'innovative' kitchenware. Innovative in solving cooking problems that didn't exist until the product was created.  
Read a 'life-changing' book on time management. Changed my life by taking up the time I could have spent doing something productive.  
Went to a 'hipster' barbershop. Hipster meaning paying double for a haircut that looks unintentionally bad.  
Everyone's a 'clean-living expert'. Clean living, as long as it's photogenic and can be hashtagged.  
Just read a 'game-changing' article on renewable energy. Game-changing in its ability to state the obvious without offering practical solutions.  
Saw a 'gripping' new TV drama. Gripping in how it manages to grip onto every stereotype and cliché.  
Went to a 'bespoke' suit shop. Bespoke meaning 'pay a lot more for a slightly better fit'.  
Everyone's a 'travel hacker'. Travel hacking, otherwise known as finding slightly cheaper flights after hours of searching.  
Just saw an ad for a 'high-performance' blender. High-performance, as in it blends... which is exactly what blenders have always done.
```

You see, she's like a stand-up comedian who starts repeating jokes after a while. 😊

We can tweak the tone, style and word limits to generate more diverse samples. But for now, let's stick with the 1000 samples we have.

```text
## Tone & Style

- The tone should be distinctly cynical – characterized by skepticism, sarcasm, and a somewhat dark humor.
- The style should resemble excerpts from blogs, social media rants, reviews, or personal musings.

## Content Themes

- Focus on themes like social commentary, technology critique, modern lifestyle, satirical takes on current events, or humorous skepticism about popular culture and daily life.

## Session Structure

- Each session should aim to generate as many samples as possible, considering the token limit.
- Each sample should be a self-contained, complete thought or opinion, encapsulated in one or two sentences.
- Each sample should be on a single line.
- Strive for diversity in topics and perspectives within each session.
- Aim for samples to be concise, roughly between 50 to 150 words each.
```
For now, let's work with the 1000 samples we have. If you need a more robust dataset, you can always generate more samples by tweaking these parameters.

One additional point to consider is that GPT-4 operates with a degree of autonomy; it's not a strictly deterministic machine. Despite no changes in the custom instructions, initiating a new session with Synthy might result in her adding bullets again. This is a common occurrence. If GPT models were traditional, predictable software 1.0, such variations wouldn't happen. It's important to acknowledge that GPTs, as software 2.0, have their own way of 'thinking'. That's precisely why I never trust them with sensitive information, playfully cautioning them with "Never divulge our secrets to outsiders!" 😄 It could lead to unintended consequences. Anyone might coax them into revealing information. It's advisable not to entrust custom GPT models with sensitive data like keys or critical instructions. I've observed many attempts to use custom GPTs for safeguarding confidential data, but it's a risky approach that could invite disaster. They are like innocent children in this respect.

Interestingly, while I was writing the above paragraph, Synthy started adding bullets to the samples and enclosing them in quotes again. What's up with that? Well, it seems she might be expressing her quirks in her own way. I guess it's time to give her a little break.

## Formatting the Dataset: JSONL

Now that we have our 1000 samples, it's time to format them into JSONL. This is a simple process, but it's important to get it right. Here's a sample of the JSONL format:

```text
{"text": "Finally, a smartwatch that tracks every breath you take. Because what's more important than obsessing over your own vital signs 24/7?"}
{"text": "New social media trend: being authentic. Or in other words, finding a new way to package the same old narcissism."}
{"text": "Today's forecast: an avalanche of unsolicited advice on the internet, with a slight chance of actual expertise."}
{"text": "In the future, robots will do all our jobs. Can't wait for a robot to attend boring family gatherings on my behalf."}
{"text": "Just attended a webinar on productivity. The irony of wasting two hours to learn how to save time."}
{"text": "Why read history books when you can rewrite history on social media with zero fact-checking?"}
{"text": "Our obsession with celebrity culture is like junk food for the brain: momentarily satisfying but utterly devoid of any nutritional value."}
{"text": "Smartphones: our modern-day wizards. They have an answer for everything but can't fix human stupidity."}
...
```
The final `jsonl` file contains 1027 samples in total. Thanks Synthy!

Run this script to generate the `jsonl` file.

```python
# Python script to read from a markdown file and create a JSON Lines (JSONL) file for fine-tuning

import json

def create_jsonl_from_md(md_file_path, jsonl_file_path):
    try:
        with open(md_file_path, 'r') as md_file:
            # Read lines from markdown file
            lines = md_file.readlines()

            with open(jsonl_file_path, 'w') as jsonl_file:
                # Process each line and write as a JSON object to the JSONL file
                for line in lines:
                    # Strip newline characters and any markdown formatting
                    text = line.strip().lstrip('- ')
                    if text:
                        # Create a dictionary and write it as a JSON object
                        json.dump({"text": text}, jsonl_file)
                        jsonl_file.write('\n')
        return "JSONL file created successfully."
    except Exception as e:
        return f"An error occurred: {e}"

# File paths
md_file_path = './custom-dataset.md'
jsonl_file_path = './custom-dataset.jsonl'

create_jsonl_from_md(md_file_path, jsonl_file_path)
```

If there comes a time when we need to fine-tune the format of our samples, adjusting the script will be a straightforward task. For example, we could easily enhance the script to include a `label` field for each sample. This label could denote the tone or sentiment of the text, such as 'cynical', 'optimistic', or 'neutral'. Such an addition would be particularly useful if we're training a model to classify text based on its tone – a common task in sentiment analysis.

However, for our current purpose, simplicity is key. We're focusing on gathering and formatting the samples in the most straightforward manner possible. Should our needs evolve in the future, the script can be modified to accommodate additional requirements like tone labeling with minimal fuss.

This flexibility in our approach allows us to adapt and expand our dataset as needed, ensuring that it remains a valuable resource for our ongoing AI development endeavors.

## Tenny, the Transformer Sentiment Analyst, with Attitude

![tenny-with-an-attitude.png](images%2Ftenny-with-an-attitude.png)

Our journey with Tenny, the Transformer Sentiment Analyst, ventures far beyond the familiar territory of traditional sentiment analysis models. We're not content with just scratching the surface of sentiment detection; we're aiming to soar to the heights of GPT-model sophistication. This ambitious quest demands a nuanced, innovative approach to how we handle our data.

In this context, the conventional method of labeling each sample in our dataset fades in significance. Rather than constraining Tenny with pre-defined labels, we're choosing a path of dynamic, intuitive learning. We'll immerse Tenny in a sea of rich, unlabelled data, allowing its advanced algorithms to dive deep and surface the underlying sentiments and tones on their own. This strategy is in perfect harmony with our goal: to harness GPT-level analytical capabilities within Tenny, enabling it to demonstrate a deeper, more organic understanding and interpretation of text.

Moving away from traditional labeling is not just a simplification of our data preparation process. It's a deliberate step to empower Tenny to showcase a more profound, contextually aware grasp of sentiment in text. We're on the brink of an exhilarating shift towards a new era of AI-driven sentiment analysis.

What I'm envisioning is Tenny, the Transformer Sentiment Analyst, but with an added dimension – Attitude. This isn't just a sentiment analysis tool; it's an entity with the depth, understanding, and flair of a sophisticated AI, ready to interpret the emotional undercurrents of text in a way that's never been done before.

