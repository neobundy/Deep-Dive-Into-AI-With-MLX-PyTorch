# Deep Dive into LLaVA - Large Language and Vision Assistant Part II

![time.jpeg](images%2Ftime.jpeg)

✍️ [Part I](README.md) | ✍️ [Part II](README2.md) | ✍️ [Part III ](README3.md)

**📝 Original Paper**: https://arxiv.org/abs/2304.08485

**📝 v1.5 Paper**: https://arxiv.org/abs/2310.03744

## Overview of the Original Paper

![conceptual-diagram-of-llava.png](images%2Fconceptual-diagram-of-llava.png)

The original paper delves into the concepts discussed in Part I, emphasizing the generation of instruction-following data as a focal point. Let's revisit this concept with a practical example.

**GPT-assisted Visual Instruction Data Generation**

With the surge in public multimodal data, including image-text pairs, from resources like CC to LAION, the scarcity of multimodal instruction-following data becomes apparent. This scarcity is partly attributed to the intricate and time-intensive process of generating such data, especially when relying on human annotation. Drawing inspiration from the prowess of recent GPT models in text annotation, the researchers explored using ChatGPT/GPT-4 for enriching the pool of multimodal instruction-following data, capitalizing on existing image-text pairs.

Given an image `X_v` and its caption `X_c`, we can naturally devise a set of instructive questions `X_q` aimed at prompting detailed descriptions of the image content. The researchers employed GPT-4 to craft such questions, thereby proposing a straightforward method to augment an image-text pair into its instruction-following counterpart, denoted as "`Human: X_q X_v<STOP> Assistant: X_c<STOP>`." Although this method is cost-effective, it often lacks the depth and diversity in both questions and answers.

![paper1-table1.png](images%2Fpaper1-table1.png)

To address this limitation, the researchers utilized text-only versions of GPT-4 or ChatGPT, which accept textual input, to generate instruction-following data that includes visual content. This involves encoding the image through two symbolic representations: captions, which offer varied perspectives of the visual scene, and bounding boxes, which identify and locate objects within the scene.

This approach enabled them to translate images into sequences recognizable by an LLM. Utilizing COCO images, the researchers generated three distinct types of instruction-following data. Each data type begins with manually created seed examples to guide in-context learning with GPT-4:
- **Conversations**: Designing dialogues that simulate an assistant responding to queries about an image, covering a wide array of visual details from object identification to spatial reasoning.
- **Detailed Descriptions**: Formulating a list of questions intended to elicit comprehensive descriptions of an image, from which GPT-4 generates detailed responses.
- **Complex Reasoning**: Developing questions that necessitate advanced reasoning, with answers relying on a logical, step-by-step analysis.

In total, they've amassed 158K unique samples of language-image instruction-following data, including 58K in conversations, 23K in detailed descriptions, and 77K in complex reasoning. Early experiments comparing the use of ChatGPT and GPT-4 indicated that GPT-4 consistently yields higher-quality instruction-following data, particularly in spatial reasoning tasks.

### Architecture of LLaVA

![conceptual-diagram-of-llava.png](images%2Fconceptual-diagram-of-llava.png)

We've previously explored the architecture of LLaVa in Part I. Now, let's focus our attention on the intricacies of its training process.

### Training LLaVA: A Two-Stage Instruction-Tuning Approach

Training the LLaVA model involves creating and utilizing multimodal instruction-following sequences derived from image data `X_v`. For each image, a multi-turn conversation dataset is generated, consisting of pairs of instructions and responses:

![paper1-expression1.png](images%2Fpaper1-expression1.png)

Where `T` represents the total number of turns in the conversation. These pairs are sequenced to form a continuous dialogue, with instructions `X^t_instruct` for each turn defined as outlined in the following formula:

![paper1-formula2.png](images%2Fpaper1-formula2.png)

This structured sequence becomes the foundation for the multimodal instruction-following sequence, which the model uses to predict the assistant's response. As shown in formula:

![paper1-formula3.png](images%2Fpaper1-formula3.png)

The probability of the target answers `X_a` for a sequence of length `L` is calculated. The probability is determined by the model's trainable parameters `θ`, and it takes into account the instructions and answers from all previous turns before the current prediction token `x_i`.

![paper1-table2.png](images%2Fpaper1-table2.png)

In the illustration provided in Table 2, you can observe the input sequence used to train the model. It emphasizes that the image context `X_v` is consistently present, grounding each answer throughout the dialogue. For clarity and readability, system messages and previous `<STOP>` tokens are omitted from the conditions:

![paper1-formula3.png](images%2Fpaper1-formula3.png)

The LLaVA model undergoes a two-phase instruction-tuning procedure during its training:

**Stage 1: Pre-training for Feature Alignment**
The researchers refine the CC3M dataset down to 595K image-text pairs to balance concept coverage with training efficiency. These pairs are transformed into instruction-following data through a straightforward expansion method described earlier. Each piece of data is viewed as a single-turn conversation where the input instruction `X_instruct` is formed by randomly choosing a question `X_q`, a language instruction prompting the assistant to describe the image briefly. The caption serves as the ground truth answer `X_a`. During training, both the visual encoder and LLM weights remain unchanged, with only the projection matrix `W` being updated to maximize the likelihood of the target answers, aligning the image features `H_v` with the pre-trained LLM embeddings. This process effectively trains a visual tokenizer compatible with the LLM.

**Stage 2: Fine-tuning End-to-End**
The visual encoder's weights remain unchanged, while the model undergoes further updates involving both the projection layer's pre-trained weights and the LLM within LLaVA, denoted as `θ={W,ϕ}` in the formula (3). Two specific scenarios are targeted:

- **Multimodal Chatbot**: Fine-tuning uses the 158K language-image instruction-following data to develop a Chatbot. Training samples include multi-turn conversations and single-turn responses from detailed descriptions and complex reasoning tasks, sampled uniformly.
  
- **Science QA**: Utilizing the ScienceQA benchmark, a dataset with multimodal science questions annotated with detailed explanations, the model is trained to provide reasoning in natural language. Training data is organized as a single-turn conversation, with questions and contexts as `X_instruct`, and reasoning and answers as `X_a`.

Through this structured, two-stage process, LLaVA is adeptly tuned to understand and generate language in response to multimodal inputs, exemplifying the model's ability to integrate visual and textual data effectively.

### Practical Evaluation

![paper1-table3.png](images%2Fpaper1-table3.png)

Despite being trained on a relatively modest dataset of approximately 80,000 unique images tailored for multimodal instruction-following tasks, LLaVA exhibits reasoning capabilities that closely parallel those of the multimodal GPT-4, especially within the context of these samples. Notably, even when presented with images that fall outside its immediate domain, LLaVA adeptly interprets the scene and adheres to the provided instructions to deliver coherent responses. This stands in contrast to models like BLIP-2 and OpenFlamingo, which are more inclined to describe images rather than engaging with user instructions to formulate relevant answers.

## Overview of the v1.5 Paper Updates

**📝 v1.5 Paper**: https://arxiv.org/abs/2310.03744

‼️ **Important Note**: The current model iteration is v1.6; however, the most recent publication available is the v1.5 paper. In the field of AI research, it is common for the model updates to outpace the corresponding academic literature.

The v1.5 paper represents the most recent scholarly work detailing enhancements and advancements in the LLaVA model framework. It is important to note that while the latest iteration of the model is v1.6, the convention within AI research typically sees the model precede the corresponding paper. Therefore, this paper sets the stage for understanding the improvements that have been integrated into LLaVA since its previous version, laying the groundwork for the subsequent release of the v1.6 paper.

This paper presents findings on the remarkable capabilities and efficiency of the fully-connected vision-language cross-modal connector within LLaVA. By integrating a _CLIP-ViT-L-336px_ with an _MLP projection_ and enriching the training set with academically oriented _VQA data_, complemented by straightforward response formatting prompts, the researchers have significantly elevated the model's performance. This has led to the establishment of robust baselines and the achievement of state-of-the-art results across a spectrum of 11 benchmark challenges. 

Notably, the 13B parameter model leverages just 1.2 million pieces of publicly accessible data and completes its training within approximately one day on a single node equipped with 8 A100 GPUs. The researchers wishes that these enhancements will democratize access to cutting-edge LMM research and they are committed to making thr code and the enhanced model publicly available for the broader research community.

### Key Enhancements in LLaVA v1.5 

LLaVA has set a precedent in the field of visual instruction tuning, demonstrating exceptional visual reasoning skills that often outshine recent models across various real-life benchmarks. However, it has been observed to fall short in academic benchmarks, which typically call for concise answers, due to its lack of pretraining on extensive datasets. 

![paper2-table1.png](images%2Fpaper2-table1.png)

The researchers' exploration began with examining the scaling effects of data, model size, and image resolution using three datasets listed in Table 1. 

![paper2-table2.png](images%2Fpaper2-table2.png)

Subsequently, they benchmarked the enhanced model against existing large multimodal models across 12 diverse benchmarks in Table 2. The findings affirm LLaVA's powerful architecture and its data-efficient approach in visual instruction tuning, achieving peak performance with considerably less computational power and training data than other methods.

**Response Formatting Prompts**

One of the challenges for models like InstructBLIP in balancing short- and long-form VQA stems from ambiguous response formatting prompts. Prompts such as "Q: {Question} A: {Answer}" do not specify the expected output format, potentially causing an LLM to default to brief answers, even when a more expansive response is appropriate. This issue is exacerbated when the LLM itself is not fine-tuned. Addressing this, the researchers proposed a singular response formatting prompt that distinctly conveys the required output format, encouraging concise answers without necessitating additional processing by ChatGPT. This simple yet effective strategy leads to LLaVA surpassing its predecessors in performance, as highlighted by the leap in MME scores (1323.8 vs 502.8) and outperforming InstructBLIP by 111 points, as demonstrated in Table 1.

**MLP Vision-Language Connector**

Motivated by the enhancements seen in self-supervised learning when transitioning from linear to MLP projections, they enhanced the connector between vision and language modalities. Implementing a two-layer MLP improves LLaVA's multimodal interaction, signaling a departure from the original linear design.

**Academic Task-Oriented Data**

In addition to the standard datasets, they incorporated academic-task-oriented VQA datasets focused on areas like OCR and region-level perception. This inclusion broadens LLaVA's expertise, as shown in Table 1, where it excels even with a subset of datasets used by InstructBLIP. Moreover, adding region-level VQA datasets enhances the model's fine-grained visual detail localization.

**Additional Scaling**

To give the LLM a clearer "view" of image details, they scaled up the input image resolution and integrate the GQA dataset as a new visual knowledge source. The incorporation of ShareGPT data and an upscale to a 13B LLM underscore the significance of the underlying LLM's capabilities for visual dialogue. The ultimate configuration, termed LLaVA-1.5 and delineated in the latter part of Table 1, marks a substantial leap in performance, outshining the initial LLaVA model and establishing new frontiers in LMM research.

### Technical Improvements of LLaVA v1.5 and v1.6

![technical-improvement.png](images%2Ftechnical-improvement.png)

The advancement of LLaVA-1.6 includes a deliberate focus on high-resolution imaging, designed to refine the model's data efficiency and perceptual acuity. By accommodating high-resolution images, the model significantly elevates its ability to discern and process fine-grained visual details. This enhancement plays a critical role in reducing instances of 'model hallucination'—a phenomenon where the model speculates or “imagines” content in lieu of clear visual information, commonly arising from low-resolution inputs.

The 'AnyRes' technique forms the backbone of this approach, offering a flexible framework that welcomes various high-resolution image formats. This technique utilizes a grid configuration system, represented as `{2×2, 1×{2,3,4}, {2,3,4}×1}`, which skilfully strikes a balance between achieving peak performance efficiency and maintaining sustainable operational costs.

This resolution-centric design, with its data-efficient underpinnings, proves instrumental in enhancing the overall competence of LLaVA-1.6, solidifying its position as a benchmark in the field of detailed visual understanding and multimodal learning models.

![paper2-table3.png](images%2Fpaper2-table3.png)

![paper2-table4.png](images%2Fpaper2-table4.png)

The recent updates to the LLaVA model reflect an intricate blend of high-quality data and strategic partnerships, ensuring that LLaVA-1.6 not only meets but also anticipates users' needs across diverse scenarios. Here's a detailed breakdown of the data mixture enhancements:

1. **High-Quality User Instruct Data:**
   - High-quality requirements are twofold: variety in task instructions to reflect a wide range of real-world user intents, and a caliber of responses designed to garner affirmative user feedback. Incorporating data from premier sources like LAION-GPT-V and ShareGPT-4V sets the foundation.
   - Complementing this, the researchers have curated a focused 15K sample visual instruction tuning dataset derived from genuine LLaVA demo user requests, ensuring relevance and practicality. Rigorous screening is applied to remove any content with privacy issues or harmful potentials, while leveraging GPT-4V's capabilities for response generation.

2. **Multimodal Document/Chart Data:**
   - In refining their OCR capabilities, the researchers have made a strategic decision to exclude TextCaps, recognizing its overlap with TextVQA training images, to hone the model's zero-shot OCR skills—assessing it with live TextVQA challenges.
   - Replacement with DocVQA and SynDog-EN datasets aim to maintain and augment OCR prowess. Meanwhile, inspired by Qwen-VL-7B-Chat’s success, they’re incorporating ChartQA, DVQA, and AI2D, thereby significantly improving our chart and diagram understanding capabilities.

3. **Scaling the LLM Backbone:**
   - The researchers are expanding beyond their current Vicuna-1.5 (7B and 13B) models to explore models like Mistral-7B and Nous-Hermes-2-Yi-34B that bring distinct advantages including flexible commercial usage, strong bilingual support, and enhanced language model capacity.
   - Such expansion allows LLaVA to cater to a broader user base and more application scenarios within the community. They've observed that the LLaVA recipe is highly adaptable across different LLM sizes and scales gracefully, even up to the substantial 34B model capacity.

These methodologically layered updates aim to solidify LLaVA-1.6's standing as a model not just of robust performance but of thoughtful construction—mindful of user diversity, data quality, and the imperative to provide real-world, applicable solutions in visual reasoning and language understanding.

### Limitations of LLaVA

While LLaVA-1.5 has made remarkable strides, it is crucial to recognize its current limitations. Firstly, the model's reliance on full image patches can lead to extended training times per iteration. Although visual resamplers have been proposed to decrease the number of patches processed by LLMs, they have yet to reach the efficient convergence levels of LLaVA with a similar volume of training data, possibly due to their increased parameter counts. The creation of a visual resampler that is both sample-efficient and scalable would significantly enhance the training of instruction-following multimodal models.

Secondly, LLaVA-1.5's current design does not support the processing of multiple images concurrently, a limitation stemming from both the absence of corresponding instruction-following data and constraints related to context length.

Thirdly, while LLaVA-1.5 can follow complex instructions, its problem-solving abilities may be constrained within certain specialized domains. Enhancements in the underlying language model's capabilities, along with the provision of high-quality, domain-specific visual instruction tuning data, could further refine its proficiency.

Lastly, LLaVA-1.5, despite showing a reduced tendency for hallucinations, is not entirely immune to producing them or occasionally spreading misinformation. As such, users should exercise caution and critical judgment when deploying the model in sensitive or high-stakes contexts, such as medical applications.

Now let's look into the codebase.

[Deep Dive into LLaVa - Multi-Modal Language Model Part III](README3.md)