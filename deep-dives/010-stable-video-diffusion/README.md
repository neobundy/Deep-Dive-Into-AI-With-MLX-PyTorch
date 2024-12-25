# Deep Dive into Stability AI's Generative Models - Stable Video Diffusion

![ai-director.png](images%2Fai-director.png)

🔗 https://arxiv.org/abs/2311.15127

In our exploration, we turn our attention to the innovative research detailed in the paper on _Stable Video Diffusion_. The researchers introduce a latent video diffusion model designed for high-resolution, state-of-the-art text-to-video and image-to-video generation. This cutting-edge approach is grounded in the recent advancements of latent diffusion models for 2D image synthesis, which have been adapted to generate videos by integrating _temporal layers_ and refining them with high-quality, albeit small, video datasets.

A notable challenge highlighted by the authors is the lack of consensus in the field regarding the optimal training methodology and the curation of video data. To address this, the paper outlines a three-tiered training strategy for video Latent Diffusion Models (LDMs): starting with text-to-image pretraining, followed by video pretraining, and culminating in finetuning with high-quality video data. The necessity of a meticulously curated pretraining dataset for the generation of high-quality videos is underscored, alongside a systematic approach for dataset curation that includes captioning and filtering strategies.

The researchers then delve into the benefits of finetuning their robust base model on premium video data, resulting in a text-to-video model that rivals proprietary video generation technologies. They also illustrate how their base model serves as a foundational motion representation, facilitating tasks like image-to-video generation and the integration with camera motion-specific LoRA modules.

A particularly striking achievement of their work is the demonstration of a strong multi-view 3D prior within their model. This capability forms the basis for finetuning a multi-view diffusion model that proficiently generates multiple views of objects in a feedforward manner, significantly outpacing image-based methods while requiring a fraction of the computational resources.

Through our analysis, we aim to unpack the methodologies, evaluate the strategic innovations, and assess the potential impact of Stable Video Diffusion on both the academic landscape and practical implementations in video generation technology.

## In Simple Terms

First, let's break down the complex concepts from the paper into simpler terms.

Imagine you want to make a movie, but instead of filming with a camera, you want to create it from scratch on a computer. This is similar to what the researchers in this study are doing, but they're using advanced computer programs to generate videos from text descriptions or a single image.

Here's how they do it:

1. **Training the Computer**: Just like you might learn how to paint by copying from a master painter, the researchers first teach the computer how to create images by showing it millions of examples. This is called "pretraining." They use a special kind of computer program called a "diffusion model," which learns to make images by starting with random noise and then slowly removing the noise to create a picture.

2. **Improving the Training**: Next, they focus on making the computer program better at creating videos. They carefully choose which videos to show the computer so it can learn better. This process is called "data curation," and it's a bit like choosing the best ingredients for a recipe.

3. **Adding Motion**: To make the images move, they teach the computer about different camera movements, like panning sideways or zooming in. They do this using a technique called "Low Rank Adaptation" (LoRA), which is like fine-tuning a musical instrument to get the perfect pitch.

4. **Making Smooth Videos**: Then, they want their videos to be smooth, so they teach the computer to fill in extra frames between the frames they already have, which makes everything look more fluid. It's like when you draw a flipbook and add more drawings between the pages to make the action smoother.

5. **Creating Different Views**: Finally, they train the computer to show objects from different angles, as if you were walking around the object and looking at it from every side. This helps make the videos more realistic.

In the end, they've created a computer program that can take a sentence or a single image and turn it into a high-quality video that follows the description or shows the image in motion. And the really cool part? This program can create these videos much faster and with less computer power than older methods, making it more practical for real-world use.

The paper wraps up by saying that they've built a very effective tool (they call it "Stable Video Diffusion" or SVD) for creating videos from text or images, and they think this tool can help a lot in the field of video generation.

## Prerequisites

Andreas Blattmann et al. Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models (2023)
🔗 https://arxiv.org/abs/2304.08818

![pre-paper-figure1.png](images%2Fpre-paper-figure1.png)

In simple terms, this prerequisite paper discusses how to create videos using a special type of AI model known as a Latent Diffusion Model (LDM). Normally, these models are good at creating individual images, but the challenge is to make them create a series of images that look like a smooth, continuous video.

Here's how they do it:

1. **Pre-trained Model**: They start with an LDM that's already good at making pictures. Think of it as an artist who's learned to paint only single scenes.

2. **Temporal Layers**: To make the artist paint a moving scene (video), they add new layers to the model that are specifically focused on understanding time and motion. These are called "temporal layers."

![pre-paper-figure2.png](images%2Fpre-paper-figure2.png)

3. **Training Temporal Layers**: They train these new layers to look at several frames (images) and understand how things move from one frame to the next. This way, the artist doesn't just paint single scenes anymore but learns how to connect these scenes to create motion.

4. **Combining Spatial and Temporal Information**: The AI model has layers that understand space (spatial layers) and now also time (temporal layers). During training, the spatial layers treat each frame as an individual image, while the temporal layers take these frames and learn to line them up in the correct order to create motion.

5. **Temporal Attention and 3D Convolutions**: The temporal layers use two special techniques to understand motion: "temporal attention," which helps the model focus on how objects move over time, and "3D convolutions," which help the model understand changes in objects' positions and appearances across frames.

6. **Encoding Time**: To help the model understand time better, they use something called "sinusoidal embeddings." It's like giving the model a clock so it knows the timing of each frame in the video.

By doing all this, the researchers teach the LDM to create not just still images, but flowing videos that are smooth and look like they were shot with a camera. It's like teaching a painter not just to paint but to animate their paintings.

## Deep Dive - The Paper

Andreas Blattmann, et al. (2023). Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets
🔗 https://arxiv.org/abs/2311.15127

In the introduction to their study, the researchers embark on an investigation propelled by the rapid advancements in generative image modeling, particularly through the lens of diffusion models. This surge of progress has significantly impacted the development of generative video models, marking notable strides in both academic research and real-world applications. The techniques for training these models vary, ranging from constructing models from the ground up to finetuning pre-existing image models with added temporal dimensions, often employing a combination of image and video datasets.

Interestingly, while much of the research has honed in on the architectural specifics of spatial and temporal layer arrangements, a surprising gap is the lack of inquiry into the role of data selection. This oversight is peculiar, given the universally acknowledged influence of training data distribution on the performance of generative models. Drawing parallels from generative image modeling—where pretraining on a vast, diverse dataset followed by finetuning on a smaller, high-quality dataset is known to enhance performance—the exploration of data and training strategy effects on video modeling remains scant. This study aims to fill this gap by examining the influence of data selection and training strategies, specifically the differentiation between video pretraining at lower resolutions and subsequent finetuning on high-quality videos.

Contrary to prior research, this paper leverages simple latent video diffusion baselines, maintaining a constant architecture and training scheme, to specifically scrutinize the impact of data curation. The researchers identify three pivotal stages in video training: text-to-image pretraining, low-resolution video pretraining on a large dataset, and high-resolution video finetuning on a smaller, high-quality dataset. They propose a systematic method for large-scale video data curation and present an empirical study on its effects, revealing that well-curated pretraining datasets can lead to significant and enduring performance improvements.

Building on these insights, the study applies the curation strategy to a vast video dataset, culminating in a robust pretrained text-to-video base model that encapsulates a general motion representation. This model is then finetuned on a smaller, high-quality dataset for high-resolution tasks, such as text-to-video and image-to-video generation, demonstrating superiority over existing image-to-video models through human preference studies.

Additionally, the model showcases a strong multi-view prior, serving as a foundation for a multi-view diffusion model that excels in generating multiple consistent views of an object and surpasses specialized novel view synthesis methods. The model's capacity for explicit motion control, either through direct prompts to temporal layers or via training with motion-specific LoRA modules, further underscores its versatility.

To summarize, the study presents three core contributions: a systematic data curation method for transforming large, uncurated video collections into high-quality datasets for video modeling; the training of state-of-the-art text-to-video and image-to-video models that eclipse previous benchmarks; and the exploration of the models' inherent motion and 3D understanding capabilities, suggesting their potential to address the challenge of data scarcity in 3D domains.

- Systematic Data Curation Method
- State-of-the-Art Text-to-Video and Image-to-Video Models
- Exploration of Motion and 3D Understanding Capabilities

### Background

In the background section of the study, the focus shifts to recent advancements in video generation, primarily hinging on the use of diffusion models. These models have garnered attention for their iterative refinement process, which meticulously denoises samples from a normal distribution. This technique has proven its mettle in both high-resolution text-to-image and video synthesis, marking a significant leap in the quality of generated content. The researchers of the study align themselves with this innovative approach by training a latent video diffusion model on a specially curated video dataset. They also provide a concise overview of related work, particularly those employing _latent video diffusion models (Video-LDMs)_, with a more detailed discussion on GANs and autoregressive models relegated to the appendices.

The concept of Latent Video Diffusion Models (Video-LDMs) is pivotal in this study, where the main generative model operates within a latent space to reduce computational load. This approach typically involves enhancing a pretrained text-to-image model with temporal layers to support video synthesis. Following the architecture proposed by Blattmann et al., the study incorporates temporal convolution and attention layers, diverging from approaches that focus solely on temporal layers or omit training entirely. The model directly conditions on text prompts, establishing a foundation for a general motion representation. This versatility allows for fine-tuning towards specific tasks like image-to-video and multi-view synthesis, with novel introductions such as micro-conditioning on frame rate and employing the EDM-framework to adjust noise levels for high-resolution finetuning.

Data curation for pretraining emerges as a critical theme in the study, underscoring the importance of large-scale datasets for developing robust models across various domains, including text-image and language modeling. Despite the success of data curation in generative image modeling, the video generation domain has seen a scarcity of systematic data curation strategies, often resulting in fragmented and ad-hoc approaches. The study points out the limitations of popular datasets like WebVid-10M, which, despite its widespread use, is hampered by issues like watermarks and suboptimal size. The researchers critique the common practice of combining image and video data for joint training, highlighting the challenges it presents in isolating the effects of each data type on model performance.

Addressing these challenges, the study pioneers a systematic exploration of video data curation methods, culminating in a three-stage training strategy. This innovative approach is poised to advance the state-of-the-art in generative video models, showcasing the researchers' commitment to pushing the boundaries of what's possible in video generation technology.

### Methodology

The researchers detail their comprehensive approach to curating data for high-quality video synthesis. They propose a multi-stage strategy to train a video diffusion model on extensive video datasets. This process involves two key components: (i) the implementation of data processing and curation methods, the impact of which will be analyzed in subsequent sections, and (ii) the identification of three distinct training regimes for generative video modeling, which include:

- **Stage I: Image Pretraining** - Leveraging a 2D text-to-image diffusion model to establish a visual foundation.
- **Stage II: Video Pretraining** - Training on large volumes of video data.
- **Stage III: Video Finetuning** - Refining the model with a smaller subset of high-resolution, high-quality videos.

Each regime's significance is evaluated individually in the study.

#### Data Processing and Annotation

![figure2.png](images%2Ffigure2.png)

The researchers collected a vast initial dataset of long videos, which serves as the base for the video pretraining stage. To mitigate the issue of cuts and fades impacting the synthesized videos, a cut detection pipeline was employed. As illustrated in Figure 2, the cut detection pipeline significantly increased the number of usable clips, suggesting the presence of numerous cuts in the raw dataset that weren't reflected in the metadata.

The team then annotated each clip with captions using three methods: an image captioner for the mid-frame of each clip, a video-based captioner, and a summarization of the first two captions generated by a Large Language Model (LLM). This process produced the Large Video Dataset (LVD), comprising 580 million video clips, equating to 212 years of content.

![table1.png](images%2Ftable1.png)

Further scrutiny identified clips within LVD that could potentially hinder the video model's performance due to low motion, text presence, or low aesthetic value. To address this, they utilized dense optical flow calculations to filter out static scenes and optical character recognition to remove clips with excessive text. Additionally, they calculated aesthetics and text-image similarity scores using CLIP embeddings on select frames of each clip. Table 1 presents the dataset statistics, including size and average clip duration.

#### Stage I - Image Pretraining

![figure3.png](images%2Ffigure3.png)
Recognizing image pretraining as a foundational stage, the researchers used a pre-existing image diffusion model, specifically Stable Diffusion 2.1, to provide the initial model with robust visual representation capabilities. To assess the impact of image pretraining, two identical video models were trained—one with pretrained spatial weights and the other without. The models were evaluated through a human preference study, with results indicating a clear preference for the image-pretrained model in terms of quality and adherence to prompts, as depicted in Figure 3a.

#### Stage II - Video Pretraining

The study elaborates on the second stage of their approach, which involves curating a video pretraining dataset. The researchers underscore that for multimodal image modeling, data curation is pivotal for the success of many high-performing models, both discriminative and generative. However, due to the absence of equivalent powerful off-the-shelf video representations for filtering, they turn to human preferences to forge a suitable pretraining dataset. They curate subsets of the Large Video Dataset (LVD) using various methods and use human-preference rankings to gauge the performance of latent video diffusion models trained on these subsets.

Specifically, they filter a randomly sampled 9.8 million subset of LVD, denoted as LVD-10M, by systematically removing the bottom percentages based on several annotation types. These types include CLIP scores, aesthetic scores, OCR detection rates, and synthetic captions, among others. For synthetic captions, instead of filtering, they evaluate the models using Elo rankings. This selective process, applied separately for each annotation type, allows them to identify the most effective filtering threshold and ultimately condense LVD to a final pretraining set of 152 million clips, referred to as LVD-F, as shown in Table 1.

The researchers demonstrate that this meticulous curation enhances the performance of video diffusion models. By applying the curation strategy to LVD-10M and creating a smaller subset (LVD-10M-F), they train a baseline model and compare it with one trained on the uncurated LVD-10M. The curated dataset model consistently outperforms its counterpart in visual quality and prompt alignment, as visualized in Figure 3b.

To further validate the curation method, they compare the LVD-10M-F trained model with models trained on WebVid-10M and InternVid-10M, which are renowned for high aesthetics. Despite LVD-10M-F's smaller size, it is preferred by human evaluators, indicating the effectiveness of the curation process even when dealing with smaller datasets.

Expanding the validation to larger datasets, they train models on both curated and non-curated 50 million clip subsets. Human preference studies indicate that the curated dataset's model is superior, suggesting that the advantages of data curation extend to larger scales. They also find that dataset size plays a crucial role, as a model trained on 50 million curated samples surpasses a model trained on LVD-10M-F with the same training steps, as depicted in Figure 4d.

#### Stage III - Video Finetuning with High-Quality Data

The study discusses the third stage: high-quality finetuning. Here, they finetune three models with different initial weights to assess the impact of the video pretraining stage on final performance. One model starts with pretrained image model weights, skipping video pretraining, while the other two are based on weights from models trained on either the curated or uncurated 50 million clip subsets. After finetuning, they use human rankings to evaluate the models, finding that those with curated pretraining weights consistently rank higher.

The study concludes that the division of video model training into pretraining and finetuning stages is advantageous and that video pretraining is most effective when performed on a large-scale, curated dataset, as the benefits carry through to the finetuning stage.

### Training Video Models at Scale

The study draws on insights from previous sections to showcase the training of video models on a large scale. They start with an optimal data strategy, using these insights to train a robust base model at a resolution of 320 × 576. The subsequent finetuning creates several cutting-edge models tailored for various tasks, such as text-to-video, image-to-video, and frame interpolation. A particularly notable achievement is the fine-tuning of image-to-video models on multi-view generation tasks, where the models demonstrate superior multi-view consistency compared to contemporary models like Zero123XL and SyncDreamer.

#### Pretrained Base Model

Building on the foundation of Stable Diffusion 2.1 (SD 2.1) and incorporating lessons on noise schedule adaptation for higher-resolution images, the researchers fine-tune the noise schedule from a discrete to a continuous one, followed by training the model on the curated LVD-F dataset. The training uses a standard EDM noise schedule over 150k iterations with a considerable batch size. A subsequent finetuning stage aims to generate higher resolution frames, where an increase in the noise level is found to be beneficial. This base model exhibits robust motion representation capabilities, outshining all baselines in zero-shot text-to-video generation tasks.

#### High-Resolution Text-to-Video Model

![figure5.png](images%2Ffigure5.png)

The base model is further fine-tuned on a high-quality video dataset of approximately 1 million samples containing dynamic object motion, steady camera work, and well-aligned captions. After 50k iterations of fine-tuning at a higher resolution and adjusting the noise schedule to accommodate the increased resolution, the model achieves remarkable text-to-video synthesis, as displayed in the samples shown in Figure 5.

#### High-Resolution Image-to-Video Model

![table2.png](images%2Ftable2.png)

![figure6.png](images%2Ffigure6.png)

In addition to text-to-video, the base model is adapted for image-to-video generation, where it takes a static image as input. The text embeddings are replaced with CLIP image embeddings, and a noise-augmented version of the input image is channel-wise concatenated to the input of the UNet. Fine-tuning is performed on models designed for different frame outputs without employing masking techniques. The study observes that the standard classifier-free guidance method can sometimes lead to artifacts, and they discover that a linearly increasing guidance scale across the frame axis mitigates this issue, striking a balance between consistency and oversaturation.

#### LoRAs for Controlled Camera Motion

![figure7.png](images%2Ffigure7.png)

![figure8.png](images%2Ffigure8.png)

The study introduces the use of Low Rank Adaptation (LoRA) to direct controlled camera motion for image-to-video generation. These LoRA parameters are integrated within the temporal attention blocks of the model, and the training is conducted on datasets that are rich with camera motion metadata. The data is segmented into three categories based on the type of camera movement: horizontal, zooming, and static. Samples showcasing the output of models for each type of camera motion, using the same initial frames, are illustrated in Figure 7.

#### Frame Interpolation

![figure9.png](images%2Ffigure9.png)

The paper discusses the advancement of their high-resolution text-to-video model to perform frame interpolation. By following the methodology of Blattmann et al., they concatenate adjacent frames and train the model to interpolate additional frames, thereby increasing the frame rate by a factor of four. Interestingly, the model reaches satisfactory performance with a notably small number of training iterations, around 10k. This process and the resulting samples are further elaborated upon in the appendices.

#### Multi-View Generation

![figure10.png](images%2Ffigure10.png)

The focus shifts to multi-view generation, where the researchers fine-tune their image-to-video SVD model on datasets designed for this purpose. They use the Objaverse, which consists of synthetic 3D objects, and MVImgNet, comprising multi-view videos of common objects. The model trained on Objaverse is conditioned on the elevation angle of the input image, while the MVImgNet-trained model is not pose-conditioned and can generate various camera paths. The finetuned Multi-View model, referred to as SVD-MV, undergoes an ablation study to highlight the significance of the video prior for successful multi-view generation. This model is benchmarked against existing state-of-the-art models using standard metrics like PSNR, LPIPS, and CLIP Similarity scores. Training details and the superior results of the SVD-MV, especially in terms of compute efficiency, are provided.

### Conclusion

The paper summarizes the work on Stable Video Diffusion (SVD), emphasizing its effectiveness in high-resolution text-to-video and image-to-video synthesis. The researchers detail their methodical approach to data curation, the three stages of video model training, and the adaptability of the SVD model to applications such as camera motion control using LoRA. The study underscores that the SVD model acts as a robust 3D prior and achieves state-of-the-art results in multi-view synthesis with significantly less computational effort compared to previous methods. The researchers express their hope that these insights will significantly impact the field of generative video modeling.

## Personal Thoughts

![s-curve-blue-ai-dragon.jpg](images%2Fs-curve-blue-ai-dragon.jpg)

Video generation through AI fascinates me deeply, blending my passion for visual storytelling with cutting-edge technology. It's astonishing to witness how quickly AI has evolved from generating static images to crafting seamless videos, a leap that seemed distant not too long ago.

Standing on the brink of this technological marvel, we're poised for a thrilling exploration into the capabilities of AI in video production. This rapid progression reminds us of the S-curve in technology, where innovations grow slowly at first, then explode in capability and adoption before stabilizing. As we navigate this burgeoning field, it's crucial to stay attuned to the implications and opportunities that lie ahead, ensuring we harness this power responsibly while pushing the boundaries of creativity and expression.