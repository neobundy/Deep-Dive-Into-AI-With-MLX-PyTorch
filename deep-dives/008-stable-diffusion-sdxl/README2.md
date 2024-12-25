# Deep Dive into Stability AI's Generative Models - Stable Diffusion XL Part II

👉 [Part I](README.md) | 👉 [Part II](README2.md)

**📝 Paper**: https://arxiv.org/abs/2307.01952

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., & Rombach, R. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.

![sdxl-paper.png](images%2Fsdxl-paper.png)

In their recent work, Podell et al. introduce SDXL, an enhanced latent diffusion model tailored for the text-to-image synthesis process. This iteration marks a significant evolution from its predecessors within the Stable Diffusion series, primarily through the expansion of the UNet backbone, which is now threefold larger. The augmentation of model parameters can be attributed to an increased number of attention blocks and an expanded cross-attention context, facilitated by the incorporation of a secondary text encoder.

The authors have ingeniously crafted various novel conditioning schemes and have trained SDXL to handle multiple aspect ratios. Additionally, they have developed a refinement model that substantially elevates the visual quality of the images produced by SDXL, employing a post-hoc image-to-image technique.

Their analysis reveals that SDXL significantly outperforms earlier versions of Stable Diffusion, delivering results that are not only competitive but also on par with the black-box models at the forefront of the image generation field. In alignment with the ethos of open scientific inquiry and to enhance transparency in the training and evaluation of large-scale models, Podell et al. have made the code and model weights publicly available, contributing to the collaborative advancement of generative AI research.

## Key Improvements over Previous Models

![paper-figure1.png](images%2Fpaper-figure1.png)

### Left: User Preference Comparison

On the left side, there is a bar chart comparing user preferences between different models:

- **SDXL w/ Refiner**: This model includes an additional refinement step and has the highest preference rate among users, indicated by the tallest bar at 48.44%.
- **SDXL Base**: The base model of SDXL, without the refinement step, comes in second, with a substantial user preference rate of 36.93%.
- **SD 1.5 and SD 2.1**: These are the previous versions of Stable Diffusion, with user preferences significantly lower at 7.91% and 6.71%, respectively, suggesting that the newer SDXL models, particularly with the refiner, are favored for generating more appealing images.

The bar chart demonstrates that the addition of a refinement stage in SDXL markedly boosts its performance, as perceived by users, when compared to the earlier Stable Diffusion models 1.5 and 2.1.

### Right: Two-Stage Pipeline Visualization

On the right, there is a schematic representation of the two-stage pipeline used for image generation with SDXL:

- **Base**: The process starts with a prompt that feeds into the base model, generating an initial latent image of size 128 x 128.
- **Unrefined Latent**: This image is considered 'unrefined' and serves as the intermediate output from the base model.
- **Refiner**: The unrefined latent image is then processed by the refiner model, which applies high-resolution refinement to enhance the visual fidelity.
- **Refined Latent**: The refined latent image is the result of the refiner model's processing, still at a latent size of 128 x 128.
- **VAE Decoder**: The refined latent image is decoded into the final image by a VAE (Variational Autoencoder) decoder.
- **Final Image**: The end product is a high-resolution 1024 x 1024 image, significantly larger and more detailed than the initial latent representation.

The diagram highlights that both SDXL and the refinement model utilize the same autoencoder structure. Additionally, the process includes the application of SDEdit, a technique for editing the latents generated in the first step using the same prompt, which helps in refining the details and overall quality of the final image.

The figure effectively illustrates how SDXL, especially when combined with its refinement model, provides a substantial advancement in generating high-resolution images from textual prompts. The user preference data underscores the model's success, while the pipeline visualization details the technical process that underpins this success.

### Architecture & Scale

![paper-table1.png](images%2Fpaper-table1.png)

The architecture of diffusion models, particularly for image synthesis, has undergone remarkable development since the pioneering studies on _Denoising Diffusion Probabilistic Models_ and _Score-Based Generative Modeling through Stochastic Differential Equations_. The convolutional UNet architecture, a mainstay in this field, has been foundational to these advancements. Successive innovations have integrated mechanisms such as self-attention and cross-attention, pushing the models' architecture to new heights of complexity and operational efficiency.

Within this evolutionary context, SDXL marks a notable departure from the original Stable Diffusion constructs, embracing a strategic allocation of transformer blocks across different levels of the UNet structure. SDXL's design excludes transformer blocks at the highest feature level, selectively employs 2 and 10 transformer blocks at mid-levels, and entirely removes the lowest level of the UNet. This architectural refinement is aimed at optimizing computational efficiency while retaining the robust generative performance of the model.

Referring to Table 1, we can compare the architectural differences between Stable Diffusion versions 1.x, 2.x, and SDXL. SDXL opts for a more powerful text encoder, using a combination of _OpenCLIP ViT-bigG_ with _CLIP ViT-L_, where the outputs from the penultimate layer of the text encoders are concatenated to enhance text conditioning. In addition to text-based cross-attention layers, SDXL also conditions on the pooled text embeddings derived from the OpenCLIP model, in line with strategies used in recent research.

These modifications have culminated in a substantial increase in the model size, with SDXL's UNet comprising 2.6 billion parameters, and the text encoders contributing an additional 817 million parameters. The table illustrates the scaling up from earlier Stable Diffusion models, showcasing SDXL's capacity to synthesize high-resolution images with improved fidelity to the text prompts. This architectural scale-up points to SDXL's potential to set a new benchmark in the landscape of generative models for image synthesis.

### Micro-Conditioning Schemes

#### Conditioning the Model on Image Size

![paper-figure2.png](images%2Fpaper-figure2.png)

In the realm of Latent Diffusion Models (LDMs), a critical challenge has been managing training models across a spectrum of image sizes, necessitated by their inherent two-stage architecture. Traditional methods either discard training images below a certain resolution threshold or upscale smaller images, each with its own drawbacks. As illustrated in Figure 2, following the first approach and setting a minimum resolution threshold for pretraining could result in the loss of a substantial portion of the training data — in the case of SDXL, a significant 39% would be discarded if the cutoff were set at 256 pixels on either dimension. Upscaling, the alternative, often introduces artifacts that could degrade the quality of the generated images.

![paper-figure3.png](images%2Fpaper-figure3.png)

To circumvent these issues, the authors propose a novel approach: conditioning the UNet model on the original resolution of the training images. This method entails embedding the original height and width using Fourier feature encoding and using this embedded vector as additional information for the model during training. This technique, depicted in Figure 3, allows the model to associate specific resolution-dependent image features with the given conditioning size, thus enabling the generation of images that more closely match the desired apparent resolution at inference time.

![paper-table2.png](images%2Fpaper-table2.png)

Table 2 presents a quantitative evaluation of this size-conditioning technique, where three different LDMs are trained and assessed on class-conditional ImageNet at a spatial size of 512x512. The models are:
- **CIN-512-only**: Discards training examples with any dimension below 512 pixels.
- **CIN-nocond**: Utilizes all training examples without size conditioning.
- **CIN-size-cond**: Employs all training examples with size conditioning.

The results, as summarized in Table 2, underscore the benefits of the CIN-size-cond model, which outperforms the baseline models in both Inception Score (IS) and Fréchet Inception Distance (FID) metrics. The poorer performance of CIN-512-only is attributed to overfitting due to a smaller training dataset, while CIN-nocond's lower FID score is linked to the presence of blurry samples in its output distribution. Despite some criticisms of traditional quantitative metrics for foundational DMs, in this context of ImageNet, metrics like FID and IS remain relevant as they are based on neural backbones trained on the ImageNet dataset itself.

#### Conditioning the Model on Cropping Parameters

![paper-figure4.png](images%2Fpaper-figure4.png)

In the pursuit of refining generative models, particularly Stable Diffusion, conditioning on cropping parameters has emerged as a crucial technique to address typical synthesis issues. As shown in the first two rows of Figure 4, earlier versions of Stable Diffusion, SD 1.5 and SD 2.1, often exhibited a failure mode where generated objects would appear cropped, such as the truncated heads of the cats. This is likely a byproduct of random cropping—a common data augmentation method during training which can inadvertently influence the model to replicate such patterns in its outputs.

To remedy these undesirable effects, the authors present a novel conditioning approach. During data loading, crop coordinates (`ctop` and `cleft`) are uniformly sampled, indicating the pixel dimensions to be cropped from the top and left edges of the images. These coordinates are then embedded using Fourier feature encoding and combined into a conditioning parameter that is fed into the model. This crop conditioning, alongside the previously discussed size conditioning, affords a more refined control over the generated images, ensuring that random cropping used during training does not adversely affect the final synthesis.

![paper-figure5.png](images%2Fpaper-figure5.png)

The technique of conditioning augmentation, illustrated in Figure 5, harnesses this approach to manipulate the perceived cropping of images at the inference stage. By adjusting the crop coordinates, one can generate images with varying degrees of focus and object-centric framing. This method, while also addressing the challenges tackled by data bucketing, offers the added benefit of enhancing the image synthesis process by providing direct control over the cropping effect.

The versatility of this approach is notable as it is not confined to Latent Diffusion Models but applicable to a wide range of Diffusion Models. The implementation is straightforward, does not require extensive data preprocessing, and can be dynamically applied during training, making it a practical solution for improving the model's ability to generate well-composed, object-centric images.

### Multi-Aspect Training

The diversity of image dimensions in real-world datasets presents a challenge for text-to-image models, which typically generate square images. However, this square format doesn't necessarily reflect the variety of aspect ratios found in everyday media, such as landscape or portrait formats prevalent on various screens. To address this, a more versatile approach is proposed, which involves finetuning models to handle multiple aspect ratios simultaneously.

![paper-algo1.png](images%2Fpaper-algo1.png)

This process entails categorizing the data into different aspect ratio buckets, aiming to maintain the total pixel count close to the target resolution, usually in the region of 1024x1024 pixels, adjusting height and width in multiples of 64. The details of the aspect ratios used are documented comprehensively in an appendix.

During training, batches are composed of images from the same aspect ratio bucket, and the model alternates between these buckets with each training step. The target size for each batch, corresponding to the bucket size, is fed into the model as a conditioning factor. This size information is encoded into a Fourier space, similar to the previously described size- and crop-conditioning strategies.

In practice, multi-aspect training is applied as a stage of finetuning following an initial pretraining phase at a fixed resolution and aspect ratio. It is then integrated with other conditioning methods through concatenation along the channel axis, as depicted in supplementary materials which include Python code for this procedure. It's important to note that crop-conditioning and multi-aspect training are complementary; crop-conditioning is applied within the constraints of the aspect ratio buckets, typically with a granularity of 64 pixels. For simplicity in implementation, this level of control is maintained even in multi-aspect models.

### Enhanced Autoencoder

In the domain of Latent Diffusion Models (LDMs), the role of the autoencoder is critical as it defines the latent space in which the model operates. The original Stable Diffusion leveraged a pre-trained autoencoder to shape the semantic content of generated images. However, the quality of the final output can be significantly improved by enhancing the autoencoder, particularly in terms of local, high-frequency details.

![paper-table3.png](images%2Fpaper-table3.png)

Building upon this premise, the same architectural framework as the original Stable Diffusion's autoencoder was retrained, but with notable changes to the training regimen. By increasing the batch size substantially from 9 to 256 and implementing weight tracking through an exponential moving average, the updated autoencoder boasts superior performance across all key reconstruction metrics.

As indicated in Table 3, the new autoencoder, used in all subsequent experiments, demonstrates clear advancements over the original. This improved autoencoder forms the backbone of the Stable Diffusion model, contributing to more refined and detailed image synthesis.

### Integrating Enhancements in SDXL

The development of SDXL, a state-of-the-art generative model, involves a multi-tiered training approach that incorporates the improved autoencoder discussed earlier. The model follows a discrete-time diffusion schedule with a significant number of steps to ensure nuanced image generation.

![paper-figure2.png](images%2Fpaper-figure2.png)

Initially, SDXL begins with a pretraining phase on a specially curated dataset, the distribution of which is showcased in Figure 2. This pretraining occurs over a large number of optimization steps at a lower resolution, using the advanced conditioning techniques on image size and cropping parameters detailed in the earlier sections. The model is then progressively trained at a higher resolution and eventually adapts to handle various aspect ratios, maintaining a near 1024x1024 pixel area, thereby accommodating the diverse dimensions seen in real-world images.

![paper-figure6.png](images%2Fpaper-figure6.png)

Despite the robust training, the base SDXL model occasionally produces samples with suboptimal local quality, as evident in Figure 6. To refine these outputs, a separate Latent Diffusion Model (LDM) is trained in the same latent space, focusing on high-resolution data. This refinement model applies a noising-denoising technique, inspired by SDEdit, on the base model's outputs. During inference, this process enhances the detail and quality of the images, particularly for intricate backgrounds and facial features, as illustrated in both Figure 6 and additional samples.

A user study conducted to evaluate the performance of SDXL, both with and without the refinement stage, against older iterations of Stable Diffusion, confirms the superiority of SDXL with the refinement stage. It emerged as the preferred choice, significantly outperforming its predecessors in terms of user preference.

## Navigating Generative AI Model Implementations

🔗 **Stability AI Official Repository**: [Generative Models Repository](https://github.com/Stability-AI/generative-models)

🔗 Automatic1111 Repo: https://github.com/AUTOMATIC1111/stable-diffusion-webui

🔗 ComfyUI Repo: https://github.com/comfyanonymous/ComfyUI

Deploying an advanced model like SDXL requires deep technical expertise, given the complexity inherent in orchestrating base and refiner models. There are numerous open-source initiatives that have facilitated access to these models, with _Automatic1111_ and _ComfyUI_ emerging as prominent options within the community.

Stability AI's official GitHub repository serves as a treasure trove for those looking to delve into the entire suite of Stability Generative AI models, including SDXL. The repository is part of an optimized framework known as SGM, which aims to streamline the implementation process.

Given the extensive nature of these codebases, a comprehensive review would be quite extensive. Therefore, for those looking to apply these models in practical scenarios, it's advisable to familiarize oneself with the overarching structure presented in the official Stability AI repository and then delve deeper into the specific implementation that aligns with one's needs.

Echoing the sentiments from earlier discussions, where we shifted from a custom transformer model to the more pragmatic Hugging Face solutions, it's prudent to leverage existing, well-established implementations like those available through Hugging Face for SDXL or other widely-recognized sources.

Reinventing such a complex 'wheel' may not be the most efficient use of resources, especially when there are expertly crafted, ready-to-use implementations at hand.

As we continue exploring the saga of Stability AI's generative models, our focus will be on understanding the theoretical underpinnings and practical applications of these models, trusting the intricacies of implementation to those who have honed their expertise in this domain.

