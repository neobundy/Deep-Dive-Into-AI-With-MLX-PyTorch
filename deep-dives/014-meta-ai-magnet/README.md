# Deep Dive into MetaAI MAGNeT - Masked Audio Generation using a Single Non-Autoregressive Transformer

![magnet-title.png](images%2Fmagnet-title.png)

In our examination of the study on MAGNET, the researchers introduce a single-stage non-autoregressive transformer for masked generative sequence modeling across multiple audio token streams. Unlike previous methods, MAGNET innovates with a novel rescoring approach using an external pre-trained model to enhance audio generation quality and a hybrid model that combines autoregressive and non-autoregressive techniques for improved efficiency. They demonstrate MAGNET's effectiveness in text-to-music and text-to-audio generation tasks through extensive empirical evaluations, showcasing its speed and performance compared to existing autoregressive models. The researchers' approach offers significant advancements in audio generation technology, emphasizing speed, efficiency, and quality.

**Autoregressive vs. Non-autoregressive Transformer:**
- **Autoregressive (AR) transformers** generate sequences sequentially, predicting each token based on the previously generated tokens. This method ensures high coherence but can be slow due to its sequential nature.
- **Non-autoregressive (NAR) transformers** generate all parts of the sequence in parallel, removing the dependency on previous tokens. This parallelism significantly speeds up generation but can sometimes result in less coherence between sequence parts.

**Masked Generative Sequence Modeling:**
This technique involves hiding (masking) parts of a sequence and then training a model to predict the masked portions based on the context provided by the unmasked parts. It's a form of self-supervised learning that enables models to understand and generate sequences, whether text, audio, or another data type, by learning to fill in the gaps.

## In Simple Terms

Imagine you're a musician, and instead of playing an instrument, you describe the music or sounds you want to hear, and a computer generates it for you in real-time. MAGNET does something similar for audio generation. It's like a highly advanced and speedy translator that takes text descriptions (like "a soothing piano melody" or "the sound of rain") and turns them into actual audio clips.

The researchers have developed two key versions of this model:

1. **Autoregressive (AR) models**: These are like meticulous artists, drawing out a picture stroke by stroke, or in this case, generating sound bit by bit, based on what was created just before. They're precise but tend to work slowly because each new piece of audio depends on completing the previous one.
2. **Non-autoregressive (NAR) models**: These are more like spontaneous painters, throwing paint across the canvas all at once. MAGNET can generate parts of the audio simultaneously, making it much faster than its AR counterparts. However, ensuring that the final piece is cohesive and maintains high quality is a challenge.

MAGNET, particularly in its NAR form, shines by generating high-quality audio much faster than traditional methods, making it suitable for applications needing quick responses, such as interactive music creation tools or real-time audio effects in gaming.

Moreover, the researchers introduced a hybrid approach, blending the meticulous nature of AR models with the swift capabilities of NAR models. This method allows for a balance between quality and speed, offering an innovative solution to the challenges of audio generation.

In essence, MAGNET represents a significant leap forward in how machines can understand textual descriptions and create corresponding audio, pushing the boundaries of creative and real-time audio applications.

## Deep Dive - Masked Audio Generation using a Single Non-Autoregressive Transformer

Alon Ziv, Itai Gat, Gael Le Lan, et al. (2024) Masked Audio Generation using a Single Non-Autoregressive Transformer 
🔗 https://arxiv.org/abs/2401.04577

The researchers begin by acknowledging the significant strides made in self-supervised representation learning, sequence modeling, and audio synthesis, which collectively enable a leap in performance for high-quality conditional audio generation. The study critiques the prevalent methods of audio representation, dividing them into two main categories: autoregressive models, typically operating on discrete audio representations, and diffusion-based models, usually working on continuous latent representations. Both approaches, while yielding impressive results, face their own set of challenges, such as high latency in autoregressive models and the difficulty of generating long-form sequences in diffusion models.

🧐 _In our exploration, it's important to note that "Stable Audio," which represents the twelfth deep dive in this context, is fundamentally rooted in diffusion models specifically tailored for audio generation. This detail underscores a key aspect of the ongoing dialogue within the study regarding the evolution and application of diffusion models in the realm of audio synthesis, highlighting their significance in the broader landscape of audio generation technologies._ 

The researchers introduce MAGNET as a pioneering solution, a non-autoregressive transformer model designed for masked generative sequence modeling across multi-stream audio signals. The approach involves a novel training regime that masks and predicts spans of input tokens, conditioned on the unmasked ones, and introduces a rescoring method leveraging an external pre-trained model to improve audio quality. Additionally, they explore a hybrid model that merges autoregressive and non-autoregressive methodologies to optimize the generation process further.

The study critically evaluates MAGNET against existing methods, highlighting its efficiency in reducing latency significantly while maintaining comparable quality, a crucial advancement for real-time applications like music generation and editing. Through comprehensive objective metrics and human studies, the researchers demonstrate MAGNET's capacity to achieve comparable results to traditional baselines, while being substantially faster, thus setting a new benchmark in the field of audio generation.

Our examination will further dissect these methodologies, assess the trade-offs between autoregressive and non-autoregressive models, and explore the broader implications of MAGNET's introduction to the domain of audio synthesis. Through this, we aim to grasp the nuanced advancements MAGNET brings to the table, not just in terms of technical efficiency but also in its potential to revolutionize audio generation workflows.

## Background

![pre-paper-figure1.png](images%2Fpre-paper-figure1.png)

Alexandre Défossez, et al. (2022). High Fidelity Neural Audio Compression

🔗 https://arxiv.org/abs/2210.13438

🧐 _The prerequisite paper introduces EnCodec, a state-of-the-art real-time high-fidelity neural audio codec designed for efficient streaming. It features an encoder-decoder architecture with quantized latent space, trained end-to-end. The model employs multiscale spectrogram adversaries to reduce artifacts and produce high-quality audio, introduces a novel loss balancer for stable training, and leverages lightweight Transformer models to further compress representations by up to 40%. Extensive subjective evaluations demonstrate its superiority over baseline methods across various settings, including monophonic and stereophonic audio. The paper also discusses key design choices, training objectives, and the potential for further bandwidth reduction with entropy coding._

The researchers delve into the evolution of audio generative models, focusing on the transition from traditional audio representation methods to modern approaches that utilize latent representations derived from compression models. Notably, the study references the work of Defossez et al. (2022), who introduced EnCodec, a convolutional auto-encoder that employs Residual Vector Quantization (RVQ) alongside an adversarial reconstruction loss. This method encodes an audio signal into a tensor, which is then quantized into discrete tokens across multiple parallel streams, highlighting the nuanced process of audio signal representation.

The core ambition of audio generative modeling, as outlined by the researchers, is to model the conditional joint probability distribution `p_θ(z|y)`, where `z` represents the discrete audio signal and `y` the semantic condition. The autoregressive (AR) framework, traditionally favored for its sequential generation capabilities, calculates the joint probability of a sequence as the product of its conditional probabilities, as shown in Equation (1):

![formula1.png](images%2Fformula1.png)

To facilitate sequential prediction while masking future tokens, a masking function `m(i)` is employed, leading to Equation (2):

![formula2.png](images%2Fformula2.png)

where:

![exp1.png](images%2Fexp1.png)


Transitioning to non-autoregressive (NAR) models, the researchers propose modifying the masking strategy to predict an arbitrary subset of tokens based on the unmasked ones across multiple decoding steps, encapsulated in Equation (3):

![formula3.png](images%2Fformula3.png)

Here, 

![exp2.png](images%2Fexp2.png)

, illustrating a strategic approach to mask and predict tokens across the sequence.

The modified version of Equation (2) for the NAR setup is presented in Equation (4):

![formula4.png](images%2Fformula4.png)

This background elucidates the researchers' methodological pivot towards a non-autoregressive approach for audio generation, underscoring the innovative aspects of their work in enhancing efficiency and prediction capabilities in audio synthesis.

## Methodology

![figure1.png](images%2Ffigure1.png)

They identify three critical areas for enhancing audio quality: a refined masking strategy, a restricted context for token generation, and a rescoring mechanism to improve audio quality. This comprehensive approach aims to tackle the inherent challenges in generating high-quality audio by addressing the interaction between tokens, the local nature of the temporal context in audio codebooks, and the diversity required at different decoding steps.

The masking strategy introduced by the researchers is particularly innovative, focusing on spans of tokens rather than individual tokens to better capture the shared information among adjacent audio tokens. They experimented with various span lengths and determined that a 60ms span length offers optimal performance. This strategy is encapsulated in their approach to dynamically calculate the number of spans to mask based on the masking rate, ensuring a balanced distribution of masked spans across the sequence.

In addressing the restricted context, the researchers leveraged the inherent structure of RVQ-based tokenization, which encodes quantization errors in a hierarchical manner. By analyzing the receptive field of the audio encoder used, they propose limiting the self-attention mechanism to tokens within a ∼200ms temporal distance, enhancing the model's focus and efficiency during training.

For model inference, the study introduces a rescoring strategy that combines predictions from MAGNET with those from an external model, leading to the equation:

![formula5.png](images%2Fformula5.png)

This method allows for a nuanced balance between the generative model's predictions and external insights to refine the audio quality further.

Additionally, the researchers incorporate Classifier-Free Guidance (CFG) annealing to adjust the guidance coefficient based on the masking rate, gradually shifting the model's focus from textual adherence to contextual infilling. This process is mathematically represented as:

![formula6.png](images%2Fformula6.png)

where `λ_0` and `λ_1` are the initial and final guidance coefficients, respectively. This approach, they argue, allows for a more natural and high-quality audio generation process.

Through these methodological innovations, the researchers aim to significantly enhance the quality of generated audio while maintaining the efficiency and speed advantages of non-autoregressive models. This detailed exploration of MAGNET's methodological underpinnings showcases the study's contributions to advancing audio synthesis technology.

## Experimental Setup

![table1.png](images%2Ftable1.png)

The researchers meticulously detail the methodology employed to evaluate MAGNET, focusing on text-to-music and text-to-audio generation. Utilizing the same datasets as Copet et al. (2023) for music generation and Kreuk et al. (2022a) for audio generation, they provide a comprehensive overview of the training data and the baseline datasets used for comparison.

The study elaborates on the use of the official EnCodec model for converting audio segments into a discrete representation, highlighting the implementation specifics, including the usage of four codebooks and the preprocessing of text to extract semantic representations using a pre-trained T5 model. This rigorous approach ensures consistency and comparability in their evaluation.

For training, the researchers utilized non-autoregressive transformer models with 300M (MAGNET-small) and 1.5B (MAGNET-large) parameters, emphasizing the scalability of their method. They detail the training regimen, including the use of 30-second audio crops, the AdamW optimizer, and a carefully designed learning rate schedule, ensuring a robust training process.

In terms of evaluation, the study employs both objective and subjective metrics, including the Frechet Audio Distance (FAD), the Kullback-Leibler Divergence (KL), and the CLAP score, to measure the effectiveness of MAGNET in generating audio that aligns with text descriptions. Furthermore, human studies were conducted to assess overall quality and relevance to the text input, providing valuable insights into the perceptual quality of the generated audio.

This detailed experimental setup, as outlined by the researchers, underscores the thoroughness of their approach in assessing the capabilities of MAGNET. By adhering to rigorous standards and employing a combination of objective metrics and human evaluation, the study provides a comprehensive assessment of MAGNET's performance in generating high-quality audio from text descriptions.

🧐 **Frechet Audio Distance (FAD):**
_FAD is an adaptation of the Frechet Inception Distance (FID), a metric originally used for evaluating the quality of images generated by GANs, to the audio domain. It measures the distance between the feature distributions of real and generated audio samples. These features are typically extracted using a pre-trained deep learning model designed for audio processing. A lower FAD indicates that the generated audio is closer to the real audio in terms of the distributions of features extracted, suggesting higher quality and realism in the generated audio._

🧐 **Kullback-Leibler Divergence (KL):**
_The KL divergence is a measure from information theory that quantifies how one probability distribution diverges from a second, expected probability distribution. In the context of audio generation, it can be used to compare the distribution of labels or features of generated audio against those of real audio. A lower KL divergence indicates that the generated audio's characteristics are more similar to those of the real audio, suggesting a higher accuracy in mimicking the real audio distribution._

🧐 **CLAP Score:**
_The CLAP score (Contrastive Language-Audio Pretraining) measures the alignment between audio content and textual descriptions. It leverages a model pre-trained to understand the relationship between sounds and their corresponding textual descriptions. A higher CLAP score indicates a better match between the generated audio and its intended description, suggesting that the audio generation model successfully captures and renders the semantic nuances indicated by the text._

## Results

The researchers delve into a comprehensive evaluation of MAGNET against several key benchmarks in text-to-music and text-to-audio generation, notably comparing it with models like Mousai, MusicGen, AudioLDM2, and MusicLM. They highlight MAGNET's ability to achieve competitive performance while significantly reducing latency, a critical advancement for real-time applications.

![figure2.png](images%2Ffigure2.png)

The study presents an in-depth analysis of latency versus throughput, revealing that MAGNET excels in scenarios requiring low latency, particularly for smaller batch sizes. This finding underscores the model's suitability for interactive applications, where it demonstrates a latency up to 10 times lower than that of MUSICGEN. Additionally, they introduce a hybrid version of MAGNET, which combines autoregressive and non-autoregressive decoding strategies. This hybrid approach allows the model to generate a short audio prompt autoregressively, then complete the generation faster through non-autoregressive decoding, offering a novel way to balance quality and latency.

![table2-3.png](images%2Ftable2-3.png)

Further, the researchers conduct ablation studies to investigate the effects of span masking, temporal context restriction, CFG annealing, model rescoring, and the number of decoding steps on model performance. These studies confirm the importance of these components in enhancing the model's efficiency and audio generation quality. Notably, they find that using a span length of 3 (equivalent to 60ms) and applying model rescoring generally improves performance across almost all metrics, albeit with a trade-off in inference speed.

Lastly, they explore the impact of reducing decoding steps on latency and performance, discovering that minimizing steps for higher codebook levels does not significantly affect quality, offering a strategy for further reducing latency without substantially compromising audio quality. The iterative decoding process of MAGNET is also visualized, illustrating its non-causal approach to audio sequence generation, which progressively fills in the gaps until a complete sequence is formed.

## Related Work

The researchers meticulously outline the advancements and methodologies prevalent in the domain of audio generation, categorizing them into autoregressive and non-autoregressive approaches, as well as masked generative modeling.

**Autoregressive audio generation** has been a focal point, with significant efforts dedicated to generating both environmental sounds and music. This category includes the use of transformer models applied over discrete audio representations, leveraging hierarchical encoders for music samples, and exploring music generation in relation to specific videos. Additionally, some approaches have integrated multi-stream representations of music, employing cascades of transformer decoders for enhanced generation capabilities.

In contrast, **non-autoregressive audio generation** primarily employs diffusion models, appreciated for their versatility across continuous and discrete representations. This approach has been adapted for a variety of applications, including text-to-audio tasks, extending its utility to inpainting and image-to-audio conversions, and even text-to-music generation. These models have been instrumental in generating audio by fine-tuning on specific representations or through the application of latent interpolation techniques to produce longer sequences.

**Masked generative modeling** emerges as another crucial area, highlighting its application across diverse domains, from machine translation to image synthesis. This technique involves parallel decoding strategies that have been adapted for text-to-speech and dialogue synthesis, distinguishing itself by the conditional use of semantic tokens derived from models. The discussion also touches upon concurrent efforts in non-autoregressive music generation, differentiating in terms of model architecture and the scope of generation tasks.

This overview encapsulates the breadth of research and development within audio generation, showcasing the diversity of approaches and the evolution of methodologies that have shaped the field. Through this lens, the unique positioning and contributions of the researchers' work in advancing audio synthesis technology are clearly articulated, setting the stage for their novel contributions to the domain.

## Conclusion

The researchers address the limitations and achievements of their non-autoregressive architecture, MAGNET, specifically tailored for scenarios demanding low latency. They acknowledge a design choice where the model re-encodes the entire sequence at each decoding step, contrasting with autoregressive models that benefit from caching mechanisms to improve efficiency. This difference highlights a potential area for further research to enhance non-autoregressive architectures by adopting similar caching strategies for unchanged time steps.

The study concludes with the introduction of MAGNET, a pioneering pure non-autoregressive method for text-conditioned audio generation, achieving competitive performance with autoregressive methods while being approximately 7 times faster. The researchers also explore a hybrid model, combining autoregressive and non-autoregressive techniques to leverage the strengths of both. Through objective metrics and human studies, MAGNET's capability for real-time audio generation with minimal quality degradation is emphasized.

Looking ahead, the researchers are keen on extending their work on model rescoring and advanced inference methods. They believe that incorporating external scoring models could further improve the decoding process, particularly in non-left-to-right model decoding, highlighting the potential for significant advancements in audio generation technologies.

## Personal Notes

Diving into this paper, and indeed any AI-centric literature today, reinforces a sense of déjà vu regarding the plethora of concepts encountered. Terms like autoregressive vs. non-autoregressive models, masked vs. unmasked techniques, scoring mechanisms, and metrics like Frechet Audio Distance, Kullback-Leibler divergence, and the CLAP Score might feel increasingly familiar. The CLAP Score, drawing parallels from CLIP, embodies a polymorphic concept that itself stems from the broader theory of contrastive learning.

This iterative exposure underscores a critical strategy in knowledge acquisition, akin to object-oriented learning: identifying and understanding core concepts as abstract "ancestor" ideas. By framing new information as extensions or implementations of these foundational principles, you simplify the learning process, focusing on the novel or domain-specific nuances without being bogged down by the fundamentals each time.

Efficiency in assimilating knowledge from academic papers should ideally improve over time, with each paper demanding less cognitive effort to comprehend. If that's not the case, it might be worthwhile to reassess your learning strategy to ensure you're building upon a solid base of core concepts, rather than approaching each paper as an isolated learning event.

On a comparative note, the contrasting methodologies in audio/music generation between Meta AI and Stability AI, particularly regarding Meta's multifaceted approach versus Stability's focus on diffusion models, highlight an evolutionary battleground in AI development. As these techniques evolve and refine, we might anticipate a sort of "natural selection" in AI methodologies, where the most efficient and robust approaches emerge predominant. This evolutionary perspective not only enriches our understanding of the field's trajectory but also primes us for future innovations and their potential impact.

## Testing Models

🤗 _Since I'm currently away from my usual setups, I'm unable to run the models with CUDA GPUs. I plan to try them out once I return home with my Windows RTX 4090 setup. In the meantime, you have the option to test them on your local machine or explore samples available on the website._

🔗 https://pages.cs.huji.ac.il/adiyoss-lab/MAGNeT/

To run MAGNET locally using the Audiocraft library, follow these steps:

1. **Install the Audiocraft Library**: First, you need to install the Audiocraft library from its GitHub repository. Open your terminal and execute the following command:

```sh
pip install git+https://github.com/facebookresearch/audiocraft.git
```

This command clones the Audiocraft repository and installs it on your system.

2. **Install FFmpeg**: MAGNET requires FFmpeg for processing audio files. Ensure that FFmpeg is installed on your system by running:

```sh
apt-get install ffmpeg
```

This command installs FFmpeg, a crucial tool for handling multimedia data, which MAGNET relies on for its operations.

3. **Run MAGNET with Python**: 

Music generation:

```python
from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

descriptions = ["80s thrash metal intro riff", "lovely piano sonata"]
model.set_generation_params(use_sampling=True, top_k=0, top_p=0.9, temperature=2.0, max_cfg_coef=10.0, min_cfg_coef=1.0, decoding_steps=[20, 10, 10, 10], span_arrangement='nonoverlap')
wav = model.generate(descriptions)  # generates 2 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")
```

Sound Effects generation:

```python
from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

model = MAGNeT.get_pretrained("facebook/audio-magnet-medium")

descriptions = ["gunfight under heavy rain", "explosion in a cave", "alien spaceship landing"]
model.set_generation_params(use_sampling=True, top_k=0, top_p=0.9, temperature=2.0, max_cfg_coef=10.0, min_cfg_coef=1.0, decoding_steps=[20, 10, 10, 10], span_arrangement='nonoverlap')
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")
```

The models are automatically downloaded from the Hugging Face model hub and used to generate audio samples based on the given textual descriptions. The generated audio samples are then saved as .wav files for further analysis and evaluation. 

1. **facebook/magnet-small-10secs**: A compact 300M parameter model designed for generating 10-second music samples.
2. **facebook/magnet-medium-10secs**: A more robust 1.5B parameter model, also focused on 10-second music samples, offering enhanced quality.
3. **facebook/magnet-small-30secs**: Similar to its 10-second counterpart but configured to produce 30-second music samples, allowing for more extended compositions.
4. **facebook/magnet-medium-30secs**: The larger 1.5B parameter variant tailored for 30-second music samples, ideal for generating more complex musical pieces.
5. **facebook/audio-magnet-small**: A 300M parameter model dedicated to creating sound effects from text descriptions, offering a wide range of audio generation possibilities.
6. **facebook/audio-magnet-medium**: The larger 1.5B version for text-to-sound-effect generation, providing higher fidelity and more nuanced soundscapes.

To effectively run MAGNET on your local machine, ensure you have a GPU with at least 16GB of memory, particularly for the medium-sized models. This hardware requirement is crucial for handling the computational demands of the models and facilitating smooth operation, especially for tasks involving the generation of high-quality audio content.

For more info, visit the official repo:

🔗 https://github.com/facebookresearch/audiocraft/blob/main/docs/MAGNET.md

🧐 _Loudness normalization at -14 dB LUFS refers to the process of adjusting the audio's average loudness to a target level of -14 LUFS (Loudness Units Full Scale). LUFS is a standard measurement used to gauge perceived loudness, and it's widely adopted in broadcasting and music streaming services to ensure a consistent listening experience across different programs and tracks._

_The specification of "-14 dB LUFS" is particularly significant in the context of streaming platforms like Spotify, YouTube, and Apple Music, which recommend or use this standard to normalize the loudness of the content on their services. This means that if your audio is mastered at a level higher than -14 dB LUFS, these platforms will automatically reduce its loudness to match this standard, and vice versa for quieter tracks._

_Normalizing audio to -14 dB LUFS ensures that your content maintains a consistent and comfortable listening level for the audience, without the need for manual volume adjustments between tracks or videos. This practice enhances the overall user experience, promoting a more enjoyable and less disruptive consumption of audio content._

_Indeed, the concept of normalization, frequently encountered across various discussions in this repository, plays a pivotal role in audio processing as well. This technique ensures uniformity in audio output levels, enhancing the overall listening experience by maintaining consistent loudness across different audio tracks or segments._