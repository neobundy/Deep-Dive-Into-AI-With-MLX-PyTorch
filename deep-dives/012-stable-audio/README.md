# Deep Dive into Stability AI's Generative Models - Stable Audio

![sd-audio-title.png](images%2Fsd-audio-title.png)

In our final analysis of Stability AI's Generative Models, we're examining the innovative introduction provided by the developers at Stability AI on their latest contribution to the generative AI landscape, particularly focusing on audio generation. The work titled "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" marks a significant advancement in the realm of diffusion-based generative models, which have been pivotal in transforming the quality and controllability of generated media, including images, video, and notably, audio.

🧐 _Stable Audio had already made its debut before the paper was officially published, a release that coincided with my work on Stable Video Diffusion. Now, with the paper freshly available, it presents an exciting opportunity for us to delve into the intricate details of this groundbreaking work. The timing couldn't be more opportune, and I'm eager to explore the nuances and insights that the paper offers on this innovative audio diffusion model. This moment marks a pivotal point in our journey through the latest advancements in generative AI, and I'm thrilled to embark on this analysis._

![diagram-tech-details.png](images%2Fdiagram-tech-details.png)

The developers highlight the challenges inherent in generating audio with diffusion models, primarily due to the models' training on fixed-size outputs which complicates the generation of audio of varying lengths, such as full songs. This limitation is addressed through "Stable Audio," a latent diffusion model architecture uniquely conditioned on text metadata as well as audio file duration and start time. This approach allows for unprecedented control over the content and length of the generated audio, enabling the generation of audio of a specified length up to the training window size.

A standout feature of Stable Audio is its efficiency, boasting the capability to render 95 seconds of high-quality stereo audio in less than one second on high-performance GPUs, a feat enabled by working with a heavily downsampled latent representation of audio and leveraging the latest advancements in diffusion sampling techniques.

Delving into the technical specifics, Stable Audio employs a variational autoencoder (VAE) for audio compression, a text encoder for text conditioning, and a U-Net-based conditioned diffusion model for generating the audio. The use of a CLAP model's frozen text encoder and the incorporation of discrete learned timing embeddings for conditioning the model on audio length are particularly innovative, allowing for nuanced control over the generated audio's timing.

Training the flagship Stable Audio model involved a substantial dataset of over 800,000 audio files, underscoring the model's robustness and the diversity of its training data. The developers' commitment to ongoing research and the promise of future open-source models and training code from Harmonai, Stability AI's generative audio research lab, hint at the exciting potential for community engagement and further innovations in audio generation technology.

Through this lens, we aim to dissect the methodologies employed, the technological breakthroughs achieved, and the potential implications of Stable Audio in the broader context of generative AI and audio synthesis. This work not only showcases a pivotal step forward in audio generation but also sets the stage for future advancements in the field.

## In Simple Terms

The researchers at Stability AI have developed a new method called Stable Audio that can quickly create a wide variety of sounds or music based on descriptions you give it. Imagine telling a computer, "I want a headbanging heavy metal track that lasts for 30 seconds," and it whips up an electrifying riff complete with shredding guitars and pounding drums for you in the blink of an eye. That's the kind of thing Stable Audio can do. This is possible because they've made improvements to a kind of AI called a diffusion model, which is good at generating new content after learning from lots of examples.

Stable Audio is special because it can make sounds that are high-quality and can last as long or as short as you need them to—up to about a minute and a half. The sound it makes can be as detailed and rich as what you'd hear in real life, which is a big step up from other systems that might only make simpler sounds or have to take a long time to do it.

The paper also talks about how they trained Stable Audio to understand what kind of sound you want. They did this by using examples that come with descriptions, almost like teaching it how different words and phrases relate to different types of sounds.

They tested Stable Audio to make sure it works well. They compared it to other AI systems by seeing how closely the sounds it made matched what was asked for, how nice the sounds were to listen to, and whether the sounds had the right feel and structure that music usually has.

Lastly, the paper discusses the importance of using Stable Audio responsibly, acknowledging that there's always a risk that the AI could pick up and repeat biases from the examples it learned from, which could be unfair to certain cultures or groups of people. They emphasize the need to keep working on the technology carefully and to collaborate with others in the field to make sure it's used in the best way possible.

## Deep Dive - Fast Timing-Conditioned Latent Audio Diffusion

Zach Evans, CJ Carr, Josiah Taylor, Scott H. Hawley, Jordi Pons. (2024). Fast Timing-Conditioned Latent Audio Diffusion

🔗 https://arxiv.org/abs/2402.04825

Generating long-form 44.1kHz stereo audio from text prompts poses significant computational challenges, particularly as most prior work overlooks the natural variation in duration of music and sound effects. This research introduces an efficient method for generating long-form, variable-length stereo music and sounds at 44.1kHz, directed by text prompts through a generative model. Stable Audio employs latent diffusion, anchored by a fully-convolutional variational autoencoder, to establish its latent framework. It incorporates conditioning on both text prompts and timing embeddings, enabling precise control over the content and length of the output. Capable of rendering up to 95 seconds of stereo audio at 44.1kHz in just 8 seconds on an A100 GPU, Stable Audio not only showcases computational efficiency and rapid inference capabilities but also stands out in two public text-to-music and audio benchmarks. Unlike contemporary state-of-the-art models, it has the unique ability to generate structured music and stereo sounds, marking a significant advancement in the field of generative audio.

### Introduction

The advent of diffusion-based generative models has significantly enhanced the quality and controllability of generated images, videos, and audio, marking a transformative period in generative AI research. These models, however, face notable challenges, particularly in computational efficiency during training and inference phases, especially when operating within the raw signal domain. A notable advancement to address this issue has been the development of latent diffusion models, which operate within the latent space of a pre-trained autoencoder, offering a more compute-efficient alternative. These models enable the generation of long-form audio by utilizing a downsampled latent representation, significantly reducing inference times compared to working with raw audio signals.

Another pressing challenge in audio generation involves the typical training of diffusion models on fixed-size outputs, which complicates the generation of audio in varying lengths, such as full songs or sound effects. To mitigate this, audio diffusion models often rely on training with randomly cropped or padded segments from longer audio files. This approach, however, may result in generating arbitrary sections of music that lack coherence in their musical phrasing.

Stable Audio emerges as a solution to these challenges, employing a latent diffusion model architecture that is conditioned on both text prompts and timing embeddings. This innovative approach allows for precise control over the content and length of the generated music and sound effects, enabling the generation of audio of specific, variable lengths up to the maximum length accommodated by the training window. Thanks to the computational efficiency of latent diffusion modeling, Stable Audio is capable of rendering long-form content swiftly, producing up to 95 seconds of stereo audio at a 44.1kHz sampling rate in just 8 seconds on high-performance computing hardware.

To evaluate the generated audio, this research proposes new metrics tailored for long-form, full-band stereo signals, including a Fréchet Distance based on OpenL3 embeddings for assessing the plausibility of generated signals, a Kullback-Leibler divergence for evaluating semantic correspondence, and a CLAP score for adherence to text prompts. Additionally, a qualitative study was conducted to assess audio quality, text alignment, musicality, stereo correctness, and musical structure, demonstrating that Stable Audio achieves state-of-the-art results in generating long-form full-band stereo music and sound effects from text and timing inputs. Unlike prior works, Stable Audio also excels in generating music with a structured composition, including introductions, developments, and outros, as well as stereo sound effects, showcasing its unparalleled capability in the realm of generative audio technology.

### Background

The exploration of generative models in audio synthesis by the researchers has covered a broad spectrum of methodologies, ranging from autoregressive models to the most recent advancements in diffusion technologies.

**Autoregressive models**, such as WaveNet and its successors, have significantly contributed to the advancement of high-quality audio generation through the modeling of quantized audio samples or latents. These models, including notable names like Jukebox, MusicLM, and MusicGen, have exhibited exceptional capabilities, particularly when conditioned on text prompts. However, they are hindered by slow inference times due to their inherently sequential processing nature. In contrast, the study diverges from autoregressive modeling, aiming for efficiency and innovation in audio generation.

**Non-autoregressive models** were developed as a solution to the computational inefficiencies inherent in autoregressive modeling. Innovations like Parallel WaveNet and various adversarial strategies have facilitated faster synthesis by parallelizing the generation process. Despite progress with non-autoregressive models such as VampNet, StemGen, and MAGNeT, the research undertaken by the team does not follow these methodologies. Instead, it emphasizes diffusion-based models for their optimal balance of quality and efficiency.

**End-to-end diffusion models** have shown promising potential for both unconditional and conditional audio synthesis across a variety of domains. From CRASH’s drum synthesis to Noise2Music’s text-conditional music generation, these models have broadened the scope of audio generation capabilities. The approach adopted by the researchers, while grounded in diffusion principles, prefers latent diffusion for leveraging computational efficiency without compromising on the generation quality.

**Spectrogram diffusion models** and **latent diffusion models** have each made distinct contributions to the field. Techniques such as Riffusion and CQT-Diff have demonstrated the viability of spectrogram-based synthesis, while Moûsai and AudioLDM have exploited latent diffusion for text-to-music and audio generation. The study extends these latent diffusion methodologies by incorporating a variational autoencoder for normalizing latents, thereby enhancing both the efficiency and quality of audio generation.

In tackling the challenge of generating **high sampling rate** and **stereo audio**, the researchers aim to bridge the gap in producing commercial-quality music and sounds at 44.1kHz stereo, a goal not fully addressed by previous models. Furthermore, the study explores the application of **text embeddings** from CLAP and T5-like models, thereby enriching the model's generative capabilities through contrastive language-audio pretraining.

For generating **variable-length**, **long-form audio**, the research leverages latent diffusion to surpass the constraints faced by earlier models limited to short audio clips. This effort is further supported by an innovative use of **timing conditioning**, a strategy not previously explored within diffusion models, allowing for precise control over the length of the generated audio.

**Evaluation metrics** alongside generative models have predominantly focused on short-form mono audio. This work introduces novel quantitative and qualitative metrics specifically designed for evaluating long-form, full-band stereo audio, effectively addressing a significant gap in the assessment of generative audio quality, text alignment, and musicality.

Lastly, the notion of **multitask generative modeling** has demonstrated its potential in addressing the synthesis of speech, music, and sounds concurrently. The model developed in this study distinctively engages in the generation of music and sound effects from text prompts, illustrating the adaptability and potential of latent diffusion in the realm of generative audio synthesis.

### Architecture

Stable Audio leverages a latent diffusion model architecture, ingeniously integrating a variational autoencoder (VAE), a conditioning signal, and a diffusion model to pioneer efficient and high-quality audio generation.

#### Variational Autoencoder (VAE)

At the core of Stable Audio's efficiency in generating 44.1kHz stereo audio is its variational autoencoder (VAE), a mechanism that compresses audio into a compact, invertible latent encoding. This encoding significantly accelerates both the generation and training processes compared to the traditional approach of working directly with raw audio samples. To accommodate audio of any length, Stable Audio employs a fully-convolutional architecture, boasting 133 million parameters, inspired by the Descript Audio Codec's encoder and decoder design, albeit excluding the quantizer. This choice is pivotal for achieving high-fidelity audio reconstruction at substantial compression ratios, outperforming other models such as EnCodec. Notably, the inclusion of Snake activations within this architecture further enhances audio quality at high compression rates, albeit at the cost of increased VRAM usage. The VAE, developed specifically for this project, effectively reduces the input stereo audio sequence by a factor of 1024, resulting in a latent sequence with a channel dimension of 64. This transformation compresses a 2 × L input into a 64 × L/1024 latent representation, achieving an overall data compression ratio of 32, thereby streamlining the audio generation process while preserving quality.

#### Conditioning

The conditioning framework of Stable Audio is a two-pronged approach, encompassing both text and timing elements to guide the generation of audio.

**Text Encoder**
To infuse the generative process with textual prompts, Stable Audio utilizes a CLAP text encoder meticulously trained from scratch on the research team's dataset. The model adopts the architecture recommended by CLAP: a HTSAT-based audio encoder coupled with a RoBERTa-based text encoder, together accounting for 141 million parameters, trained with a contrastive loss that bridges language and audio. This choice of CLAP embeddings over T5 embeddings is strategic, leveraging CLAP's multimodal design to capture nuances in the relationship between linguistic elements and corresponding audio features. Empirical findings, further corroborate the superiority of CLAP embeddings trained on the specific dataset over their open-source counterparts. Stable Audio, drawing inspiration from NovelAI's use of CLIP text features for Stable Diffusion, harnesses the penultimate layer of the text encoder to provide a more potent conditioning signal than the final layer's output. These text features are adeptly integrated into the diffusion U-Net via cross-attention layers, refining the generation process.

![figure1-2.png](images%2Ffigure1-2.png)

**Timing Embeddings**
The timing aspect of conditioning is operationalized by computing two attributes for each segment of the training audio: the starting second of the chunk (`seconds_start`) and the total duration of the original audio (`seconds_total`), as illustrated in Figure 2. To clarify, if a 95-second slice is extracted from a 180-second track starting at the 14-second mark, `seconds_start` would be 14, and `seconds_total` would be 180 (see Figure 2, Left). These temporal markers are converted into discrete learned embeddings and amalgamated with the text prompt features along the sequence axis prior to their incorporation into the U-Net's cross-attention layers. In cases where the training involves audio shorter than the 95-second window, silence padding is applied to align with the training window's length (see Figure 2, Right). During the inference phase, `seconds_start` and `seconds_total` serve as conditioning inputs, granting users the flexibility to define the overall length of the generated audio. For instance, with the 95-second model, setting `seconds_start` to 0 and `seconds_total` to 30 cues the model to produce 30 seconds of audio followed by 65 seconds of silence. This novel conditioning technique empowers users to generate music and sound effects of varying lengths tailored to their specific requirements.

#### Diffusion Model

The diffusion model at the heart of Stable Audio is a U-Net architecture with an impressive 907 million parameters, drawing inspiration from Moûsai's architecture. This model is meticulously structured with four levels of symmetrical downsampling encoder blocks and corresponding upsampling decoder blocks. These are interconnected with skip connections that bridge each encoder and decoder block at identical resolutions. The channel counts for these four levels are set at 1024 for the first three levels and 1280 for the final level, with respective downsampling factors of 1 (indicating no downsampling), 2, 2, and 4. At the culmination of the encoding process, a 1280-channel bottleneck block serves as the pivotal juncture before the upsampling begins.

Each block in this architecture is composed of two convolutional residual layers, succeeded by a sequence of self-attention and cross-attention layers to capture the intricate dependencies within the data. Notably, each encoder or decoder block houses three such attention layers, with the exception of those in the initial U-Net level, which contain only one. Leveraging a fast and memory-efficient attention mechanism allows the model to efficiently handle longer audio sequences without excessive computational demands.

The diffusion process is finely tuned by timestep conditioning, introduced through FiLM layers, which adapt the model's activations according to the noise level at each step. The incorporation of the prompt and timing information is achieved via cross-attention layers, ensuring the generated content aligns with the user's input.

#### Inference

During inference, Stable Audio's sampling employs the DPM-Solver++ strategy, complemented by classifier-free guidance at a scale of 6, following the approach of Lin et al. (2024). A total of 100 diffusion steps are utilized. The design of Stable Audio is inherently flexible, catering to the generation of variable-length, long-form music and sound effects. It operates within a 95-second window, utilizing the timing condition to extend or truncate the audio signal as specified by the user, with any excess duration filled with silence.

For the presentation of audio content that is shorter than the window length, a straightforward silence-trimming approach can be applied. 

### Training

Stable Audio's training paradigm was extensive, covering various components of the model, each contributing to the final quality of the generated audio.

#### Dataset

The dataset employed for training Stable Audio is expansive, comprising 806,284 audio files totaling over 19,500 hours. This dataset is diverse, with a substantial portion (66% or 94%) consisting of music, while sound effects (25% or 5%) and instrument stems (9% or 1%) make up the rest. The dataset, sourced from AudioSparx, includes corresponding text metadata, providing rich context for each audio file.

#### Variational Autoencoder (VAE)

The VAE training utilized automatic mixed precision and was carried out over 1.1 million steps on 16 A100 GPUs, with an effective batch size of 256. After 460,000 steps, the encoder was frozen, and the decoder received further fine-tuning for an additional 640,000 steps. A multi-resolution sum and difference STFT loss tailored for stereo signals was applied to ensure high-quality stereo reconstruction, alongside A-weighting before the STFT, employing window lengths ranging from 32 to 2048. Adversarial and feature matching losses were incorporated using a multi-scale STFT discriminator adapted for stereo input. The losses were balanced with different weights to achieve the desired training effect, emphasizing spectral fidelity while regulating adversarial and feature matching influences.

#### Text Encoder

For the text encoder, the training strategy involved 100 epochs on the research team's dataset, using 64 A100 GPUs and an effective batch size of 6,144. The training setup adhered to the recommendations of the CLAP authors, focusing on a language-audio contrastive loss to fine-tune the model for the task.

#### Diffusion Model

The diffusion model's training took advantage of exponential moving average and automatic mixed precision, spanning 640,000 steps on 64 A100 GPUs. The effective batch size was maintained at 256. Audio files were resampled to 44.1kHz and segmented into 95.1-second slices, with longer files being randomly cropped and shorter ones padded with silence. A v-objective with a cosine noise schedule and continuous denoising timesteps was implemented, and dropout was applied to the conditioning signals to facilitate classifier-free guidance. Notably, during the diffusion model training, the text encoder remained frozen.

#### Prompt Preparation

The audio files in the dataset are each associated with descriptive text metadata, which includes both natural-language content descriptions and domain-specific metadata like BPM, genre, moods, and instruments for music tracks. In training the text encoder and the diffusion model, text prompts were constructed by randomly concatenating subsets of this metadata. This method allowed for the specification of particular properties during inference without necessitating constant metadata presence. The strategy varied, with half the samples including the metadata-type and using the "|" character for separation, while the other half omitted the metadata-type and used commas for a more natural flow. Lists of metadata values were randomized to avoid any bias in the order of presentation.

### Methodology

#### Quantitative Metrics

The study's evaluation of Stable Audio's capabilities is anchored in a collection of quantitative metrics, each meticulously crafted to measure specific attributes of the generated audio output.

**FDopenl3**
The Fréchet Distance (FD) is a key metric in the study, utilized to assess the fidelity of the audio produced by Stable Audio. It compares the statistical characteristics of the generated audio against a reference set within a particular feature space. A lower FD indicates a higher resemblance to the reference audio, suggesting a more plausible and authentic sound. The study deviates from previous methods that use the VGGish feature space, which is restricted to 16kHz signals. Instead, it opts for the Openl3 feature space capable of processing up to 48kHz signals, aligning with Stable Audio's 44.1kHz output. The study introduces a unique approach for stereo signals by extracting Openl3 features separately from the left and right channels and then concatenating them. This results in the FDopenl3 metric, which comprehensively evaluates the authenticity of full-band stereo signals.

**KLpasst**
The Kullback–Leibler (KL) divergence in the study serves to evaluate the semantic likeness between the generated audio and the reference audio. Utilizing the advanced audio tagging capabilities of PaSST, which operates at 32kHz, the KL divergence measures how closely the generated audio aligns semantically with the reference. To suit the model's generation of long-form audio, the study adapts the KL metric to handle varying and extended lengths of audio. This adjustment entails segmenting the audio into overlapping windows, computing the average logits, and then applying a softmax function. Consequently, the study introduces the KLpasst metric, tailored to assess the semantic correspondence of extensive audio lengths up to 32kHz.

**CLAPscore**
In the study, the CLAPscore is computed to determine the degree to which the generated audio aligns with the provided text prompt. It is determined by the cosine similarity between the CLAPLAION text embeddings of the prompt and the audio embeddings of the generated output. A higher CLAPscore signifies a stronger adherence to the text prompt. Moving away from the standard practice of evaluating 10-second audio inputs, the study employs the 'feature fusion' variant of CLAPLAION for longer audio segments. This method combines inputs from different parts of the audio, including a global input and random crops from distinct sections. The evaluation audios are resampled to 48kHz to match the model's 44.1kHz generated audio. The CLAPscore metric, therefore, is innovatively designed to evaluate how well extended audios beyond 10 seconds adhere to the text prompts at 48kHz.

In essence, the study has refined and adapted established metrics to more accurately evaluate long-form, full-band stereo audio outputs. These modifications ensure that the quantitative metrics can accommodate variable-length inputs, offering a robust and comprehensive evaluation of the audio quality and semantic fidelity produced by Stable Audio.

#### Qualitative Metrics

The study introduces a comprehensive suite of qualitative metrics to assess the perceptual attributes of audio generated by Stable Audio.

**Audio Quality**
This metric evaluates the perceptual fidelity of the audio. It ranges from low-fidelity, which may contain noticeable artifacts, to high-fidelity, characterized by clarity and absence of artifacts. The listeners' evaluations across this spectrum provide insight into the audio's overall quality.

**Text Alignment**
This metric determines how accurately the generated audio matches the given text prompt. It is critical for assessing the model's capacity to interpret and convert textual descriptions into congruent audio representations.

**Musicality (Music Only)**
Exclusive to music signals, musicality measures the model's ability to produce harmonious and melodically coherent sequences, reflecting an understanding of music theory and compositional structure.

**Stereo Correctness (Stereo Only)**
For stereo outputs, this metric gauges the precision of spatial audio imaging. It evaluates whether the stereo representation contributes positively to the listening experience through proper spatial distribution of sound.

**Musical Structure (Music Only)**
This metric examines the presence of traditional compositional elements within the generated music, such as intros, developments, and outros. It considers whether these structural components are discernible in the generated pieces.

**Human Ratings and Mean Opinion Scores**
The study collects human ratings for these metrics, utilizing a scale from 'bad' (0) to 'excellent' (4) for audio quality, text alignment, and musicality. Recognizing the difficulty that listeners may have in evaluating stereo correctness, the study opts for a binary response to this metric. The evaluation of musical structure is similarly binary, asking respondents to simply confirm the presence or absence of standard musical elements.

**Perceptual Experiment Platform**
The perceptual evaluations were conducted using the webMUSHRA platform, which provides a standardized method for such assessments. The study notes that this approach to evaluating musicality, stereo correctness, and musical structure is novel, with no known precedents in previous research, underscoring the innovative nature of the study's evaluation methodology.

#### Evaluation Data

**Quantitative Experiments**
The study utilizes the MusicCaps and AudioCaps benchmarks for its quantitative analysis. MusicCaps contains 5,521 music segments from YouTube, each with a caption; however, only 5,434 audio files were available for download. The AudioCaps test set features 5,979 audio segments, each with several captions, with 881 audios comprising 4,875 captions available for download. The study generates one audio per caption, culminating in 5,521 audio pieces for MusicCaps and 4,875 for AudioCaps. Despite these benchmarks not being tailored for full-band stereo signal evaluation, the original data largely consists of stereo and full-band recordings. The study resamples this data to 44.1kHz to align with the output of Stable Audio. Recognizing that the typical MusicCaps and AudioCaps segments are 10 seconds long, the study extends its analysis to the full-length of the tracks, addressing variable-length, long-form content. It is noted that the provided captions do not consistently represent the full duration of these longer tracks but are instead accurate for 10-second segments. As a result, the reference audios are limited to 10 seconds, while the generated audios range from 10 to 95 seconds. Adapting the evaluation metrics and datasets was a crucial step to ensure compatibility with the study's full-band stereo generation objectives.

**Qualitative Experiments**
For qualitative evaluation, prompts were randomly selected from MusicCaps and AudioCaps datasets. The study focused on high-fidelity synthesis, excluding prompts indicative of "low quality" or the like. Ambient music prompts were also omitted due to the difficulty users experience in assessing musicality within this genre. Furthermore, prompts associated with speech were not considered, as they fall outside the intended scope of the model's capabilities.

#### Baselines

The researchers acknowledge that a direct comparison with certain models like Moûsai or JEN1 is not feasible due to the non-availability of their model weights. Therefore, they opt to benchmark Stable Audio against other accessible state-of-the-art models such as AudioLDM2, MusicGen, and AudioGen. These models span the current spectrum of latent diffusion and autoregressive models, offering both stereo and mono outputs across a range of sampling rates.

For AudioLDM2, the variants evaluated are 'AudioLDM2-48kHz' for generating full-band mono sounds and music, 'AudioLDM2-large' for 16kHz mono outputs, and 'AudioLDM2-music' which is dedicated to 16kHz mono music generation. The MusicGen models assessed include 'MusicGen-small' for compute-efficient music generation, 'MusicGen-large' for higher capacity generation, and 'MusicGen-large-stereo' specifically for stereo outputs. However, given that MusicCaps contains vocal-related prompts which MusicGen models do not cater to, the researchers conducted additional evaluations of MusicGen without vocal prompts, as detailed in Appendix E. Furthermore, 'AudioGen-medium' was included in the assessment as the only open-source autoregressive model available for sound synthesis.

### Experiments

The study conducted a series of experiments to evaluate various aspects of Stable Audio's performance, from audio reconstruction fidelity to inference speed. Below are the findings from these evaluations.

#### Autoencoder Impact on Audio Fidelity

The study began by assessing audio reconstruction quality using the autoencoder. A subset of the training data was encoded into the latent space and then decoded. The comparison of FDopenl3 scores between the original and autoencoded audio indicated a minor decrease in quality post-reconstruction. Despite this, informal listening tests implied that the perceptual quality differences were minimal. For a public assessment, audio samples from these tests were made available on the demonstration website.

#### Optimal Text Encoder Performance

An ablation study was conducted to evaluate the efficacy of different text encoders, which included the open-source CLAP (CLAPLAION), a privately trained CLAP-like model (CLAPours), and open-source T5 embeddings. During the base diffusion model training, these text encoders were kept frozen, and their performance was gauged using qualitative metrics on the MusicCaps and AudioCaps benchmarks. The results showed a marginally better performance for CLAPours, which prompted its selection for subsequent experiments. The primary benefit of using a CLAP model trained on the same dataset is the harmonization of text embeddings with the diffusion model, which minimizes potential mismatches.

#### Timing Conditioning Accuracy

The study examined the accuracy of timing conditioning by generating audio at different specified lengths and measuring the actual output lengths. While some variance was noted, particularly for audio lengths in the 40-60 second range, the model consistently produced audio of the intended lengths. The length measurement was based on a simple energy threshold, and some of the shorter length detections may be attributed to the limitations of the silence detection method used. The overall results, which are elaborated in Appendix C, were deemed reliable.

#### Comparison with State-of-the-Art

In performance assessments, Stable Audio excelled in audio quality and exhibited competitive text alignment and musicality scores, especially on the MusicCaps benchmark. Text alignment scores were slightly lower on AudioCaps, possibly due to a limited representation of sound effects in the training dataset. The model also proficiently generated accurate stereo signals, although it received lower scores for stereo correctness on AudioCaps, potentially due to the nature of the selected prompts. However, the consistency of stereo quality was maintained. Notably, Stable Audio was capable of generating music with discernible structure, such as intros and outros, distinguishing it from other leading models.

#### Model Inference Speed

When it came to inference speed, Stable Audio showcased remarkable efficiency. Operating at stereo 44.1kHz, it outperformed autoregressive models and was faster than AudioLDM2 variants, which operated at mono 16kHz. This efficiency was even more significant in comparison to AudioLDM2-48kHz, which operates at mono 48kHz. These results, detailed in the corresponding tables, illustrate the advantages of the latent diffusion approach in terms of inference speed.

### Conclusions

The latent diffusion model introduced in this study enables the rapid generation of variable-length, long-form stereo music and sounds at a 44.1kHz sample rate, informed by textual and timing inputs. Through the exploration of novel qualitative and quantitative metrics designed specifically for evaluating long-form full-band stereo signals, the researchers have established Stable Audio as a top performer in two public benchmarks. This model notably differentiates itself from other state-of-the-art models by its ability to generate structured music and stereo sound effects.

The technology developed by the researchers marks a significant enhancement in assisting humans with audio production tasks, offering the capability to generate variable-length, long-form stereo music and sound effects based on textual descriptions. This innovation broadens the toolkit available to artists and content creators, potentially enriching the creative landscape. However, the researchers also acknowledge the risks associated with such technology, particularly in reflecting biases present in the training data. This could have implications for cultures underrepresented in the dataset. Additionally, the contextual richness of audio recordings and music underscores the need for careful consideration and collaboration with various stakeholders. In recognition of these complexities, the researchers commit to ongoing research and collaboration with stakeholders, such as artists and data providers, to ensure responsible stewardship of AI in the field of audio production.

### Appendix

#### A. Inference Diffusion Steps

In diffusion generative modeling, the study highlights a critical trade-off between the quality of generated outputs and the number of inference steps employed. This balance between output quality and inference speed is a pivotal aspect for practical deployment.

![figure4.png](images%2Ffigure4.png)

Figure 4 illustrates that a significant improvement in output quality is typically achieved within the initial 50 inference steps, beyond which there are diminishing returns on additional computational investment. Based on these findings, the authors of the study set the total inference steps for their model at 100. This choice was made to ensure a high-quality threshold for the generated audio while maintaining computational efficiency.

The study notes that there may be potential to further reduce the number of diffusion steps, which could lead to faster inference times with only a slight decrease in audio quality. This suggests a promising direction for future research, aiming to optimize the inference process by finding a more aggressive, yet efficient, balance between the number of steps and the quality of the generated content.

#### B. MusicCaps and AudioCaps: The Original Data from YouTube

![figure5-6.png](images%2Ffigure5-6.png)

The original datasets from MusicCaps and AudioCaps, as obtained from YouTube, are predominantly composed of stereo and full-band signals, contrary to the common practice of utilizing mono versions at 16kHz for evaluation purposes. Figures 5 and 6 illustrate the distribution of the sampling rates within the original datasets, highlighting the prevalence of stereo recordings and a wide range of sampling rates extending up to 48kHz.

In this study, the researchers have chosen to utilize this original, higher-quality data, resampling it to 44.1kHz to align with the target bandwidth of Stable Audio. This decision was made to ensure that the evaluation of Stable Audio would be conducted in an environment that closely mimics the rich and complex nature of real-world audio data. By doing so, the study offers a more accurate assessment of the model's performance, capitalizing on the full spectrum and spatial attributes inherent in the original stereo recordings.

#### C. Timing Conditioning: Additional Evaluation

![figure7.png](images%2Ffigure7.png)

Further investigation into the timing conditioning of Stable Audio was conducted by generating MusicCaps prompts of various lengths—specifically, 30, 60, and 90 seconds. Figure 7 presents a histogram that illustrates the measured lengths of the generated audios, grouped according to the specified lengths: 30 seconds (blue), 60 seconds (red), and 90 seconds (green). Consistent with the observations in Section 6.3, the measurement of audio length was determined by the point at which the signal transitions to silence, identified using a basic energy threshold.

The results of this additional evaluation show that the timing conditioning in Stable Audio operates with a high degree of accuracy, producing audio segments that closely match the expected lengths, albeit with a minor tendency towards generating slightly shorter audios. This indicates that the generated audio typically concludes just before the targeted endpoint, validating the approach of cutting off the audio signal at the specified length for a precise duration match.

It should be noted that the shortest measured lengths observed in the histogram could be attributed to the limitations of the simple silence detection method employed. These might not accurately reflect the actual duration of the audio content but rather the sensitivity of the threshold used to define silence. This minor discrepancy underscores the importance of refining the method of silence detection to ensure the most accurate assessment of audio lengths generated by the model.

#### D. Related Work: Additional Discussion on Latent Diffusion Models

The discussion on latent diffusion models in the related work section elaborates on how Moûsai and JEN-1, while being related to the study's work, differ in several crucial aspects. Here's a deeper dive into the distinctions:

**Moûsai**
- **Latent Representation**: Unlike Moûsai, which uses a spectrogram-based encoder and requires 100 decoding steps, the model in this study employs a fully-convolutional end-to-end VAE, significantly contributing to faster inference times.
- **Realtime Factor**: The study's model achieves a realtime factor of ×10, compared to Moûsai's ×1, reflecting superior efficiency.
- **Timing Conditioning**: While Moûsai incorporates a form of timing conditioning within its prompts, the study's model introduces explicit timing conditioning, allowing for the generation of variable-length audio.

**JEN-1**
- **Model Complexity**: JEN-1's approach includes a masked autoencoder with reduced-dimensionality latents and multitask training. In contrast, the study's model opts for a more straightforward architecture without these complexities, simplifying implementation and training without compromising on performance.
- **Audio Length**: JEN-1 is designed for generating 10-second music segments, whereas the study's model is capable of producing variable-length, long-form audio for both music and sound effects.

**AudioLDM2**
- **Shared Representations**: AudioLDM2 relies on a shared representation for diverse audio inputs, including speech and music. The study's model, Stable Audio, does not require such a complex representation since it does not involve speech generation, thus simplifying the model with a dedicated CLAP text encoder.
- **Performance**: In terms of speed, audio quality, and text alignment for music generation, Stable Audio surpasses AudioLDM2. However, AudioLDM2 excels in text alignment for sound effects generation.

**Text Embeddings**
- Moûsai, JEN-1, and AudioLDM2 typically use open-source T5 or FLAN-T5 text embeddings, but the study's model utilizes CLAP text embeddings trained on the same dataset as the diffusion model. This ensures a higher level of coherence and alignment across all components of the model, avoiding mismatches in distribution or vocabulary between text embeddings and the diffusion model.

#### E. Additional MusicCaps Results: Quantitative Evaluation Without Singing-Voice Prompts

The MusicCaps dataset incorporates prompts related to vocals. However, the MusicGen model's publicly available weights, which were utilized for benchmarking purposes, are not trained to generate vocal sounds. To ensure a fair comparison with MusicGen, the study also conducted evaluations using a subset of 2,184 MusicCaps prompts that exclude vocals.

![table1.png](images%2Ftable1.png)

![table4.png](images%2Ftable4.png)

The results presented in Table 4 align with the findings from Table 1: Stable Audio outperforms the other models across most metrics, except for the KL_passt score, where MusicGen-large shows comparable performance to Stable Audio. These results underscore the capability of Stable Audio to generate high-quality audio that aligns well with text prompts, even in the absence of vocal content, thereby reinforcing its position as a leading model in the field. The quantitative outcomes highlight the proficiency of Stable Audio, particularly in the context of non-vocal music generation, and illustrate the model's strength in this specific area of audio synthesis.

#### F. Implementation Details

- The codebase for reproducing Stable Audio, including the tools and scripts used for its development, is openly accessible at [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools).
- The configuration file detailing the structure and parameters for the VAE autoencoder used in Stable Audio is available for public access. This file provides essential information for replicating the model's training and setup.
- Similarly, the configuration for the latent diffusion model component of Stable Audio is also available online, allowing for transparency and reproducibility of the model's training and definition.
- These configuration files are integral for those looking to delve into the architecture and finer details of Stable Audio's implementation, offering a clear and concise representation of the model's setup.
- The code required to reproduce the metrics used to evaluate Stable Audio's performance is hosted at [Stable Audio Metrics](https://github.com/Stability-AI/stable-audio-metrics).
- For training the text encoder within Stable Audio, the study utilized the code provided by the CLAP authors, which is based on their private dataset. This code is accessible at [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP), facilitating the training of text encoders with customized datasets.

## Personal Notes

![object-oriented-approach.png](images%2Fobject-oriented-approach.png)

As we conclude our comprehensive exploration into Stability AI's Generative Models, I trust the intention behind this series has become clear: Stability AI exemplifies an object-oriented philosophy applied to technological innovation. Each of their models is a derivative of the foundational diffusion model concept, yet uniquely specialized, incorporating polymorphic characteristics into the core structure. They expertly encapsulate the complexities of specific tasks while maintaining the integrity of the underlying abstract model.

Your journey through the world of AI should mirror this experience: initially overwhelming, yet progressively becoming more manageable and enlightening as you build upon fundamental concepts. By focusing on these foundations, the complexities start to unravel, allowing you to encapsulate the more intricate details within layers of abstraction.

Consider the in-depth exploration of Stability AI models as a prime example. Grasping the concept of the diffusion model lays the groundwork; from there, everything else begins to flow more smoothly. It becomes a matter of iteratively adding polymorphic traits to models and methods, all the while neatly encapsulating the complexities. This approach simplifies the learning process, making the journey through AI much more accessible. 

Reflecting on this series, I am struck by the eloquence of Stability AI's methodology. It serves as a robust endorsement of the object-oriented approach as a universal principle for innovation, not just in technology.

![latent-dream-factory.png](images%2Flatent-dream-factory.png)

It all begins with a diffusion model, the latent space acting as a sort of enchanting dream factory, conjuring up creative pieces. These initial creations are then meticulously refined into intricate, high-dimensional outcomes. This process is a remarkable manifestation of creativity, transforming abstract ideas into tangible, expressive works.

For those seeking a profound understanding of AI, embarking on a journey through Stability AI's developments could be profoundly instructive. Their work provides a robust framework that can either stand as a foundation for further exploration or as a comparative benchmark for alternative AI methodologies. At its best, it equips you with a solid base for crafting your unique approach to AI.

On a personal note, music resonates with me deeply. While the number of domains that ignite my passion is vast—perhaps surprisingly so—music and SFX generative AI hold a special place. It is a field where I intend to channel significant energy and dedication going forward. Stability AI's strides in this arena undoubtedly lay a firm groundwork for my endeavors.

Without a doubt, their models will play a pivotal role in shaping my contributions to the landscape of generative AI in music and sound effects.