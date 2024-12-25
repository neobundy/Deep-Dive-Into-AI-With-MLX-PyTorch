# Deep Dive into Google Gemini 1.5

![gemini-title.png](images%2Fgemini-title.png)

🔗 **Blog**:
https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/

🔗 **Tech Report**:
https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf

🧐 _Disclaimer: While we approach the claims made by Google in their blog and technical report with an open mind, it's prudent to maintain a healthy dose of skepticism. Our cautious stance stems from a history of instances where the reality of technological advancements didn't fully align with initial promises. It's not that we lack enthusiasm for the potential breakthroughs—quite the opposite. We eagerly anticipate the possibility of these claims being substantiated. However, experience advises that we temper our expectations and scrutinize these advancements with a finer grain of salt. I'm intrigued to see the model myself, as you've probably guessed. I'm currently on their waitlist, eagerly awaiting my turn._

In light of our measured approach, we cautiously examine the advancements claimed by the researchers with the rollout of Google Gemini Pro 1.5. According to their statements, this model marks a significant advancement in AI capabilities, notably in the realm of long-context understanding across multiple modalities.

At this pivotal moment in AI development, the field is abuzz with transformative advancements that promise to broaden AI's impact worldwide. Following the introduction of Gemini 1.0, there has been a continuous push to refine and expand its functionalities. The researchers have now introduced their next-generation model, Gemini 1.5, which purportedly delivers a substantial boost in performance. This progress is attributed to a series of research and engineering breakthroughs that encompass the full scope of foundational model development and infrastructure, including a shift towards a more efficient Mixture-of-Experts (MoE) architecture.

The first iteration of this model made available for preliminary testing is Gemini 1.5 Pro. This mid-size multimodal model is crafted to perform adeptly across a diverse set of tasks. It reportedly matches the performance of the largest model to date, 1.0 Ultra, and features a pioneering experimental capability for enhanced long-context understanding. Initially, the model offers a standard context window of 128,000 tokens, with plans to extend this up to 1 million tokens for a select cadre of developers and enterprise clients via AI Studio and Vertex AI in a private preview.

The team behind Gemini 1.5 Pro is dedicated to further refining the model, focusing on reducing latency, cutting computational demands, and improving the overall user experience in anticipation of making the full 1 million token context window available.

Grounded in leading-edge research on Transformer and MoE architectures, Gemini 1.5 Pro employs a modular approach by segmenting into smaller "expert" networks to boost training and operational efficiency. This strategy is designed to enable swift mastery of complex tasks while ensuring quality, thus accelerating the rollout of more sophisticated iterations of Gemini.

A key feature of Gemini 1.5 Pro is its expanded context window, which allows for the processing of extensive volumes of information in a single query, thereby broadening its application across various data formats and significantly augmenting its analytical and inferential capabilities.

The claimed superior performance of the model across diverse evaluations, including text, code, images, audio, and video, suggests that it outperforms Gemini 1.0 Pro in the majority of benchmarks and is on par with the performance of 1.0 Ultra. Its ability to sustain high performance levels, even as the context window expands, and to demonstrate remarkable in-context learning capacity, hints at its potential to assimilate new skills from extensive prompts without additional fine-tuning.

Ethical considerations and safety assessments form a core part of the model's development process, with exhaustive evaluations being conducted in line with stringent safety policies and AI principles. The researchers have committed to a responsible deployment strategy, ensuring Gemini 1.5 Pro is subjected to rigorous testing for ethics, content safety, and representational harms, in addition to developing new tests tailored to its unique long-context features.

Our review cautiously acknowledges the potential breakthroughs associated with Gemini 1.5 Pro, emphasizing its possibilities for advancing AI applications through improved efficiency, extended context understanding, and a strong commitment to ethical and safety standards.

## Gemini 1.5 In Simple Terms

The technical report on Gemini 1.5 Pro details the development of a new AI model designed to handle a wide range of tasks involving text, images, and audio. A key feature of this model is its ability to understand and process information over much longer sequences of data—or "context"—than its predecessors. In simple terms, if you think of context as the amount of information the model can consider at one time to make decisions or generate responses, Gemini 1.5 Pro can keep track of much more information at once. This is significant because, in many real-world applications, the ability to consider a larger context can lead to better understanding and more accurate responses.

Specifically, while previous models like those in the Gemini 1.0 series could handle up to 32,000 "tokens" (a token can be a word, part of a word, or a piece of an image), Gemini 1.5 Pro extends this to multiple millions of tokens. This leap forward means the model can process, for example, entire books, long videos, or extensive databases in a single task without needing to break the data into smaller chunks. 

The report also highlights that this extended context window doesn't come at the cost of the model's performance on standard tasks. Instead, Gemini 1.5 Pro maintains or even improves performance across a range of tasks compared to its predecessors. Additionally, the model shows a remarkable ability to learn from new, highly specific information it hasn't seen during its training phase—like translating a language with very few speakers by simply reading a grammar manual.

Evaluating models capable of processing such long contexts presents new challenges, especially for tasks that mix different types of data (like text and images). The traditional benchmarks used to assess AI models often don't test their ability to handle these longer sequences effectively. The report suggests new methods for evaluating these capabilities, indicating a shift towards more complex and realistic assessments of what AI models can do.

In essence, the Gemini 1.5 Pro report outlines the creation of a more capable AI model that can understand and interact with much larger volumes of information at once, a step that could significantly enhance AI applications across various fields. However, it also acknowledges the need for new ways to measure and understand these capabilities accurately.

## Deep Dive - Technical Report

Gemini Team, Google. (2024). Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context
🔗 https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf

In our analysis, we explore the innovative strides made by the researchers in developing Google Gemini Pro 1.5. This iteration marks the inaugural release from the new Gemini 1.5 series, distinguished by its highly-capable multimodal models. It employs a novel mixture-of-experts architecture alongside significant advancements in both training and serving infrastructure. This enables the model to redefine the boundaries of efficiency, reasoning, and long-context performance. Capable of handling up to at least 10 million tokens, Gemini 1.5 Pro sets a new precedent in the field, handling extensive long-form mixed-modality inputs like comprehensive document collections, multiple hours of video, and nearly a full day's worth of audio. This leap forward not only surpasses the prior Gemini 1.0 Pro in performance across a broad spectrum of benchmarks but does so with markedly less computational demand for training.

![figure1.png](images%2Ffigure1.png)

In our analysis, we focus on the architecture of Gemini 1.5 Pro, a cutting-edge development within the Gemini series. The model is a transformative leap from its predecessors, employing a sparse mixture-of-expert (MoE) Transformer-based framework that builds upon the multimodal capabilities and research advancements of Gemini 1.0. The researchers have meticulously integrated a rich lineage of MoE research alongside the broad spectrum of language model research, demonstrating a commitment to pushing the boundaries of model efficiency and capability.

![table1.png](images%2Ftable1.png)

Gemini 1.5 Pro stands out for its innovative use of a learned routing function, which skillfully directs inputs to a specific subset of the model's parameters for processing. This approach of conditional computation enables the model to expand its total parameter count while maintaining a constant number of activated parameters for any given input. This design choice represents a significant stride towards achieving higher efficiency and efficacy in processing complex tasks.

The model benefits from a comprehensive suite of improvements across its entire stack, including architecture, data optimization, and systems. These enhancements allow Gemini 1.5 Pro to deliver quality comparable to that of Gemini 1.0 Ultra while demanding significantly less computational power for training and offering increased efficiency in deployment. Notably, Gemini 1.5 Pro introduces substantial architectural modifications that facilitate a deep understanding of long-context inputs, accommodating up to 10 million tokens without compromising performance.

This capability to process long-context inputs translates into the model's ability to handle nearly a day's worth of audio, more than ten times the length of the text of "War and Peace," the entire Flax codebase, or three hours of video at one frame per second. The native multimodality of Gemini 1.5 Pro and its capacity to interleave data from different modalities within the same input sequence mark a significant evolution in model design. The researchers highlight several novel capabilities unlocked by these advancements, including positive results on context lengths up to 10 million tokens, underscoring the ongoing exploration and potential of long-context understanding and application.

🧐 _The **Mixture-of-Experts (MoE) architecture** can be explained in simpler terms as a way to build a smarter, more efficient team to tackle big problems. Imagine you're the coach of a sports team with a roster full of players, where each player has a unique skill set. Some are great at defense, others excel at scoring, and some are fantastic all-rounders. Now, instead of having all the players on the field at the same time, you decide who plays based on what the team needs at any moment in the game. If you need to defend a lead, you bring on your best defenders. If you need to score, you bring in your top scorers._

_In the MoE architecture, the "players" are smaller neural networks called "experts," and the "coach" is a part of the model called the "gating network." The gating network evaluates the task at hand (like a piece of data that needs processing) and decides which experts are best suited to handle it. Only those selected experts will work on the task, making the process more efficient because you're not using all the resources all the time, just the ones best for the job._ 

_This setup allows the model to handle a wide range of tasks very efficiently because it can dynamically adapt to the complexity or simplicity of each task. It's like having a highly specialized team ready to tackle anything, but you only call on the specialists you need, when you need them._

🧐 _The concept of a **large context window** in models like Transformers is akin to a person trying to solve a complex puzzle by considering more pieces at once. Ideally, the more pieces (or context) you can consider, the better you understand the puzzle (or the task at hand). However, this doesn't necessarily guarantee better performance due to a phenomenon known as the "diminishing rate of returns" and inherent limitations of the Transformer architecture._

_Firstly, the diminishing rate of returns refers to the point at which adding more context starts to offer less and less improvement in performance. Imagine trying to read a book where, in order to understand the sentence you're reading, you also need to keep in mind every sentence you've read before. Up to a certain point, more context helps you grasp the story better. But after a while, the additional information becomes too much to process effectively, making it harder, not easier, to understand the current sentence. Similarly, for AI models, beyond a certain context size, the additional information can't be utilized as effectively, leading to minimal performance gains compared to the computational cost._

_Secondly, the inherent limitations of the Transformer architecture come into play. Transformers process information through attention mechanisms, which allow them to weigh the importance of different parts of the input data. However, this process requires significant computational resources, especially as the amount of data increases. The architecture struggles with long sequences because it has to compute relationships between every pair of elements in the input, leading to a quadratic increase in computation with respect to the sequence length. This not only makes processing very large contexts computationally expensive but also can dilute the focus of the model on the most relevant parts of the context due to the sheer volume of connections it needs to evaluate._

_Moreover, Transformers, by their standard design, don't have an inherent mechanism to prioritize newer or more relevant information over older or less relevant data in a long sequence. This means that as the context window grows, the model may not effectively distinguish which parts of the context are most important, potentially leading to a "watering down" of useful information and a decrease in performance on tasks requiring precise understanding or reasoning._

_Moreover, it's crucial to recognize the role of memory-compressing algorithms in managing these limitations. No AI system possesses infinite memory or attention span. Essentially, these algorithms are designed to optimize how information is stored and accessed, ensuring that the model can handle large volumes of data more efficiently. However, this doesn't equate to an unlimited capacity to process or comprehend information. Thus, the presence of a 1 million token context window doesn't directly translate to an equally expansive playground for context understanding. It's important not to equate the sheer size of the context window with an AI's ability to effectively utilize that context._

_In summary, while a larger context window offers the promise of better understanding and performance by providing more information, the diminishing rate of returns and the computational and architectural challenges of the Transformer design mean that simply increasing the context size is not a panacea for achieving higher performance._

### Training

Similar to its predecessors, Gemini 1.0 Ultra and 1.0 Pro, Gemini 1.5 Pro underwent training on Google's TPUv4 accelerators, spread across multiple data centers, leveraging multimodal and multilingual data. This pre-training dataset comprised a diverse range of sources, including web documents, code, and content such as images, audio, and video. For the instruction-tuning phase, the model was fine-tuned on a collection of multimodal data that included paired instructions and corresponding responses, with additional adjustments based on human preference data. 

### Evaluations

The study underscores a significant shift in the landscape of Large Language Models (LLMs) research, with a concerted effort to expand the models' context windows, thereby enhancing their ability to assimilate and utilize vast amounts of new, task-specific information not present in the training data. This advancement is pivotal for improving performance across a spectrum of natural language and multimodal tasks.

![figure2-3.png](images%2Ffigure2-3.png)

![figure4.png](images%2Ffigure4.png)

![figure5.png](images%2Ffigure5.png)

![figure6.png](images%2Ffigure6.png)

![figure7.png](images%2Ffigure7.png)

![figure8.png](images%2Ffigure8.png)

![figure9.png](images%2Ffigure9.png)

![figure10.png](images%2Ffigure10.png)

The researchers categorize recent strides in enhancing long-context abilities into several key areas: novel architectural innovations, post-training modifications, the integration of retrieval-augmented and memory-augmented models, and the creation of more coherent long-context datasets. Such concerted efforts have led to notable improvements in LLMs' long-context capabilities, highlighted by concurrent research exploring context windows up to 1 million multimodal tokens.

![table2.png](images%2Ftable2.png)

![table3.png](images%2Ftable3.png)

![table4.png](images%2Ftable4.png)

![table5.png](images%2Ftable5.png)

![table6.png](images%2Ftable6.png)

Gemini 1.5 Pro emerges as a formidable contender in this arena, significantly pushing the boundaries of context length to multiple millions of tokens without compromising performance. This capability is not only a technical marvel but also enables the processing of considerably larger inputs, a leap forward compared to existing models like Claude 2.1 and GPT-4 Turbo. The study reveals Gemini 1.5 Pro's impressive recall rates across varying token lengths, showcasing its adeptness at handling extensive and complex inputs across text, vision, and audio modalities.

The evaluation methodology adopted by the researchers is both diagnostic, focusing on probing the long-context capabilities, and realistic, tailored to multimodal long-context tasks. This dual approach facilitates a nuanced understanding of the model's long-context understanding and application across diverse scenarios.

![figure11.png](images%2Ffigure11.png)

Diagnostic evaluations, including perplexity over long sequences and needle-in-a-haystack retrieval studies, alongside more practical evaluations designed for multimodal long-context tasks, such as long-document question answering (QA), automatic speech recognition, and video QA, offer a comprehensive assessment of Gemini 1.5 Pro's capabilities. These evaluations not only benchmark the model's performance against leading models but also quantify its long-context understanding reliably up to 10 million tokens.

Through this meticulous examination, the study illuminates the profound capabilities of Gemini 1.5 Pro in navigating the complexities of long-context understanding and application, setting a new standard for future developments in the field of LLMs.

🧐 _Perplexity is a metric used in language modeling to measure how well a model predicts a sample of text. You can think of it like this: If the model were a person taking a multiple-choice quiz where each question is a word in a sentence and the choices are what the next word could be, perplexity tells us, on average, how many choices the model feels uncertain among._

_For example, if a model has a perplexity of 10 on a test, it's as if, each time it tries to predict the next word, it's hesitating among 10 equally likely words. A lower perplexity means the model is more confident in its predictions (it's like having fewer choices to hesitate over), indicating it understands the language better. High perplexity means more uncertainty and less understanding._

_So, perplexity is essentially a way of measuring how "perplexed" or "confused" the model is when it tries to predict what comes next in the text. The goal in language modeling is to reduce this confusion by training the model so it can make predictions with higher certainty and lower perplexity._

🧐 _I've purposefully condensed this section, which might come across as somewhat self-congratulatory. It's important to remember that claims regarding the number of tokens and the size of the context window can be made quite liberally. Drawing from personal experience, Claude 2.1, with its extensive 100K context window, seems to fall short when compared to GPT-4, even when GPT-4 operates with much smaller context windows of 32K, 16K, or even 8K. This serves to highlight that the sheer volume of tokens or the extent of the context window doesn't directly translate to superior performance. What truly counts is the model's performance when normalized across various metrics. The plethora of benchmarks and tables touting dominance are not as indicative of a model's capability as one might think. The real test of a model's efficacy comes from hands-on experience and extended usage, which provides a more tangible sense of its practical value and performance._

## Core Capability Evaluations

The evaluations conducted spanned publicly recognized benchmarks as well as proprietary tests, covering a diverse range of modalities including text, vision, and audio. The aim was to evaluate the advancements Gemini 1.5 Pro has made over its predecessors, particularly the Gemini 1.0 series models, Gemini 1.0 Pro and Gemini 1.0 Ultra, and to assess whether enhancements in long-context capabilities might impact the model's performance on these foundational tasks.

![table7.png](images%2Ftable7.png)

The study's results underscored a significant leap forward in the Gemini series' capabilities, with Gemini 1.5 Pro surpassing Gemini 1.0 Pro in every aspect and either matching or surpassing Gemini 1.0 Ultra in the majority of benchmarks. Notably, this was achieved with Gemini 1.5 Pro being considerably more efficient in terms of its training demands. 

![table8.png](images%2Ftable8.png)

- In the realm of text, Gemini 1.5 Pro demonstrated significant improvements in Math Science & Reasoning, Multilinguality, and Coding, highlighting the model's improved reasoning skills and its versatility across different languages.

![table9.png](images%2Ftable9.png)

- Within vision-based tasks, the analysis showcased Gemini 1.5 Pro's adeptness in interpreting complex visual data, evidencing its advanced capabilities in understanding images and videos.

![table10.png](images%2Ftable10.png)

![table11.png](images%2Ftable11.png)

- The model's performance in audio tasks further emphasized its strength in recognizing and translating speech, suggesting its potential to enhance human-computer interaction through more natural and effective communication methods.

These evaluations of core capabilities reveal the considerable advancements achieved with the development of Gemini 1.5 Pro, illustrating its wide-ranging proficiency across various tasks and modalities. The findings affirm the model's sustained high performance, even as it advances the frontiers of long-context comprehension, thereby confirming its versatility and its promise to spearhead future innovations in the field.

🧐 _The rationale for condensing this section is similar to the previous one._

### Responsible Deployment

![figure12.png](images%2Ffigure12.png)

The researchers adopts the strategic approach for responsible deployment in developing Gemini 1.5 Pro, consistent with the practices established for the Gemini 1.0 models. This process, as depicted in their report, involves a structured framework designed to ensure the model's development aligns with ethical standards and safety considerations. The report updates the impact assessment methodology and mitigation strategies for addressing potential risks associated with the deployment of Gemini 1.5 Pro, building upon the foundation laid by the Gemini 1.0 series.

The impact assessment conducted by the study aims to comprehensively identify, evaluate, and document the societal benefits and potential harms stemming from the deployment of advanced models like Gemini 1.5 Pro. This assessment, carried out by the Responsible Development and Innovation team and reviewed by the Google DeepMind Responsibility and Safety Council, serves to reinforce adherence to the Google AI Principles.

Furthermore, the researchers underscore the enhanced societal benefits facilitated by Gemini 1.5 Pro's advanced capabilities, such as improved efficiency for users and enabling new applications through its extended understanding of long-content. However, they also caution against potential risks, emphasizing ongoing evaluations to identify and mitigate any new adverse effects introduced by the model, particularly those related to safety performance in processing longer input files.

Through this approach to responsible deployment, the study demonstrates a commitment to navigating the complex balance between leveraging technological advancements and ensuring ethical and safe utilization of AI models.

### Conclusion

In our analysis, we introduced Google's Gemini 1.5 Pro, the inaugural model from the Gemini 1.5 family, which utilizes a mixture-of-experts architecture to enhance efficiency, multi-modality, and long-context reasoning. The researchers expanded the content window significantly, setting a new industry standard by surpassing the previous maximum token limits with multi-million token capabilities. This enables Gemini 1.5 Pro to outperform its predecessors in long-context tasks and maintain or exceed performance in core multi-modal capabilities, even with reduced training computation.

The study also addresses the challenges in evaluating such advanced models, noting current benchmarks' limitations for long-context processing. It proposes innovative evaluation methodologies, including a "multiple needles-in-a-haystack" approach, to better assess model performance on complex tasks without heavy reliance on human labeling. This effort aims to advance the development and evaluation of multi-modal models, highlighting the need for new benchmarks that require complex reasoning over extended inputs to fully unlock the potential of long-context AI models.

## Personal Notes

I must admit, the technical report left me feeling underwhelmed. It seems to be filled with self-praise for 'evaluations' that lack clear methodological details beyond the expected use of a transformer with a Mixture-of-Experts (MoE) architecture.

Stripping away the evaluation sections that lack external validation, the report seems to lack substantial content.

It's surprising to see such a report coming from Google, a company I once held in high regard.

Nonetheless, given my past admiration for Google, I'm inclined to reserve judgment and give the model a chance once I gain access from the waitlist.