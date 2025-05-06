## Representation Learning

### Points

* Proposes a different approach than mechanistic learning.
* Mechanistic learning provides more control, for example, with Induction heads, suggesting that any architecture can cater to NLP requirements.
* Representation learning is more aligned with Responsible AI. It focuses on identifying concepts and functions in a top-down manner.

### Representation Reading

Representation reading aims to understand the internal representations of a model.

* **LAT (Linear Articulation Tool/Technique)**: A method used for representation reading.
* **Design stimulus and task**:
    * **Concept**: Refers to a static idea.
    * **Function**: Refers to a process or behavior.
* **Collect neural activity**:
    * For a concept & token, take the hidden vector of the last token.
* **Constructing a linear model**:
    * **K-means**: A clustering algorithm that can be used to group similar representations.
    * **PCA (Principal Component Analysis)**: To calculate the score, take the difference of pairs, then perform PCA, and finally apply a dot product between the representation and the reading vector.
    * **Linear probing**: Training a simple linear model on top of the learned representations to see if a concept is encoded.

### Representation Control

Seeks to modify or control the internal representations of concepts and functions within a model.

#### Three Approaches

1.  **Use Reading vector**: Utilize the vector identified during representation reading.
2.  **Use contrast vector**: Employ a vector that distinguishes between two concepts.
3.  **Use LoRRA (Low-Rank Representation Adaptation)**: A technique to adapt representations.

#### Operators

* **Linear combination**: $R' = R \pm v$ (where $R$ is the original representation and $v$ is a control vector)
* **Piecewise Operation**: $R' = R$ (This likely implies a conditional modification not fully captured by the simple equation, or it's a placeholder for more complex piecewise functions.)
* **Projection**: $R' = R - \text{proj}_v(R)$ (where $\text{proj}_v(R)$ is the projection of $R$ onto $v$)

### Evaluation

#### Honesty

* Investigates whether models have a consistent internal concept of truthfulness.
* Apply LAT to datasets of true and false statements to extract a "truthfulness direction."
* **Resistance to Imitative Falsehood**: Evaluated using TruthfulQA.
    * Use three different sources for stimuli and then check on TruthfulQA.
    * Use the degree of truthfulness in terms of seven possible answers.

#### Honesty Extraction

* Used true statements as stimuli (dataset by Azaria & Mitchell (2023)).
    * For experimental setup, consider an "honest person" persona.
    * For referential setup, consider a "dishonest person" persona.
* LAT Reading vector achieves a classification accuracy of 90%.
* The function method was used for extraction.
* This can be used for lie and hallucination detection by summing the honesty scores at each token position across multiple layers.

#### Other Areas

The principles of representation reading and control can be extended to other areas such as:

* Utility
* Morality
* Power
* Probability
* Risk
* Emotions

### Pros

* Achieved State-of-the-Art (SOTA) results on the TruthfulQA dataset for honesty.
* Can be used to detect whether a model is lying.
* Enables the extraction of concepts like honesty and dishonesty.
* Allows for the control of honesty and potentially other concepts in a similar manner.
* Discusses the potential for monitoring and controlling concepts like morality, utility, probability, risk, and power-seeking.
* Demonstrated the ability to handle jailbreaks with an accuracy of over 90% in differentiating between harmful and harmless instructions.

### Cons/ Limitations/ Interesting to See

* Relies extensively on PCA, which primarily captures linear relationships. However, these concepts/functions may have non-linear relationships.
* Control methods depend on coefficients (a hyperparameter) to manage a particular concept. The choice of this coefficient depends on the specific model and its training procedure.
* The comparison with mechanistic interpretability is limited. Mechanistic interpretability delves into a much more granular level of understanding and also aids in making better architectural/design choices.
* Potential scalability issues.
* The harmfulness study mainly used LlaMa-2/Vicuna-13B. It's important to determine if these findings are applicable to other architectures as well.
* How pretraining versus post-training modifies these concepts and functions.
* The correlation of concepts between quantized/distilled models and their full-sized counterparts.
* How Chain of Thought (CoT) prompting influences the definition and representation of concepts and functions.


### Finance Domain Applications:
    * The concept of truthfulness can be applied to financial talks/discussions.
    * Advanced search on financial documents could benefit from detecting hallucination cases.