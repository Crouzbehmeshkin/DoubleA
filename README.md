# DoubleA
Embedding Augmentation for Large Pre-Trained Language Models

While data augmentation is straightforward for images, due to the complexities and discrete nature of language, it remains challenging to manipulate sentences without changing their meanings and labels.
Importantly, semi-supervised and unsupervised machine learning techniques rely on data augmentations that don't change the labels. For example, a method called Consistency Regularization works by encouraging a deep neural network's outputs to be similar for multiple augmentations of the same data sample.
Several data augmentation methods have been proposed for natural language processing, most notably Easy Data Augmentation (EDA). The performance gain of these techniques, however, has been questioned for large language models. In this project, I propose a new data augmentation technique at the embedding level to flexibly generate samples with a controllable count and strength without changing the sentence meanings.

## Easy Data Augmentation (EDA)
This simple yet effective method (for small tasks) suggests using these four operations to augment samples at the sentence level:
- Synonym Replacement: Randomly choose $n$ words that are not stop words and replace them with a randomly chosen synonym.
- Random Insertion: Randomly choose a word that is not a stop word, choose a random synonym of it, and insert it in a random position in the sentence. Repeat $n$ times.
- Random Deletion: Randomly delete a word.
- Random Swap: Randomly choose two words and swap their positions.

While methods like EDA can be useful for small tasks with limited data, they do not improve performance when a large pre-trained language model is being employed (see [How Effective is Task-Agnostic Data Augmentation for Pretrained Transformers?](https://arxiv.org/pdf/2010.01764.pdf)).

## Desiderata
Ideally, the data augmentation technique should be able to generate new samples with a controllable count and augmentation strength, without changing the meaning of the sentence.
  
## The Algorithm
Given a data sample $x$ (a sentence), first, EDA is used to generate $m$ new samples, which constitute $m+1$ samples with the original data sample. Next, a pre-trained language model is used to map these $m+1$ sentences to embeddings. Stacking these embeddings, we can create a $(m+1)\times(\mathrm{EmbeddingDim})$ matrix.

Noting that these sentences have more or less the same meaning, the obtained embeddings lie in a subspace of the large embedding space of the language model. To describe this subspace, I simply used SVD (any other dimensionality reduction method can be used based on the use case) to get a $(m+1)\times(m+1)$ matrix showing where individual sentences lie in this subspace. We can then take the embedding mean and variance.

According to the computed mean and variance, we can now flexibly generate new embeddings with arbitrary count and strength, while making sure they don't get too far from the mean (and thus the sentence meaning has not changed). 

## Report
A more detailed report of this project can be found [here](https://github.com/Crouzbehmeshkin/DoubleA/blob/master/report/AI2_Project_FinalReport.pdf).

## Achknowledgement
This project was initially done as an undergraduate final project under the supervision of Dr. Mahideh Soleymani and Mr. Ali Karimi (M.Sc.) at Sharif University of Technology. The report and additional work to wrap up the project was done in collaboration with Sepehr Asgarian at Western University. 
