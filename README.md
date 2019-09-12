# LASERtrain
This package reproduces the architecture described by Artetxe and Schwenk (2018a, 2018b) to train language-agnostic sentence embeddings. The authors have released a large model covering 93 languages as part of the [LASER project](https://github.com/facebookresearch/LASER); however the code used to train them remains unreleased. The code in this repository is an approximation to the actual code implemented by the authors using the description of the architecture and training parameters provided in their recent publications.

At the moment, the models produced with this software are *not compatible* with the models available in [LASER project](https://github.com/facebookresearch/LASER); this limitation will be tackled iin the near future.

The package includes instructions to reproduce the experiments described in Artetxe and Schwenk (2018a) in which a model is trained on the UN v.10 corpus and evaluated on the data released for the BUCC'18 shared task.

## Requirements
The following packages are required to reproduce run this package and reproduce the results reported:
- Fairseq
- Python
- PyTorch
- NumPy
- Sentencepiece
- Faiss, for fast similarity search and bitext mining
- jieba 0.39, Chinese segmenter (pip install jieba)

## References
Mikel Artetxe and Holger Schwenk, [Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings](https://arxiv.org/abs/1811.01136) arXiv, Nov 3 2018.
Mikel Artetxe and Holger Schwenk, [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464) arXiv, Dec 26 2018.
