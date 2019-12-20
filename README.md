# LASERtrain
This package reproduces the architecture described by Artetxe and Schwenk (2018, 2019) to train language-agnostic sentence embeddings. The authors have released a large model covering 93 languages as part of the [LASER project](https://github.com/facebookresearch/LASER); however the code used to train them remains unreleased. The code in this repository is an approximation to the actual code implemented by the authors using the description of the architecture and training parameters provided in their recent publications.

At the moment, the models produced with this software are *not compatible* with the models available in [LASER project](https://github.com/facebookresearch/LASER); this limitation will be tackled iin the near future.

The package includes instructions to reproduce the experiments described in Artetxe and Schwenk (2019) in which a model is trained on the UN v1.0 corpus and evaluated on the data released for the BUCC'18 shared task.

## Requirements
The following packages are required to reproduce run this package and reproduce the results reported:
- Fairseq
- Python
- PyTorch
- NumPy
- Sentencepiece
- Faiss, for fast similarity search and bitext mining
- jieba 0.39, Chinese segmenter (pip install jieba)

## Tutorial: train and evaluate your LASER model
In this section, we reproduce the experiments carried out by Artetxe and Schwenk (2019).

### Download and prepare data
First step is to download the datasets needed to train and evaluate our model. Two datasets are required:
- [UN v1.0 corpus](https://cms.unov.org/UNCorpus/): the multilingual corpus on which our model will be trained
- [BUCC 2018 shared task data](https://comparable.limsi.fr/bucc2018/bucc2018-task.html
): the training and test data for the BUCC 2018 shared task that will be used to evaluate our model
Note that UN corpus does not cover one of the language pairs in BUCC 2018: German-English. To deal with this, Artetxe and Schwenk (2019) train a second model on Europarl multilingual corpus. We will not cover this second experiment in this tutorial, although the same steps described could be applied to train it.

For BUCC 2018, download all the 4 training data packages and the 4 test data packages to the sub-directory `data`. Once downloaded, uncompress all the packages using the coommand: `tar xjf bucc2018-ru-en.test.tar.bz2` 

## Acknowledgements
Developed by Universitat d'Alacant as part of its contribution to the [GoURMET](https://gourmet-project.eu/) project, which  received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825299.

## References
- Mikel Artetxe and Holger Schwenk, [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464) arXiv, Dec 26 2018.
- Mikel Artetxe and Holger Schwenk, [Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings](https://arxiv.org/abs/1811.01136), in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, Florence, Italy, 2019.
