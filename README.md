# Sharpness-Aware Minimization for Topic Models with High-Quality Document Representations

## Setup

1. Install the required libraries:
    ```bash
    numpy==1.26.4
    torch_kmeans==0.2.0
    pytorch==2.2.0
    sentence_transformers==2.2.2
    scipy==1.10
    bertopic==0.16.0
    gensim==4.2.0
    ```
2. Install Java and download [this JAR file](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to `./evaluations/palmetto.jar`.
3. Download and extract [this Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to `./datasets/wikipedia/`.

## Usage

To run the model, use the command:

```bash
python main.py --model <MODEL_NAME> --dataset <DATASET_NAME> --num_topics 50 --beta_temp 0.15 --num_groups 20 --epochs 200 --device cuda --lr 0.002 --use_pretrainWE --use_SAM <USE_SAM> --SAM_name <SAM_METHOD>
```

## Options:

- **Models**: `ETM`, `ECRTM`, `NeuroMax`, `FASTopic`
- **Datasets**: `20NG`, `AGNews`, `YahooAnswers`, `SearchSnippets`, `GoogleNews`
- To **not use SAM**, set `use_SAM=0`.
- To **use SAM**, set `use_SAM=1` and specify `SAM_name` from: `SAM`, `FSAM`, `DREAM`.


## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for the evaluation of topic coherence.