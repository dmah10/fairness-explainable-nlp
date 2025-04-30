# fairness-xai

To reproduce the experiments, first create an environment with 
```bash
conda env create --file=environment.yml
```

To run the main disparity experiments, create a project on wandb and run the following command:
```bash
python main.py --datasets gecobench gecobench-subject --models facebook/FairBERTa huawei-noah/TinyBERT_General_4L_312D nlptown/bert-base-multilingual-uncased-sentiment openai-community/gpt2 --train --epochs 50 --wandb <project-name>
```
Then the created csv files contain all the evaluation scores for all the explanations. Finally follow the `analysis` notebook to download the csv files from wandb and generate tables similar to the ones in the paper. 