** Description **

Simple LoRA finetuning of GPT2 model.

** Installation **
`pip install requirements.txt`
If any trouble, please just install manually:
    * pytorch
    * transformers
    * peft
    * trl

Place your dataset into file: `censor_samples.txt`

** Usage **
Please modify config in `finetune_censor.py` to match your hardware capabilities.

To start finetunig, run:
`python finetune_censor.py`

To test immediate result of finetuning:
`python infer_lora.py`

To merge into baseline:
`python merge.py`

To run inference on the merged model:
`python start.py`
