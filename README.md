# Efficient-Empathy
In recent years, with the rapid advancements in large language models (LLMs), achieving excellent empathetic response capability has become a crucial prerequisite. Consequently, managing and understanding empathetic datasets has gained increasing importance. However, empathetic data are typically trained without any quality selection, leading to inefficient data usage and wasted computational resources. Additionally, using raw data can result in low performance in empathetic dialogues. In this work, we present Efficient-Empathy, a sensibility and rationality score-based data selection algorithm that automatically selects sensibility and rationality data while discarding low-quality data. With only the sensibility data (59\% of the full dataset), our trained sensibility model efficiently achieves state-of-the-art (SoTA) performance. Furthermore, with multiple data selection hyperparameters, the sensibility model demonstrates SoTA performance, showcasing the robustness of our method. By integrating sensibility and rationality data with a MoE structure, we achieve even higher performance, demonstrating the effectiveness of our Efficient-Empathy algorithm.


# Data
Sensibility Dataset $M_s$, Rationality Dataset $M_r$, Discard Dataset $M_d$ are located in `data` folder.

# Expert Training
Train domain expert based on LLaMA using the followling commands:
The domain expert will be saved on `train_llama/sft-llama/{expert_name}`.
```
cd train_llama
python sft_llama3.py
```


Train domain expert based on qwen using the followling commands:
The domain expert will be saved on `train_qwen/{expert_name}`.
```
cd tran_qwen
bash finetune_ds.sh
```


# Mixture-of-Expert
Before starting the merge, you need to convert the lora-style expert into a normal structure. For LLaMA3, we recommond to use LLaMA-Factory tool.

After that, Use repository `mergoo` to finish mixturing process,
The initail moe model saved on `examples/moe_model`.
```
cd mergoo/examples
python compose_llama.py
```

For Qwen MoE, we make our own implementation:

```
cd mergoo/examples
python compose_qwen.py
```


# MoE Training
For LLaMA3:
```
cd train_llama
python sft_llama3_moe.py
```

For Qwen:
```
cd tran_qwen
bash finetune_ds_moe.sh
```



# Empathetic Response Generation
```
cd empathy
python test_response_llama3.py
```

# Evalutaion
```
cd empathy
python resposne_eval.py
```




