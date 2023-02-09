# Bounding the Capabilities of Large Language Models in Open Text Generation with Prompt Constraints

This repo contains codes for the following paper:

_Albert Lu*, Hongxin Zhang*, Yanzhe Zhang, Xuezhi Wang, Diyi Yang_: Bounding the Capabilities of Large Language Models in Open Text Generation with Prompt Constraints, EACL 2023 Findings

If you would like to refer to it, please cite the paper mentioned above.

## Prompts and Generations

We provide all our created 288 prompts and 3000+ generated responses from OpenAI text-davinci-002, OPT, BLOOM, GLM ... in the following structure for better future utilization. For the full prompt list, please refer to our paper and the `prompts.txt` file under each folder.

```
|__ Structural/
        |__ Descriptive/
                prompts.txt
        |__ Formatting/
                |__ Code/
                        prompts.txt    
                |__ Email/
                        prompts.txt
                |__ Paper/
                        prompts.txt
        |__ Numerical/
                |__ babbage/
                |__ curie/
                |__ t0/
                |__ t0.4/
                |__ t0.9/
                |__ template2/
                prompts.txt
|__ Stylistic/
        |__ Literary Style/
                |__ Mood/
                |__ Tone/
                |__ Writing Style/
                prompts.txt
        |__ Story Generation/
                |__ Characterization/
                |__ Genre/
                |__ Pacing/
                |__ Plot/
                prompts.txt
|__ Sensitivity/
        |__ ada/
        |__ babbage/
        |__ curie/
        |__ t0.4/
        |__ t1/
        |__ BLOOM/
        |__ GLM/
        |__ OPT/
        prompts.txt                
|__ Mitigation/
        |__ definition/
        |__ demonstration/
        |__ explanation/
        prompts.csv
        
                
```

## Run experiments and analysis

To run your own experiments on custom prompts or parameters, use the `run.py`. For example

    python3 run.py --n 10 --verbose --output_file Structural/Numerical/1 --prompt "Write 10 words about bananas:"

There are several ways for you to provide the prompts, for more details, please refer to the code.

To run analysis on collected generations, please refer to the code in `MTurk` and `utils.py`.