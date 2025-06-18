# LLMs Deployment
The 4th course assignment of Introduction to Artificial Intelligence. Deploy and test the large language models on the ModelScope.
# Environment Construction
1. Starting the PAI-DSW CPU(8-core 32GB) environment and view Notebook.
2. Click the Terminal icon to open the terminal command line environment.
3. Installation foundation dependency in the directory `/mnt/workspace`.
```
pip install \
 "intel-extension-for-transformers==1.4.2" \
 "neural-compressor==2.5" \
 "transformers==4.33.3" \
 "modelscope==1.9.5" \
 "pydantic==1.10.13" \
 "sentencepiece" \
 "tiktoken" \
 "einops" \
 "transformers_stream_generator" \
 "uvicorn" \
 "fastapi" \
 "yacs" \
 "setuptools_scm"
```
# Download Large Language Models to Local
1. Switch to data directory.
```
cd /mnt/data
```
2. Download the corresponding large language models
- 2.1 Dowmload Qwen-7B-Chat.
   ```
   git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
   ```
- 2.2 Download ZhipuAI chatglm3-6b.
   ```
   git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
   ```
# Building model instances
1. Switch to working directory.
```
cd /mnt/workspace
```
2.  Write instance codes
- 2.1 Write the script run_qwen_cpu.py.
```
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
# Local path of model storage
model_name = "/mnt/data/Qwen-7B-Chat"
# Questions input to the large language model. It can be replaced by any of the five example questions
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少" 
tokenizer = AutoTokenizer.from_pretrained(
model_name,
trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
model_name,
trust_remote_code=True,
torch_dtype="auto"
).eval()
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```
- 2.2 Write the script chatglm.py.
```
import torch
from transformers import AutoTokenizer, AutoModel
# Local path of model storage
model_name = "/mnt/data/chatglm3-6b"   
prompt = "请你仔细理解这段两个人之间的对话话，回答一个问题：领导：你这是什么意思？小明：没什么意思，意思意思。领导：你这就不够意思了。小明：小意思，小意思。领导：你这人真有意思。小明：其实也没有别的意思。领导：那我就不好意思了。小明：是我不好意思。请问：以上“意思”分别是什么意思。"
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, eos_token=None, pad_token=None, unk_token=None)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()
# Check if there is a GPU, use GPU if there is, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
# Encode input text
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
# Set generation parameters
generation_config = model.generation_config
generation_config.max_length = 2048
generation_config.temperature = 0.7
generation_config.top_p = 0.95
# Perform reasoning
try:
    response, history = model.chat(tokenizer, prompt, history=None)
    print(response)
except Exception as e:
    print(f"Error generating response: {e}")
```
# Run Instance
Enter the working directory `/mnt/workplace`. And run the instance.
- 1. run Qwen-7B-Chat instance.
```
python run_qwen_cpu.py
```
- 2. run ChatGLM3-6 instance.
```
python chatglm.py
```
# Horizontal Comparative Analysis of LLMs
According to the five sample questions and the output of two big language models (Wisdom Spectrum ChatGLM3-6B and Tongyi Qianwen Qwen-7B-Chat), we compare the two big language models horizontally.
- **Qwen-7B-Chat:**
  - Advantages: The answer is straightforward and natural.
  - Inadequate: Limited ability to handle complex logic and insufficient attention to detail processing.
- **ChatGLM3-6:**
  - Advantages: The answer is in-depth and logical.
  - Inadequate: The answer is lengthy and the language is rigid.

For more details, please read the report hw4_2354283_黄迪.docx











