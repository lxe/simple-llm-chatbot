### Simple-LLM-Chatbot

A minimalistic chatbot script to experiment with LLMs. 

![./screenshot.png](https://raw.githubusercontent.com/lxe/simple-llm-chatbot/master/screenshot.png)

### Usage

0. Be on a machine with an NVIDIA card with 12-24 GB of VRAM.

1. Get the environment ready

```bash
conda create -n llm-playground python=3.10
conda activate llm-playground
conda install -y cuda -c nvidia/label/cuda-11.7.0
conda install -y pytorch=2 pytorch-cuda=11.7 -c pytorch
```

2. If on WSL, help bitsandbytes understand where to grab the dynamic libs from

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
```

3. Install the requirements

```bash
pip install -r requirements.txt
```

4. Run the script

```bash
python app.py
```

```
usage: app.py [-h] [-s]

Chatbot Demo

options:
  -h, --help   show this help message and exit
  -s, --share  Enable sharing of the Gradio interface
```

### License

MIT
