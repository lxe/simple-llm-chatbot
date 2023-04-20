import torch
import gradio as gr
import transformers
import traceback
import argparse

from queue import Queue
from threading import Thread
import gc

CUDA_AVAILABLE = torch.cuda.is_available()

DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

MODEL_NAME = 'stabilityai/stablelm-tuned-alpha-7b'

DEFAULT_SYSTEM_PROMPT = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    load_in_8bit=True, 
    torch_dtype=torch.float16,

    # Pin to single device if CUDA is available for colab
    device_map={'':0} if CUDA_AVAILABLE else 'auto',
)

class StopOnTokens(transformers.StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


# Streaming functionality taken from https://github.com/oobabooga/text-generation-webui/blob/master/modules/text_generation.py#L105
class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False

class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """
    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc=func
        self.c_callback=callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                traceback.print_exc()
                pass
            except:
                traceback.print_exc()
                pass

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True,None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()

def clear_torch_cache():
    gc.collect()
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()

def generate_text(
    history,  
    max_new_tokens, 
    do_sample, 
    temperature, 
    top_p, 
    top_k, 
    repetition_penalty, 
    typical_p, 
    num_beams,
    system_prompt=DEFAULT_SYSTEM_PROMPT
):
    # Create a conversation context of the last 4 entries in the history
    inp = system_prompt + ''.join([
        f"<|USER|>{h[0]}<|ASSISTANT|>{'' if h[1] is None else h[1]}\n" for h in history[-6:]
    ]).strip()

    print(inp)
     
    input_ids = tokenizer.encode(
        inp, 
        return_tensors='pt', 
        add_special_tokens=False
    ).to(DEVICE) # type: ignore

    generate_params = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "typical_p": typical_p,
        "num_beams": num_beams,
        "stopping_criteria": transformers.StoppingCriteriaList(),
        "pad_token_id": tokenizer.eos_token_id,
    }

    def generate_with_callback(callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        kwargs['stopping_criteria'].append(StopOnTokens())
        clear_torch_cache()
        with torch.no_grad():
            model.generate(**kwargs) # type: ignore

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            new_tokens = len(output) - len(input_ids[0])
            reply = tokenizer.decode(output[-new_tokens:], skip_special_tokens=True)

            # if reply contains 'EOS' then we have reached the end of the conversation
            if output[-1] in [tokenizer.eos_token_id]:
                yield history
                break

            history[-1][1] = reply.strip()
            yield history

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(value="How old is the Earth?", placeholder="Type a message...")
            with gr.Row():
                clear = gr.Button("Clear")

        with gr.Column():
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, placeholder="System prompt")
            max_new_tokens = gr.Slider(0, 4096, 200, step=1, label="max_new_tokens")
            do_sample = gr.Checkbox(True, label="do_sample")
            with gr.Row():
                with gr.Column():
                    temperature = gr.Slider(0, 2, 0.1, step=0.01, label="temperature")
                    top_p = gr.Slider(0, 1, 0.15, step=0.01, label="top_p")
                    top_k = gr.Slider(0, 100, 40, step=1, label="top_k")
                with gr.Column():
                    repetition_penalty = gr.Slider(0, 10, 1.1, step=0.01, label="repetition_penalty")
                    typical_p = gr.Slider(0, 1, 1, step=0.01, label="typical_p")
                    num_beams = gr.Slider(0, 10, 1, step=1, label="num_beams")

    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def fix_history(history):
        update_history = False
        for i, (user, bot) in enumerate(history):
            if bot is None:
                update_history = True
                history[i][1] = "_silence_"
        if update_history:
            chatbot.update(history) 

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        generate_text, inputs=[
            chatbot,
            max_new_tokens, 
            do_sample, 
            temperature, 
            top_p, 
            top_k, 
            repetition_penalty, 
            typical_p, 
            num_beams,
            system_prompt
        ], outputs=[chatbot],
    ).then(fix_history, chatbot)

    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chatbot Demo")
    parser.add_argument("-s", "--share", action="store_true", help="Enable sharing of the Gradio interface")
    args = parser.parse_args()

    demo.queue().launch(share=args.share)