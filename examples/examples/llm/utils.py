from pathlib import Path
from timeit import default_timer
from typing import Literal, Any, Dict, Optional, Union

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# import fire
from llama_cpp import Llama


class HFModel:
    def __init__(
        self,
        model_name: str,
        model_class: type,
        device: str = "cuda",
    ):
        self.device = device  # the device to load the model onto
        self.model = model_class.from_pretrained(model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )


class CausalShell(HFModel):
    def __init__(
        self, model_name: str = "Qwen/Qwen2-7B-Instruct", device: str = "cuda"
    ):
        super().__init__(model_name, AutoModelForCausalLM, device)

    def generate(self, prompt: str, generation_args: Optional[Dict[str, Any]] = None):
        upd_args = generation_args or {}
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.7,
            "do_sample": True,
            # "num_beams": 3,
        }

        generation_args.update(upd_args)

        output = self.pipe(prompt, **generation_args)
        return output[0]["generated_text"]


class T5Shell(HFModel):
    def __init__(
        self, model_name="utrobinmv/t5_translate_en_ru_zh_large_1024_v2", device="cuda"
    ):
        super().__init__(model_name, T5ForConditionalGeneration, device=device)
        self.model.eval()
        self.model.to(self.device)

    def translate(self, text, to: Literal["ru", "en", "zh", "sbl"]):
        prefix = f"translate to {to}: "
        input_ids = self.tokenizer(prefix + text, return_tensors="pt")
        generated_tokens = self.model.generate(**input_ids.to(self.device))
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


class Translator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def translate(
        self, from_lng: str, to_lng: str, text: str, max_length: int = 300
    ) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang=from_lng)
        inputs = tokenizer(text, return_tensors="pt")
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[to_lng],
            max_length=max_length,
        )
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


class SaigaShell:
    SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

    def __init__(
        self, model_path: Union[Path, str], n_ctx=8192, verbose=False, n_threads=None
    ):
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_parts=1,
            verbose=verbose,
            n_gpu_layers=-1,
            n_threads=n_threads,
        )

    def generate(
        self, prompt: str, generation_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        generation_args = generation_args or dict(
            top_k=30,
            top_p=0.9,
            temperature=0.6,
            repeat_penalty=1.1,
        )
        a = default_timer()
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        parts = self.model.create_chat_completion(
            messages,
            **generation_args,
        )
        return {
            "n_tokens": parts["usage"]["total_tokens"],
            "time": default_timer() - a,
            "generated_text": parts["choices"][0]["message"]["content"],
        }
