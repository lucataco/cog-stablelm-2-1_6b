# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, ConcatenateIterator
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "stabilityai/stablelm-2-1_6b"
MODEL_CACHE = "./checkpoints"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype="auto",
            cache_dir=MODEL_CACHE,
        )
        self.model.cuda()

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="The weather is always wonderful"),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate", default=64),
        temperature: float = Input(description="Temperature for sampling", default=0.7),
        top_p: float = Input(description="Top p for sampling", default=0.95),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        for token in self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )[0]:
            yield self.tokenizer.decode(token, skip_special_tokens=True)
