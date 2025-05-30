from cog import BasePredictor, Input
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Predictor(BasePredictor):
    def setup(self):
        self.model_name = "./lyrics-finetuned"  # or your LoRA-tuned version
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()

    def predict(
        self,
        prompt: str = Input(description="Malayalam lyrics prompt"),
        max_new_tokens: int = Input(default=80, ge=10, le=300),
        temperature: float = Input(default=0.9),
        top_p: float = Input(default=0.95),
        top_k: int = Input(default=50),
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result[len(prompt):].strip()
