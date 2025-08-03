import collections
import json
import subprocess
import sys
import transformers
import os
import time
from typing import Any, Dict, List, Literal, NamedTuple
import warnings

import requests
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

try:
    # run which vllm
    vllm_loc = subprocess.run(["which", "vllm"], capture_output=True, text=True).stdout.strip()
    sys.path.append(os.path.dirname(vllm_loc))
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    warnings.warn("vLLM not available. HFMetaWrapper will fall back to transformers.")
    LLM = None
    SamplingParams = None

Reponse = collections.namedtuple("Response", ["content"])


class HUITMistral:
    """
    Custom chat model for a HUIT AWS Bendrock endpoint.
    **Only for internal use at Harvard.
    """

    def __init__(
        self,
        model: str = "mistral.mistral-large-2407-v1:0",
        max_attempts: int = 3,
        wait_time_between_attempts: int = 60,
    ):
        metadata = {}
        metadata["endpoint_url"] = "https://go.apis.huit.harvard.edu/ais-bedrock-llm/v1"
        metadata["api_key"] = os.getenv("HUIT_AI_API_KEY")
        self.model = model
        self.metadata = metadata
        self.max_attempts = max_attempts
        self.wait_time_between_attempts = wait_time_between_attempts

    @property
    def max_tokens_key(self) -> str:
        return "max_tokens"

    def invoke(
        self,
        messages: List[Dict[Literal["role", "content"], str]],
        max_tokens: int = 100,
        temperature: float = 0.5,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> NamedTuple:
        # 1. Convert from LangChain messages -> the custom format
        aws_style_messages = []
        for msg in messages:
            aws_style_messages.append(
                {
                    "role": msg["role"],
                    "content": [{"type": "text", "text": str(msg["content"])}],
                }
            )

        # 2. Construct the payload
        payload = json.dumps(
            {
                "modelId": self.model,
                "contentType": "application/json",
                "accept": "application/json",
                "body": {
                    "messages": aws_style_messages,
                    # "prompt": "hello",
                    "temperature": temperature,
                    "top_p": top_p,
                    self.max_tokens_key: max_tokens,
                },
            }
        )

        # 3. Send the request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.metadata["api_key"],
        }

        attempts = 0
        while True:
            attempts += 1
            if attempts > self.max_attempts:
                raise RuntimeError("Failed to get a response from the endpoint.")
            try:
                response = requests.post(
                    self.metadata["endpoint_url"],
                    headers=headers,
                    data=payload,
                )
                response.raise_for_status()
                print(response.text)
                break
            except requests.exceptions.HTTPError as e:
                print(e)
                warnings.warn(f"Attempt {attempts} failed: {e}")
                time.sleep(self.wait_time_between_attempts)

        # 4. Parse the response
        #    Adjust based on how your custom endpoint returns the data
        #    For example, maybe the response has a field "generated_text":
        # content = result_json.get("generated_text", "")
        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"]

        return Reponse(content=content)


def llama_prompt_from_messages(
    messages: List[Dict[Literal["role", "content"], str]]
) -> str:
    """
    Formats OpenAI-style messages for Llama 3.

    Args:
        messages (list): List of dictionaries representing messages, each with 'role'
                        and 'content' keys.

    Returns:
        str: Formatted string for Llama 3.
    """
    # Initialize the prompt with the begin_of_text token
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # prompt = ""
    # for m in messages:
    #     prompt += (
    #         f"<|start_header_id|>{m['role']}<|end_header_id|>{m['content']}\n<|eot_id|"
    #     )
    # prompt += "<|start_header_id|>assistant<|end_header_id|>"
    prompt = prompt.replace("(", "[")
    prompt = prompt.replace(")", "]")
    return prompt


class HUITMeta:
    """
    Custom chat model for a HUIT AWS Bendrock endpoint.
    **Only for internal use at Harvard.
    """

    def __init__(
        self,
        model: str = "meta.llama3-1-8b-instruct-v1:0",
        max_attempts: int = 3,
        wait_time_between_attempts: int = 60,
    ):
        metadata = {}
        metadata["endpoint_url"] = "https://go.apis.huit.harvard.edu/ais-bedrock-llm/v1"
        metadata["api_key"] = os.getenv("HUIT_AI_API_KEY")
        self.model = model
        self.metadata = metadata
        self.max_attempts = max_attempts
        self.wait_time_between_attempts = wait_time_between_attempts

    @property
    def max_tokens_key(self) -> str:
        return "max_gen_len"

    # @staticmethod
    # def prompt_from_messages(
    #     messages: List[Dict[Literal["role", "content"], str]]
    # ) -> str:
    #     """
    #     Formats OpenAI-style messages for Llama 3.

    #     Args:
    #         messages (list): List of dictionaries representing messages, each with 'role'
    #                         and 'content' keys.

    #     Returns:
    #         str: Formatted string for Llama 3.
    #     """
    #     # Initialize the prompt with the begin_of_text token
    #     prompt = ""
    #     for m in messages:
    #         prompt += f"<|start_header_id|>{m['role']}<|end_header_id|>{m['content']}<|eot_id|>"
    #     prompt += "<|start_header_id|>assistant<|end_header_id|>"
    #     prompt = prompt.replace("(", "[")
    #     prompt = prompt.replace(")", "]")
    #     return prompt

    def invoke(
        self,
        messages: List[Dict[Literal["role", "content"], str]],
        max_tokens: int = 100,
        temperature: float = 0.5,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> NamedTuple:
        # 1. Convert from LangChain messages -> the custom format
        prompt = llama_prompt_from_messages(messages)

        # 2. Construct the payload
        payload = json.dumps(
            {
                "modelId": self.model,
                "contentType": "application/json",
                "accept": "application/json",
                "body": {
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    self.max_tokens_key: max_tokens,
                },
            }
        )

        # 3. Send the request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.metadata["api_key"],
        }

        attempts = 0
        while True:
            attempts += 1
            if attempts > self.max_attempts:
                raise RuntimeError("Failed to get a response from the endpoint.")
            try:
                response = requests.post(
                    self.metadata["endpoint_url"],
                    headers=headers,
                    data=payload,
                )
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                print(e)
                warnings.warn(f"Attempt {attempts} failed: {e}")
                time.sleep(self.wait_time_between_attempts)

        # 4. Parse the response
        #    Adjust based on how your custom endpoint returns the data
        #    For example, maybe the response has a field "generated_text":
        # content = result_json.get("generated_text", "")
        result_json = response.json()
        content = result_json["generation"]
        return Reponse(content=content)


class HUITOpenAI:
    """
    Custom chat model for a HUIT OpenAI endpoint.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_attempts: int = 3,
        wait_time_between_attempts: int = 60,
    ):
        metadata = {}
        metadata["endpoint_url"] = (
            "https://go.apis.huit.harvard.edu/ais-openai-direct/v1/chat/completions"
        )
        metadata["api_key"] = os.getenv("HUIT_AI_API_KEY")
        self.model = model.replace("-huit", "")
        self.metadata = metadata
        self.max_attempts = max_attempts
        self.wait_time_between_attempts = wait_time_between_attempts

    def invoke(
        self,
        messages: List[Dict[Literal["role", "content"], str]],
        max_tokens: int = 100,
        temperature: float = 0.5,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> NamedTuple:
        # 1. Construct the payload
        payload_dict = {
            "model": self.model,
            "messages": messages,
        }
        if not self.model.startswith("o3"):
            payload_dict["max_tokens"] = max_tokens
            payload_dict["temperature"] = temperature
            payload_dict["top_p"] = top_p

        payload = json.dumps(payload_dict)

        headers = {
            "Content-Type": "application/json",
            "api-key": self.metadata["api_key"],
        }

        # 2. Send the request
        attempts = 0
        while True:
            attempts += 1
            if attempts > self.max_attempts:
                raise RuntimeError("Failed to get a response from the endpoint.")
            try:
                response = requests.post(
                    self.metadata["endpoint_url"],
                    headers=headers,
                    data=payload,
                )
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                warnings.warn(f"Attempt {attempts} failed: {e}")
                time.sleep(self.wait_time_between_attempts)

        # 3. Parse the response
        result_json = response.json()
        # print(result_json)
        content = result_json["choices"][0]["message"]["content"]

        return Reponse(content=content)


class HFMetaWrapper:
    def __init__(
        self,
        model_name: str,
        use_vllm: bool = True,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        dtype: str = "auto",
    ):
        """
        Wrapper for Hugging Face models with vLLM optimization support.
        
        Args:
            model_name: The model name/path from Hugging Face
            use_vllm: Whether to use vLLM for inference (more efficient)
            gpu_memory_utilization: GPU memory utilization for vLLM
            max_model_len: Maximum model length for vLLM
            dtype: Data type for model weights
        """
        self.model_name = model_name
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        
        # Load tokenizer for prompt formatting (needed regardless of vLLM vs transformers)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
        if self.use_vllm:
            self.llm = LLM(
                model=model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=dtype,
                trust_remote_code=True,
            )
        else:
            self._init_transformers()
    
    def _init_transformers(self):
        """Fallback initialization using transformers."""
        self.llm = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.device = next(self.llm.parameters()).device
        
        # prepare model so that there are no gradients
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

    def _format_prompt(self, messages: List[Dict[Literal["role", "content"], str]]) -> str:
        """Format messages using the appropriate tokenizer's chat template."""
        try:
            # Use the model's own tokenizer to format the prompt
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            return prompt
        except Exception as e:
            warnings.warn(f"Failed to use chat template for {self.model_name}: {e}. Using fallback formatting.")
            # Fallback to simple concatenation
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "assistant: "
            return prompt

    def invoke(
        self,
        messages: List[Dict[Literal["role", "content"], str]],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 0.9,
        n: int = 1,  # Number of generations per prompt
        **kwargs: Any,
    ) -> NamedTuple:
        if self.use_vllm:
            return self._invoke_vllm(messages, max_tokens, temperature, top_p, n, **kwargs)
        else:
            return self._invoke_transformers(messages, max_tokens, temperature, top_p, n, **kwargs)
    
    def _invoke_vllm(
        self,
        messages: List[Dict[Literal["role", "content"], str]],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 0.9,
        n: int = 1,
        **kwargs: Any,
    ) -> NamedTuple:
        """vLLM-based inference with support for multiple generations."""
        prompt = self._format_prompt(messages)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,  # Number of generations per prompt - this is the key!
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        
        if n == 1:
            # Single generation - return content as string for backward compatibility
            content = outputs[0].outputs[0].text
        else:
            # Multiple generations - return list of strings
            content = [output.text for output in outputs[0].outputs]
        
        return Reponse(content=content)
    
    def _invoke_transformers(
        self,
        messages: List[Dict[Literal["role", "content"], str]],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 0.9,
        n: int = 1,
        **kwargs: Any,
    ) -> NamedTuple:
        """Fallback transformers-based inference with support for multiple generations."""
        prompt = self._format_prompt(messages)

        # use the hf generate method
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,  # Only sample if temperature > 0
            num_return_sequences=n,  # This is the key parameter!
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # remove the input tokens from the outputs
        outputs = outputs[:, inputs["input_ids"].shape[-1] :]
        
        if n == 1:
            # Single generation - return content as string for backward compatibility
            content = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Multiple generations - return list of strings
            content = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        return Reponse(content=content)


class HUITEmbeddings:
    """
    Custom embeddings model for a HUIT AWS Bendrock endpoint.
    **Only for internal use at Harvard.
    """

    def __init__(
        self,
        model: str = "mistral.mistral-large-2407-v1:0",
        max_attempts: int = 3,
        wait_time_between_attempts: int = 60,
    ):
        self.metadata = {}
        self.metadata["endpoint_url"] = (
            "https://go.apis.huit.harvard.edu/ais-openai-direct/v1/embeddings"
        )
        self.metadata["api_key"] = os.getenv("HUIT_AI_API_KEY")
        self.max_attempts = max_attempts
        self.wait_time_between_attempts = wait_time_between_attempts
        self.model = model

    def invoke(self, input) -> NamedTuple:
        # 1. Construct the payload
        payload = json.dumps(
            {
                "contentType": "application/json",
                "accept": "application/json",
                "body": {
                    "model": self.model,
                    "input": input,
                },
            }
        )

        # 3. Send the request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.metadata["api_key"],
        }

        attempts = 0
        while True:
            attempts += 1
            if attempts > self.max_attempts:
                raise RuntimeError("Failed to get a response from the endpoint.")
            try:
                response = requests.post(
                    self.metadata["endpoint_url"],
                    headers=headers,
                    data=payload,
                )
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                warnings.warn(f"Attempt {attempts} failed: {e}")
                time.sleep(self.wait_time_between_attempts)

        # 4. Parse the response
        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"]

        return Reponse(content=content)


ModelAPIDict = {
    "google/gemma-2b-it": ChatTogether,
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": ChatTogether,
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ChatTogether,
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ChatTogether,
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ChatTogether,
    "meta-llama/Llama-3.2-1B-Instruct-Turbo": ChatTogether,
    "meta.llama3-1-8b-instruct-v1:0": HUITMeta,
    "meta.llama3-1-70b-instruct-v1:0": HUITMeta,
    "meta.llama3-2-1b-instruct-v1:0": HUITMeta,
    "meta.llama3-2-3b-instruct-v1:0": HUITMeta,
    "meta.llama3-2-11b-instruct-v1:0": HUITMeta,
    "meta.llama3-3-70b-instruct-v1:0": HUITMeta,
    "mistral.mistral-large-2407-v1:0": HUITMistral,
    "gpt-4o-mini-huit": HUITOpenAI,
    "gpt-4o-mini": ChatOpenAI,
    "o3-mini-huit": HUITOpenAI,
    "nbd22/Llama-3.1-8B-Instruct-GRPO-gsm8k-ft-lora": HFMetaWrapper,
    "AlistairPullen/llama-3.1-8B-grpo": HFMetaWrapper,
    # New models with vLLM support
    "Qwen/Qwen2.5-0.5B-Instruct": HFMetaWrapper,
    "meta-llama/Llama-3.2-1B-Instruct": HFMetaWrapper,
    "meta-llama/Llama-3.2-3B-Instruct": HFMetaWrapper,
    "meta-llama/Llama-3.1-8B-Instruct": HFMetaWrapper,
}

ValidLLMs = Literal[
    "google/gemma-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "meta-llama/Llama-3.2-1B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "nbd22/Llama-3.1-8B-Instruct-GRPO-gsm8k-ft-lora",
    "AlistairPullen/llama-3.1-8B-grpo",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-2-3b-instruct-v1:0",
    "meta.llama3-2-1b-instruct-v1:0",
    "meta.llama3-2-11b-instruct-v1:0",
    "meta.llama3-3-70b-instruct-v1:0",
    "mistral.mistral-large-2407-v1:0",
    "gpt-4o-mini-huit",
    "gpt-4o-mini",
    "o3-mini-huit",
    # New models with vLLM support
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct"
]


def invoke_with_retries(
    model: ValidLLMs,
    messages: List[Dict[Literal["role", "content"], str]],
    *args,
    max_attempts: int = 10,
    wait_time_between_attempts: int = 60,
    temperature=0.5,
    max_tokens=100,
    **kwargs,
):
    attempts = 0
    while True:
        if attempts > max_attempts:
            raise RuntimeError("Failed to get a response from the endpoint.")
        try:
            attempts += 1
            result = model.invoke(
                messages, *args, temperature=temperature, max_tokens=max_tokens, **kwargs
            )
            return result
        except Exception as e:
            warnings.warn(f"Attempt {attempts} failed: {e}")
            time.sleep(wait_time_between_attempts)


def get_llm_api(model: ValidLLMs, **kwargs) -> Any:
    api = ModelAPIDict.get(model)
    if api is None:
        raise ValueError(f"Model {model} not supported.")
    elif api is HFMetaWrapper:
        # Use the new vLLM-based wrapper
        return api(model_name=model, **kwargs)
    else:
        return api(model=model, **kwargs)


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Be helpful and concise."},
        {"role": "user", "content": "What's the capital of France?"},
    ]

    # Test the new vLLM-based wrapper with one of the new models
    print("Testing vLLM-based HFMetaWrapper...")

    # llm = get_llm_api("Qwen/Qwen2.5-0.5B-Instruct", use_vllm=False)
    llm = get_llm_api("meta-llama/Llama-3.1-8B-Instruct", use_vllm=True)
    result = llm.invoke(messages, max_tokens=50, n=2, temperature=0.5, do_sample=True)
    print(f"Result: {result.content}")

    # # Original tests (commented out)
    # llm = HUITMistral("mistral.mistral-large-2407-v1:0")
    # result = llm.invoke(messages, max_tokens=10)
    # print(result.content)

    # embedder = HUITEmbeddings("text-embedding-3-large")
    # input = "Hello"
    # result = embedder.invoke(input)
    # print(result.content)
