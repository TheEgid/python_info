import json
import os
import textwrap
from typing import Dict, List, Optional, Union

import requests


class LLMService:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        # self.payed_model = "anthropic/claude-haiku-4.5"
        self.payed_model = "z-ai/glm-4.5-air:free"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://n8n-seo.space",
            "X-Title": "My optim",
        }

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        payload: Dict[str, Union[str, float, int, list]] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter API error: {e}") from e

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> Union[str, Dict]:
        """Универсальный метод генерации ответа"""
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages, model or self.payed_model, temperature, max_tokens)
        content = response["choices"][0]["message"]["content"]
        return json.loads(content) if json_mode else content

    def generate_paid_response(self, prompt: str, temperature: float = 0.3, max_tokens: int = 25000) -> str:
        """Упрощённый вызов для платной модели"""
        return self.generate(prompt, model=self.payed_model, temperature=temperature, max_tokens=max_tokens)

    def summarize_texts(
        self,
        texts: List[str],
        role: str = "space exploration expert",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 8000,
    ) -> str:
        """
        Объединяет входные тексты и запрашивает у модели сводку или анализ.
        """
        text_input = "\n".join(texts)
        prompt = f"Please summarize or elaborate on the following content:\n{text_input}"

        messages = [
            {"role": "system", "content": f"You are {role}."},
            {"role": "assistant", "content": "You can read the input and answer in detail."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._make_request(
                messages=messages,
                model=model or self.payed_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response["choices"][0]["message"]["content"].strip()
            self.print_formatted_response(content)
            return content
        except Exception as e:
            error_message = f"❌ Error during summarization: {str(e)}"
            print(error_message)
            return error_message

    def print_formatted_response(self, response: str) -> None:
        """
        Выводит текст красиво, разбивая на параграфы по двойным переносам.
        """
        paragraphs = response.split("\n\n")  # Разделяем по пустым строкам
        wrapper = textwrap.TextWrapper(width=80)

        print("\nText Response:")
        print("--------------------")
        for para in paragraphs:
            para = para.strip()
            if para:
                print(wrapper.fill(para))
                print()  # Пустая строка между параграфами
        print("--------------------\n")
