import json
import re
import logging
from groq import Groq
from langchain_core.prompts import PromptTemplate
from ..config.settings import GROQ_API_KEY, CHAT_MODEL

class LLMService:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = CHAT_MODEL

    def _extract_answer(self, input_string: str, pattern: str = None) -> dict:
        """Extract JSON answer from LLM response."""
        json_start = input_string.find("{")
        json_end = input_string.rfind("}") + 1

        if json_start == -1 or json_end == -1:
            raise ValueError("Invalid input: No JSON data found.")

        json_data = input_string[json_start:json_end]

        try:
            return json.loads(json_data)
        except json.JSONDecodeError:
            if not pattern:
                pattern = r'{"Answer":\s*".*?"}'

            match = re.search(pattern, input_string, re.DOTALL)
            if match:
                return json.loads(match.group())
            else:
                logging.error("No dictionary with the specified pattern found in LLM response")
                return {"error": "No dictionary with the specified pattern found"}

    def generate_completion(self, prompt: str, pattern: str = None) -> dict:
        """Generate completion using Groq API."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stream=True
        )

        answer = ""
        for chunk in completion:
            answer += chunk.choices[0].delta.content or ""
        
        return self._extract_answer(answer, pattern)

    def enhance_query(self, query: str, query_rewriter_prompt: str) -> str:
        """Enhance the user query using the query rewriter prompt."""
        prompt = PromptTemplate(
            template=query_rewriter_prompt,
            input_variables=["user_query"]
        )
        final_prompt = prompt.format(user_query=query.lower())
        response = self.generate_completion(final_prompt, pattern=r'{"Query":\s*".*?"}')
        return response.get("Query", query)

    def generate_answer(self, query: str, context: str, answer_prompt: str) -> str:
        """Generate answer based on context and query."""
        prompt = PromptTemplate(
            template=answer_prompt,
            input_variables=["context", "question"]
        )
        final_prompt = prompt.format(context=context, question=query)
        response = self.generate_completion(final_prompt)
        return response.get("Answer", "Sorry, I couldn't generate an answer.") 