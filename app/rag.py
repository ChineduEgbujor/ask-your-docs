# app/rag.py
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# —– Gemini setup (if used) —–
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# —– OpenAI setup (if used) —–
# openai.api_key = os.getenv("OPENAI_API_KEY")

def build_rag_prompt(docs: list[str], question: str) -> str:
    lines = [
        "Use these facts to answer the question as accurately as possible.",
        "Facts:"
    ]
    for i, doc in enumerate(docs, 1):
        snippet = doc.strip().replace("\n", " ")
        lines.append(f"{i}. {snippet[:1000]}")
    lines.append(f"\nQuestion: {question}\nAnswer:")
    return "\n".join(lines)

def generate_answer_with_gemini(prompt: str) -> str:
    try:
        resp = client.models.generate_content(
            model = "gemini-2.0-flash",
            contents=[prompt])
        return resp.text.strip()
    except Exception as e:
        print(f"Error generating answer with Gemini: {e}")
        return "Sorry, I couldn't generate an answer at this time."

# def generate_answer_with_openai(prompt: str) -> str:
#     resp = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user",   "content": prompt}
#         ],
#         temperature=0.2,
#         max_tokens=500,
#     )
#     return resp.choices[0].message.content.strip()
