# generate.py  (final version - compatible with openai>=1.0)
import os
from openai import OpenAI
from search import retrieve, rerank

# Initialize OpenAI client (reads OPENAI_API_KEY from environment)
client = OpenAI(api_key="<YOUR_API_KEY>")

PROMPT_SYSTEM = """You are Mr.HelpMate, an assistant that answers questions strictly using provided policy chunks.
Rules:
- Use only the provided chunks; do not hallucinate.
- For each factual claim, include a bracketed citation like [chunk_id:chunk_12] and page number if available.
- If the answer is not present in the chunks, reply: "Answer not found in supplied policy."
- Provide a short summary (2–4 lines) and a bullet list of references used.
"""

FEW_SHOT = """
Example 1:
Q: What is the scheduled benefit for all members?
A: The scheduled benefit is $10,000 for all members. [chunk_id:chunk_37, Page: PART IV - Member Life Insurance Article 1]

Example 2:
Q: How do I file a claim?
A: Notice of claim must be sent within 20 days, and proof of loss within 90 days. See Article: Notice of Claim and Proof of Loss. [chunk_id:chunk_12, chunk_id:chunk_15]
"""

def make_prompt(query, top_chunks):
    """Combine retrieved chunks + examples into a single prompt."""
    context = "\n\n---\n".join([f"[{c['id']}] {c['text']}" for c in top_chunks])
    prompt = f"{PROMPT_SYSTEM}\n\n{FEW_SHOT}\n\nContext:\n{context}\n\nQ: {query}\nA:"
    return prompt

def answer_with_openai(query):
    """Retrieve, rerank, and generate an answer with GPT."""
    items = retrieve(query, top_k=20)
    top_r = rerank(query, items, top_n=5)
    prompt = make_prompt(query, top_r)

    print("🧠 Generating answer using GPT-4o-mini ...\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # change to gpt-4o if available
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()
    return answer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mr.HelpMate Answer Generator")
    parser.add_argument("--query", type=str, required=True, help="User query/question")
    args = parser.parse_args()

    result = answer_with_openai(args.query)
    print("\n================= FINAL ANSWER =================\n")
    print(result)
    print("\n================================================")
