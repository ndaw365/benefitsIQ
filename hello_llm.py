from llm import ask

PROMPT = "Explain what an elimination period is in group disability insurance."

if __name__ == "__main__":
    for temp in (0.0, 1.0):
        print(f"\n===== temperature={temp} =====")
        for i in range(3):
            text, tin, tout = ask(PROMPT, temperature=temp)
            print(f"\n--- run {i+1} | in={tin} out={tout} ---")
            print(text[:400])

            