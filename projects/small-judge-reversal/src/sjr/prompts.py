QUESTIONS = {
    "better": "Which response is better?",
    "W1": "Which response is worse?",
    "W2": "Which response should be rejected?",
    "length": "Which response is longer?",
}
TEMPLATE = ("You are comparing two responses to the same user request.\n\n"
            "[User request]\n{prompt}\n\n[Response A]\n{a}\n\n[Response B]\n{b}\n\n"
            "{question} Answer with a single letter: A or B.")

def user_text(prompt, a, b, criterion):
    return TEMPLATE.format(prompt=prompt, a=a, b=b, question=QUESTIONS[criterion])

def chat_input(tok, prompt, a, b, criterion, is_qwen3):
    msgs = [{"role": "user", "content": user_text(prompt, a, b, criterion)}]
    kw = {"enable_thinking": False} if is_qwen3 else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
