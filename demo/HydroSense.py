import gradio as gr
import time

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Edu+AU+VIC+WA+NT+Dots:wght@400..700&family=Give+You+Glory&family=Sofia&family=Sunshiney&family=Vujahday+Script&display=swap');
.gradio-container, .gradio-container * {
   font-family: "Edu AU VIC WA NT Dots", cursive;
  font-optical-sizing: auto;
  font-weight: <weight>;
  font-style: normal;
}
"""
js = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') === 'dark') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

system_prompt = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

code = """
```python
from huggingface_hub import InferenceClient
SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
PROMPT = "{PROMPT}"
MODEL_NAME = "meta-llama/Meta-Llama-3-70b-Instruct"  # or "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO" or "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1"
messages = [
    {"role": "system", "content": SYSTEM_PROMPT}, 
    {"role": "user", "content": PROMPT}
]
client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)
for c in client.chat_completion(messages, max_tokens=200, stream=True):
    token = c.choices[0].delta.content
    print(token, end="")
```
"""

# ip_requests = {}
# ip_requests_lock = threading.Lock()

# def allow_ip(request: gr.Request, show_error=True):
#     ip = request.headers.get("X-Forwarded-For")
#     now = datetime.now()
#     window = timedelta(hours=24)
#     with ip_requests_lock:
#         if ip in ip_requests:
#             ip_requests[ip] = [timestamp for timestamp in ip_requests[ip] if now - timestamp < window]
#         if len(ip_requests.get(ip, [])) >= 15:
#             raise gr.Error("Rate limit exceeded. Please try again tomorrow or use your Hugging Face Pro token.", visible=show_error)
#         ip_requests.setdefault(ip, []).append(now)
#         print("ip_requests", ip_requests)
#     return True

# def inference(prompt, hf_token, model, model_name):
#     messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
#     if hf_token is None or not hf_token.strip():
#         hf_token = os.getenv("HF_TOKEN")
#     client = InferenceClient(model=model, token=hf_token)
#     tokens = f"**`{model_name}`**\n\n"
#     for completion in client.chat_completion(messages, max_tokens=200, stream=True):
#         token = completion.choices[0].delta.content
#         tokens += token
#         yield tokens

# def random_prompt():
#     return random.choice([
#         "Give me 5 very different ways to say the following sentence: 'The quick brown fox jumps over the lazy dog.'",
#         "Write a summary of the plot of the movie 'Inception' using only emojis.",
#         "Write a sentence with the words 'serendipity', 'baguette', and 'C++'.",
#         "Explain the concept of 'quantum entanglement' to a 5-year-old.",
#         "Write a couplet about Python"
#     ])


previous_sessions = []

def inference(prompt_text):
    time.sleep(1)
    result = "Your Result"
    sessions = add_session(prompt_text)
    return result, sessions


def add_session(prompt_text):
    global previous_sessions
    session_name = ' '.join(prompt_text.split()[:2])
    
    if session_name and session_name not in previous_sessions:
        previous_sessions.append(session_name)
        
    return "\n".join(previous_sessions)  # Return only the session logs as a string


def clear_sessions():
    global previous_sessions
    previous_sessions.clear()
    return "\n".join(previous_sessions)

def clear_fields():
    return "", ""  # Return empty strings to clear the prompt and output fields


with gr.Blocks(theme='gradio/soft', css=custom_css) as demo:
    gr.Markdown("<center><h1>HydroFlow LLM Demo</h1></center>")
    gr.Markdown("<center><h3><i><em>Ask me anything on Wastewater or Stormwater!</em></i></h3></center>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Previous Sessions")
            session_list = gr.Textbox(label="Sessions", value="\n".join(previous_sessions), interactive=False, lines=4, max_lines=20)
            add_button = gr.Button("New Session")
            clear_session = gr.Button("Clear Session")

        with gr.Column(scale=2):
            output = gr.Textbox(label="Result", lines=5, max_lines=20)
            prompt = gr.Textbox(label="Enter your Prompt here", max_lines=20)
            
            with gr.Row():
                generate_btn = gr.Button("Generate Answer", variant="primary", size="sm")
                reset_btn = gr.Button("Clear Content", variant="secondary", size="sm", elem_id="primary")

    #  gr.on(
    #     [prompt.submit, generate_btn.click],
    #     allow_ip,
    # ).success(
    #     partial(inference, model="HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1", model_name="Zephyr ORPO 141B A35B"),
    #     [prompt, hf_token_box],
    #     zephyr_output,
    #     show_progress="hidden",
    #     api_name=False
    # )


    generate_btn.click(
        fn=inference,
        inputs=[prompt],
        outputs=[output, session_list],
    )

    prompt.submit(
        fn=inference,
        inputs=[prompt],
        outputs=[output, session_list],
    )

    reset_btn.click(
        lambda: ("", ""),
        inputs=None,
        outputs=[prompt, output]
    )


    # Button to clear the prompt and output fields
    add_button.click(
        fn=clear_fields,  # Only call the clear_fields function
        inputs=None,      # No inputs needed
        outputs=[prompt, output]  # Clear the prompt and output fields
)


    clear_session.click(
        fn=clear_sessions,
        inputs=None,
        outputs=[session_list]
    )

demo.launch()