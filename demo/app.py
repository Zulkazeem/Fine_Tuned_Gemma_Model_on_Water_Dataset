#importing libraries
import gradio as gr
import tensorflow.keras as keras
import time
import keras_nlp
import os


model_path = "Zul001/HydroSense_Gemma_Finetuned_Model"
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(f"hf://{model_path}")




custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Edu+AU+VIC+WA+NT+Dots:wght@400..700&family=Give+You+Glory&family=Sofia&family=Sunshiney&family=Vujahday+Script&display=swap');
.gradio-container, .gradio-container * {
     font-family: "Playfair Display", serif;
  font-optical-sizing: auto;
  font-weight: <weight>;
  font-style: normal;
}
"""
js = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') === 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""


previous_sessions = []
memory = [{}]



def inference(prompt_text):
  prompt_text = prompt_text
  generated_text = gemma_lm.generate(prompt_text)

  #Apply post-processing
  formatted_output = post_process_output(prompt_text, generated_text)
  print(formatted_output)

  #adding a bit of delay
  time.sleep(1)
  result = formatted_output
  sessions = add_session(prompt_text)
  return result, sessions


def remember(prompt, result):
    global memory
    # Store the session as a dictionary
    session = {'prompt': prompt, 'result': result}
    memory.append(session)

    # Update previous_sessions for display
    session_display = [f"Q: {s['prompt']} \nA: {s['result']}" for s in memory]
    
    return "\n\n".join(session_display)  # Return formatted sessions as a string


def add_session(prompt_text):
    global previous_sessions
    session_name = ' '.join(prompt_text.split()[:5])
    
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
    

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Previous Sessions")
            session_list = gr.Textbox(label="Sessions", value="\n".join(previous_sessions), interactive=False, lines=4, max_lines=20)
            add_button = gr.Button("New Session")
            clear_session = gr.Button("Clear Session")

        with gr.Column(scale=2):
            output = gr.Textbox(label="Result", lines=5, max_lines=20)
            prompt = gr.Textbox(label="Enter your Prompt here", max_lines=20, placeholder = "Ask me anything on Wastewater or Stormwater!")
            
            with gr.Row():
                generate_btn = gr.Button("Generate Answer", variant="primary", size="sm")
                reset_btn = gr.Button("Clear Content", variant="secondary", size="sm", elem_id="primary")


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

demo.launch(share=True)
