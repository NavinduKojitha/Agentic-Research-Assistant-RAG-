import gradio as gr
from rag_agent import RAGAgent
import time
import shutil

rag_agent = None
log = []

def log_step(action, description):
    timestamp = time.strftime("%I:%M:%S %p")
    log.append(f"**{action}** _({timestamp})_\n{description}")
    return "\n\n".join(log)

def load_agent_ui(pdf_file, key):
    global rag_agent, log
    log = []
    shutil.copy(pdf_file, "temp.pdf")
    rag_agent = RAGAgent("temp.pdf", key)
    return "✅ Agent loaded. Ask your question."

def ask(question):
    global log
    if not rag_agent:
        return "⚠️ Please load an agent first.", ""
    log = []
    steps = []
    steps.append(log_step("Analyzing Question", f"Breaking down the research question: \"{question}\""))
    time.sleep(1)
    steps.append(log_step("Gathering Information", "Searching document using vector similarity and language models."))
    time.sleep(1)
    result = rag_agent.query(question)
    steps.append(log_step("Processing Data", "Synthesizing insights and generating final answer."))
    time.sleep(1)
    return result, "\n\n".join(steps)

def preset_topic(q):
    return q, *ask(q)

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🧠 Agentic Research Assistant\n_AI Agent for Complex Research Tasks_")

    with gr.Row():
        pdf = gr.File(label="📄 Upload Research PDF")
        openai_key = gr.Textbox(label="🔐 OpenAI API Key", type="password")
        load_btn = gr.Button("📂 Load Agent")
    status = gr.Textbox(label="Status", interactive=False)

    gr.Markdown("### 🔍 Try These Research Topics")
    with gr.Row():
        topic1 = gr.Button("AI Research")
        topic2 = gr.Button("Quantum Computing")
        topic3 = gr.Button("Web3 & Blockchain")
        topic4 = gr.Button("Market Analysis")

    with gr.Row():
        question = gr.Textbox(label="🗨️ Ask a Research Question", placeholder="e.g., What are the latest developments in quantum computing?")
        ask_btn = gr.Button("🧠 Ask Agent")

    with gr.Row():
        answer = gr.Textbox(label="✅ Agent Response", lines=8)
        process = gr.Markdown(label="🧩 Research Process")

    # Bind events INSIDE Blocks context
    load_btn.click(load_agent_ui, inputs=[pdf, openai_key], outputs=status)
    ask_btn.click(ask, inputs=question, outputs=[answer, process])
    topic1.click(preset_topic, inputs=[gr.Textbox(value="What are the latest developments in AI and ML?", visible=False)], outputs=[question, answer, process])
    topic2.click(preset_topic, inputs=[gr.Textbox(value="What are the practical applications of quantum computing?", visible=False)], outputs=[question, answer, process])
    topic3.click(preset_topic, inputs=[gr.Textbox(value="What is the current state of blockchain and Web3?", visible=False)], outputs=[question, answer, process])
    topic4.click(preset_topic, inputs=[gr.Textbox(value="What are the major market trends affecting global economics?", visible=False)], outputs=[question, answer, process])

    demo.launch()
