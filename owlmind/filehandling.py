import os
import discord
import asyncio
from docx import Document
import pdfplumber
from transformers import pipeline
from .discord import DiscordBot
from .botengine import BotBrain, BotMessage

class StudentAssistantBot(DiscordBot):
    """
    Local BART+BERT summarizer with LLaMA 3.2-powered Discord bot for summaries and student-friendly explanations.
    """

    def __init__(self, token, brain: BotBrain, debug: bool = False):
        super().__init__(token, brain, promiscous=False, debug=debug)
        self.chat_history = []
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    async def on_message(self, message):
        if message.author == self.user:
            return

        allowed_extensions = ['.pdf', '.txt', '.docx']

        if message.attachments:
            for attachment in message.attachments:
                ext = os.path.splitext(attachment.filename)[1].lower()
                if ext not in allowed_extensions:
                    await message.channel.send(f"❌ Unsupported file: {attachment.filename}. Use PDF, DOCX, or TXT.")
                    return

                file_path = f"./downloads/{attachment.filename}"
                os.makedirs('./downloads', exist_ok=True)

                try:
                    await attachment.save(file_path)
                    print(f"✅ Saved file: {file_path}")
                except Exception as e:
                    await message.channel.send("⚠️ Error saving file.")
                    print(e)
                    return

                content = self.extract_text(file_path, ext)
                if not content.strip():
                    await message.channel.send("⚠️ No readable content found.")
                    return

                response = await self.ask_llama_chained(content, mode="summary")
                await self.send_in_chunks(message.channel, response, prefix="📘 **Summary & Explanation:**\n")

                os.remove(file_path)
            return

        # No file – treat as student question
        response = await self.ask_llama_chained(message.content, mode="qa")
        chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
        for i, chunk in enumerate(chunks):
            prefix = "💬 " if i == 0 else ""
            await message.channel.send(prefix + chunk)

    def extract_text(self, file_path, file_extension):
        if file_extension == '.docx':
            return "\n".join([p.text for p in Document(file_path).paragraphs])
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension == '.pdf':
            text = ""
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                print(f"PDF error: {e}")
            return text
        return ""

    def summarize_with_bert(self, content):
        print("📄 Summarizing content using BART...")
        if len(content) > 1024:
            content = content[:1024]
        summary = self.summarizer(content, max_length=150, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    
    async def send_in_chunks(self, channel, content, prefix=""):
        max_length = 2000 - len(prefix)
        chunks = [content[i:i + max_length] for i in range(0, len(content), max_length)]
        for i, chunk in enumerate(chunks):
            message = prefix + chunk if i == 0 else chunk
            await channel.send(message)

    async def ask_llama_chained(self, content, mode="summary"):
        if mode == "summary":
            summary = self.summarize_with_bert(content)
            prompt = f"""You are an expert tutor. Here's a summary of some academic material:

SUMMARY:
\"\"\"
{summary}
\"\"\"

Please explain the key ideas in simple terms a student can understand.
Keep your explanation under 450 words."""
        else:
            context = self.chat_history[-1]['content'] if self.chat_history else ''
            prompt = f"""You are a helpful tutor. Here's some context from a previous explanation:

\"\"\"
{context}
\"\"\"

Now answer this student question:
{content}

Make sure your response is concise and stays under 450 words."""

        self.chat_history.append({"role": "user", "content": content})

        try:
            process = await asyncio.create_subprocess_exec(
                "ollama", "run", "llama3.2",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate(input=prompt.encode('utf-8'))

            if stderr:
                print("⚠️ LLaMA stderr:", stderr.decode())

            response = stdout.decode().strip()
        except Exception as e:
            print(f"🔥 Ollama subprocess error: {e}")
            response = "⚠️ Failed to get a response from the model."

        self.chat_history.append({"role": "assistant", "content": response})
        return response
