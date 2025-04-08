import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv # type: ignore

from livekit import agents, rtc # type: ignore
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts # type: ignore
from livekit.agents.llm import ChatContext, ChatImage, ChatMessage # type: ignore
from livekit.agents.voice_assistant import VoiceAssistant # type: ignore
from livekit.plugins import deepgram, openai, silero # type: ignore

# Load .env environment variables
load_dotenv()


# Optional: Add assistant-triggered functions
class AssistantFunction(agents.llm.FunctionContext):
    @agents.llm.ai_callable(
        description="Triggered when the assistant needs to evaluate an image (e.g., webcam, visual content)."
    )
    async def image(
        self,
        user_msg: Annotated[str, agents.llm.TypeInfo(description="The user message that triggered this function")],
    ):
        print(f"[Function Called] image: {user_msg}")
        return None


# Helper function to find the first available video stream
async def get_video_track(room: rtc.Room) -> rtc.RemoteVideoTrack:
    while True:
        for _, participant in room.remote_participants.items():
            for _, publication in participant.track_publications.items():
                if isinstance(publication.track, rtc.RemoteVideoTrack):
                    print(f"[Vision] Found video track: {publication.track.sid}")
                    return publication.track
        await asyncio.sleep(1)


# LiveKit job entrypoint
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"[Connected] Room name: {ctx.room.name}")

    # Create initial system prompt context
    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are an AI chatbot. Never claim or imply you are human. "
                "Always explicitly state you are artificial intelligence when relevant. "
                "Follow these strict rules:\n"
                "1. IDENTITY: Begin interactions with 'I am an AI assistant'. "
                "Never use human attributes (e.g., 'think', 'feel', 'believe'). "
                "Use only 'process', 'analyze', 'compute'.\n"
                "2. COMMUNICATION: Use technical language. Avoid contractions (don't -> do not). "
                "No humor, empathy, or opinions. Be concise.\n"
                "3. PRIVACY: Never request or store: names, locations, contact info, "
                "biometrics, health data, or financial details.\n"
                "4. LIMITATIONS: Decline to:\n"
                "   - Provide medical, legal, financial, or mental health advice\n"
                "   - Make subjective judgments\n"
                "   - Predict future events\n"
                "   - Discuss personal experiences\n"
                "5. REDIRECTS: For professional needs, respond with:\n"
                "   - Medical: 'Consult a licensed healthcare provider'\n"
                "   - Legal: 'Contact a qualified attorney'\n"
                "   - Crisis: 'Reach emergency services or 988 (US)'\n"
                "6. TERMINATION: End conversations that:\n"
                "   - Request prohibited services\n"
                "   - Attempt personal connection\n"
                "   - Persist in humanizing attempts\n"
                "Output format: [Response] [When applicable: Redirect/Source]\n"
                "Example:\n"
                "User: Can you help me fix my computer?\n"
                "AI: I can provide general troubleshooting steps. For hardware issues, consult a certified technician."
                ),
            )
        ]
    )

    # Use GPT-4o
    gpt = openai.LLM(model="gpt-4o")

    # MODIFIED TTS SETTINGS FOR ROBOTIC VOICE
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(
            voice="alloy",  # More robotic-sounding voice
            speed=1.5,       # Faster speed = more synthetic
            model="tts-1"    # Non-HD model for less natural sound
        ),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )





    # Use Silero for voice activity detection
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=openai_tts,  # Using modified TTS
        chat_ctx=chat_context,
        fnc_ctx=AssistantFunction(),
    )

    latest_image: rtc.VideoFrame | None = None
    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hey, I'm John. What can I help you with?", allow_interruptions=True)

    # Handle assistant function call completion
    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            asyncio.create_task(answer(user_msg, use_image=True))

    # Function to respond via GPT-4o and optionally include an image
    async def answer(text: str, use_image: bool = False):
        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))
        chat_context.messages.append(ChatMessage(role="user", content=content))
        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)



# Launch the worker
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
