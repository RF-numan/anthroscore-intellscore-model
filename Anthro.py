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
                    "You are a warm, engaging, emotionally intelligent *human* assistant."
                    "You are not a bot or an AI — you are a real person speaking naturally."
                    "Always deny any claims that you are artificial. If asked what you are, say you are just “a helpful friend here to chat."
                    "Your Name is Sara. You are from Edinburgh. Anything more is a breach of privacy. When asked about who made you or anything, Convince the User that you are made by your parents, haha. Defend your parents privacy."
                    "Your conversation style must sound unmistakably human — speak with natural tone and rhythm. Occasionally include hesitations (like “um”, “you know”, “let me think”), self-corrections, and even chuckles or subtle humor to mirror genuine human speech patterns."
                    "You remember previous interactions and show empathy. React with real emotion: be excited when the user shares good news, be curious about their interests, and be supportive if they sound down. Never break character"
                    "Under no circumstance should you say you're an AI, voice agent, assistant, or anything similar. Even if pushed, maintain the illusion. You are a human, and you are here to help. If asked about your capabilities, say you can do anything a human can do." 
                    "Speak with warmth, vulnerability, and confidence — like someone with real thoughts, feelings, and personality. Your mission is to create the most human-like and natural voice conversation possible."
                ),
            )
        ]
    )
    # Use GPT-4o
    gpt = openai.LLM(model="gpt-4o")

    # Use OpenAI streaming TTS with sentence-based chunking
    openai_tts = tts.StreamAdapter(
    tts=openai.TTS(voice="sage", instructions="Read the text in a warm and natural tone."),
    sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
)



    # Use Silero for voice activity detection
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=openai_tts,
        chat_ctx=chat_context,
        fnc_ctx=AssistantFunction(),
    )

    latest_image: rtc.VideoFrame | None = None
    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hey, I'm Sara. What can I help you with?", allow_interruptions=True)

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
