import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal, List
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
    metrics,
    MetricsCollectedEvent,
)
from livekit.plugins import silero, deepgram, murf, google, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging & env
# -------------------------
logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)s %(message)s"))
logger.addHandler(h)

load_dotenv(".env.local")

# -------------------------
# Load course content
# -------------------------
def load_content():
    base_dir = Path(__file__).resolve().parent.parent
    json_path = base_dir / "shared-data" / "day4_tutor_content.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Content file not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))

COURSE_CONTENT = load_content()

# -------------------------
# State & userdata
# -------------------------
@dataclass
class TutorState:
    mode: Literal["learn", "quiz", "teach_back"] = "learn"
    current_topic_id: Optional[str] = None
    history: List[dict] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def topic(self):
        if not self.current_topic_id:
            return None
        return next((t for t in COURSE_CONTENT if t["id"] == self.current_topic_id), None)

@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None

# -------------------------
# Voice mapping
# -------------------------
VOICE_MAP = {
    "learn": ("en-US-matthew", "Promo"),
    "quiz": ("en-US-alicia", "Conversation"),
    "teach_back": ("en-US-ken", "Conversational"),
}

# -------------------------
# Tools
# -------------------------
@function_tool
async def list_topics(ctx: RunContext[Userdata]) -> str:
    return "Available topics: " + "; ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])

@function_tool
async def select_topic(ctx: RunContext[Userdata], topic_id: str) -> str:
    tid = (topic_id or "").strip().lower()
    found = next((t for t in COURSE_CONTENT if t["id"] == tid), None)
    if not found:
        return f"Topic '{topic_id}' not found. " + await list_topics(ctx)
    ctx.userdata.tutor_state.current_topic_id = tid
    return f"Selected topic: {found['title']}."

@function_tool
async def set_mode(ctx: RunContext[Userdata], mode: str) -> str:
    m = (mode or "").strip().lower()
    if m not in VOICE_MAP:
        return "Invalid mode. Choose learn, quiz, or teach_back."
    ctx.userdata.tutor_state.mode = m

    # update TTS voice immediately
    session = ctx.userdata.agent_session
    if session and hasattr(session, "tts"):
        try:
            voice, style = VOICE_MAP[m]
            session.tts.update_options(voice=voice, style=style)
        except Exception as e:
            logger.warning(f"TTS update failed: {e}")

    return f"Mode set to {m}."

@function_tool
async def explain_topic(ctx: RunContext[Userdata]) -> str:
    topic = ctx.userdata.tutor_state.topic()
    if not topic:
        return "No topic selected. Use select_topic first."
    return topic.get("summary", "No summary available.")

@function_tool
async def ask_quiz(ctx: RunContext[Userdata]) -> str:
    topic = ctx.userdata.tutor_state.topic()
    if not topic:
        return "No topic selected. Use select_topic first."
    return topic.get("sample_question", "No sample question available.")

@function_tool
async def prompt_teachback(ctx: RunContext[Userdata]) -> str:
    topic = ctx.userdata.tutor_state.topic()
    if not topic:
        return "No topic selected. Use select_topic first."
    return topic.get("teaching_prompt", topic.get("sample_question", "Please explain this topic back to me."))

@function_tool
async def evaluate_teachback(ctx: RunContext[Userdata], user_answer: str) -> str:
    topic = ctx.userdata.tutor_state.topic()
    if not topic:
        return "No topic selected."
    text_src = (topic.get("title","") + " " + topic.get("summary","")).lower()
    keywords = [w for w in set(text_src.split()) if len(w) > 3][:12]
    user_lower = (user_answer or "").lower()
    match = sum(1 for kw in keywords if kw in user_lower)
    total = max(1, len(keywords))
    score = int((match / total) * 10)
    score = max(1, min(10, score))
    if score >= 8:
        feedback = "Great — clear and covers main points."
    elif score >= 5:
        feedback = "Good — covers some points; try adding a short example or definition."
    else:
        feedback = "Needs improvement — try a short definition and an example."
    return f"{feedback} (score: {score}/10)"

# -------------------------
# Agent
# -------------------------
class TeachTutorAgent(Agent):
    def __init__(self):
        topics = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])
        super().__init__(
            instructions=f"""
You are a friendly tutor with three modes: learn, quiz, teach_back.
1) Greet user and ask which mode they want.
2) Ask which topic (user can ask list_topics).
3) Learn: explain topic.summary (Matthew).
4) Quiz: ask topic.sample_question (Alicia).
5) Teach_back: prompt user to explain topic (Ken), then evaluate.
Users can switch modes anytime by "switch to X".
Available topics: {topics}
""",
            tools=[list_topics, select_topic, set_mode, explain_topic, ask_quiz, prompt_teachback, evaluate_teachback],
        )

# -------------------------
# Prewarm & Entry
# -------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    userdata = Userdata(tutor_state=TutorState())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Promo",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
        preemptive_generation=True,
    )

    userdata.agent_session = session

    # metrics
    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage: %s", usage_collector.get_summary())
    ctx.add_shutdown_callback(log_usage)

    # connect & start session
    await ctx.connect()
    await session.start(
        agent=TeachTutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    # initial greeting
    try:
        await session.say("Hello! I'm your tutor. Which mode would you like: learn, quiz, or teach-back?")
    except Exception:
        logger.debug("session.say failed; agent will prompt when run() starts.")

    # session loop
    try:
        await session.run()
    except Exception as e:
        logger.exception("session.run failed: %s", e)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

