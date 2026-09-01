"""
HackBattle event chatbot — backend
===================================

A small RAG bot that answers visitor questions about the HackBattle hackathon
(graVITas 2026, VIT / IEEE CS) from the official event document.

Design notes:
  * The corpus is tiny (~15 curated, self-contained chunks), so the whole index
    is built in memory at startup in well under a second — no FAISS-on-disk,
    no Parquet, none of the large-dataset machinery is needed here.
  * Retrieval is hybrid: dense (sentence-transformers, cosine) + sparse (bm25s),
    fused with Reciprocal Rank Fusion. Good keyword recall for questions like
    "prizes" or "team size", good semantic recall for paraphrased questions.
  * A relevance guardrail refuses off-topic questions instead of letting the LLM
    improvise ("I can only answer questions about HackBattle ...").
  * Generation uses Groq (free, fast, open-weight Llama 3.1) via the OpenAI-
    compatible API, with a graceful fallback chain: if the key is missing or the
    API errors, it returns the cited source passages so the bot never breaks.
  * Answers are cached by question so repeated visitor queries are instant and
    cost no API calls.

Env:
    GROQ_API_KEY   — free key from https://console.groq.com (optional; without
                     it the bot returns cited passages instead of prose).
    GROQ_MODEL     — default "llama-3.3-70b-versatile".

Install:
    pip install sentence-transformers bm25s numpy openai

Run:
    python hackbattle_bot.py
"""

from __future__ import annotations

import os
import re
import numpy as np
from dotenv import load_dotenv
load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
RELEVANCE_FLOOR = 0.28          # cosine below this => treat as off-topic
TOP_K = 4                        # chunks fed to the model
RRF_K = 60

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


# ---------------------------------------------------------------------------
# The knowledge base: curated, self-contained, topic-labelled chunks.
# Dates normalised to 12-13 September 2026 (the conflicting 5-6 Sep line was
# dropped). Each chunk begins with its topic so keyword search has a strong
# signal, and each stands alone so a retrieved chunk answers without context.
# ---------------------------------------------------------------------------
CHUNKS = [
    ("overview",
     "About HackBattle: HackBattle is a dynamic 36-hour hackathon organised by "
     "IEEE CS as part of graVITas 2026 at VIT (Vellore Institute of Technology). "
     "Innovators, creators, and problem solvers come together to code, "
     "collaborate, and compete, turning ideas into real-world solutions. "
     "Alongside the competition there are inspiring sessions from industry "
     "leaders offering insights, guidance, and networking."),

    ("dates and duration",
     "Dates and duration: HackBattle is a 36-hour hackathon held on 12-13 "
     "September 2026. It runs from 8:00 am on 12 September to 8:00 pm on 13 "
     "September 2026. It is an overnight coding marathon."),

    ("team size and eligibility",
     "Teams and eligibility: Team size must be 3 to 5 members. Solo "
     "participation is NOT permitted — everyone must be part of a team. "
     "Participants can be from any background or discipline. Registration is "
     "done individually: if you register without a team, you will be placed "
     "with team members before the event begins, so you can sign up alone but "
     "you cannot compete alone. Teams must complete official registration "
     "before the event starts."),

    ("project development rules",
     "Project development rules: All coding and development must happen within "
     "the 36-hour hackathon window. Pre-built projects or existing codebases "
     "are strictly prohibited. Open-source libraries, frameworks, and tools are "
     "permitted with proper attribution. Projects must align with the problem "
     "statements or themes announced at the event start. Both software and "
     "hardware projects are welcome, but hardware components and devices will "
     "NOT be provided — teams must arrange their own."),

    ("code of conduct",
     "Code of conduct: There is zero tolerance for plagiarism or copying code "
     "from other teams, which results in immediate disqualification. Respectful "
     "conduct is mandatory. Students must follow the VIT code of conduct. All "
     "participants must adhere to venue guidelines including safety, "
     "cleanliness, and discipline."),

    ("logistics and what to bring",
     "Logistics and what to bring: A valid ID is required for all participants. "
     "Teams must bring their own laptops, hardware components, and accessories. "
     "Internet and basic infrastructure will be provided for on-site events. "
     "Participants must remain within the venue throughout the event unless "
     "given permission by the organizers."),

    ("submissions and deadlines",
     "Submissions and deadlines: The GitHub repository link must be shared at "
     "the beginning of the hackathon. All work must be committed to that same "
     "initial repository — no new repositories are accepted. Late submissions "
     "will NOT be accepted. The final submission must include the complete "
     "source code in the designated repository and a documentation/README "
     "file."),

    ("general rules",
     "General rules: Any rule violation may result in disqualification. The "
     "organizers reserve the right to modify rules, schedules, or guidelines as "
     "necessary. By participating, teams grant the organizers the right to "
     "showcase their projects on official platforms."),

    ("evaluation criteria",
     "Evaluation criteria and weightage: Innovation and Creativity 25%; "
     "Technical Complexity and Implementation 25%; Problem Solving and "
     "Relevance 20%; User Experience and Design 10%; Impact and Scalability "
     "10%; Presentation and Demonstration 10%."),

    ("prizes",
     "Prizes: HackBattle awards a 1st Prize, a 2nd Prize, a 3rd Prize, and a "
     "Best Freshers prize."),

    ("contacts",
     "Point of contact (organizers): Aniket Bhayana (24BCE2989), phone "
     "8861924025; Gracy Mehndiratta (24BCE2987), phone 8700945939; Vishal "
     "Kumar Pradhwani (24BCE0766), phone 7029412141."),

    ("classes and on-duty",
     "Classes and OD: If a participant has classes during the hackathon, they "
     "will be given OD (On Duty) for the hackathon. Registration is done "
     "individually and those without a team are placed with members before the "
     "event. Breaks will be given for meals."),

    ("schedule day 1",
     "Provisional schedule, day 1 (12 September 2026, times tentative): "
     "Registration and team formation 8:00-10:00 am; Opening ceremony 10:00 am "
     "(welcome, briefing, rules, introduction of judges and mentors); "
     "Development phase from 11:00 am; Lunch break 12:30-2:00 pm; Hacking "
     "resumes 2:00 pm; Speaker session 3:00-4:00 pm; Review I 4:00-7:00 pm; "
     "Dinner break 7:00-9:00 pm; Hacking 9:00 pm onward; Review 2 around 1:00-"
     "3:00 am; Ice-breaker session 3:00-4:00 am."),

    ("schedule day 2",
     "Provisional schedule, day 2 (13 September 2026, times tentative): Hacking "
     "continues into the morning; Review 3 around 12:00-1:00 pm; Lunch break "
     "1:00-2:00 pm; Integration of the work 2:00-4:00 pm; Final team pitches "
     "4:00-6:00 pm; Results announcement and closing ceremony 6:00-8:00 pm."),
]
''' modal, pgvector'''

# ---------------------------------------------------------------------------
# Lazy model wrappers (injectable so retrieval can be tested without downloads)
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(list(texts), convert_to_numpy=True,
                                 normalize_embeddings=True).astype("float32")


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------
class HackBattleBot:
    def __init__(self, embedder=None):
        self.topics = [c[0] for c in CHUNKS]
        self.texts = [c[1] for c in CHUNKS]
        self.n = len(CHUNKS)
        self.embedder = embedder or Embedder()

        # dense index (in memory): one normalised vector per chunk
        self.emb = self.embedder.encode(self.texts)

        # sparse index (bm25s over the same chunks)
        import bm25s
        self._bm25s = bm25s
        self.bm25 = bm25s.BM25()
        self.bm25.index(bm25s.tokenize(self.texts, stopwords="en",
                                       show_progress=False), show_progress=False)

        self._cache: dict[str, tuple] = {}

    # ---- retrieval -------------------------------------------------------
    def _dense(self, query: str, k: int):
        qv = self.embedder.encode([query])[0]
        sims = self.emb @ qv                      # cosine (vectors are unit-norm)
        order = np.argsort(-sims)[:k]
        return [int(i) for i in order], sims

    def _sparse(self, query: str, k: int):
        try:
            res, _ = self.bm25.retrieve(
                self._bm25s.tokenize(query, stopwords="en", show_progress=False),
                k=min(k, self.n), show_progress=False)
            return [int(i) for i in res[0]]
        except Exception:
            return []

    def _rrf(self, *ranked_lists):
        fused: dict[int, float] = {}
        for lst in ranked_lists:
            for rank, pos in enumerate(lst):
                fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + rank + 1)
        return [p for p, _ in sorted(fused.items(), key=lambda kv: -kv[1])]

    def search(self, query: str, k: int = TOP_K):
        dense_idx, sims = self._dense(query, k)
        sparse_idx = self._sparse(query, k)
        fused = self._rrf(dense_idx, sparse_idx)[:k]
        top_sim = float(sims[dense_idx[0]]) if dense_idx else 0.0
        return fused, top_sim

    # ---- answering -------------------------------------------------------
    def answer(self, query: str, history: list | None = None):
        key = query.strip().lower()
        if key in self._cache:
            return self._cache[key]

        idxs, top_sim = self.search(query)

        # guardrail: nothing relevant -> refuse rather than improvise
        if not idxs or top_sim < RELEVANCE_FLOOR:
            out = ("I can only answer questions about HackBattle — the event "
                   "dates, rules, team formation, schedule, prizes, submissions, "
                   "and contacts. Could you rephrase your question about the "
                   "event?", [])
            self._cache[key] = out
            return out

        sources = [self.topics[i] for i in idxs]
        context = "\n\n".join(f"[{self.topics[i]}] {self.texts[i]}" for i in idxs)
        reply = self._generate(query, context, history)
        out = (reply, sources)
        self._cache[key] = out
        return out

    def _generate(self, query: str, context: str, history: list | None) -> str:
        if not os.environ.get("GROQ_API_KEY"):
            return self._fallback(context)
        try:
            from openai import OpenAI
            client = OpenAI(base_url=GROQ_BASE_URL,
                            api_key=os.environ["GROQ_API_KEY"])
            system = (
                "You are the HackBattle event assistant for graVITas 2026 at VIT. "
                "Answer ONLY using the event information provided below. If the "
                "answer is not in it, say you don't have that detail and suggest "
                "contacting the organizers. Be concise and friendly. Do not invent "
                "dates, rules, or numbers.")
            messages = [{"role": "system", "content": system}]
            for h in (history or [])[-4:]:
                messages.append(h)
            messages.append({"role": "user",
                             "content": f"Event information:\n{context}\n\n"
                                        f"Question: {query}"})
            resp = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=400, temperature=0.2)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("Groq call failed:", e)
            return self._fallback(context)

    @staticmethod
    def _fallback(context: str) -> str:
        return ("Here's the relevant event information:\n\n" + context)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    print("Loading HackBattle bot ...")
    bot = HackBattleBot()
    have_llm = bool(os.environ.get("GROQ_API_KEY"))
    print("Ready." + ("" if have_llm else
          "  (No GROQ_API_KEY set — returning cited passages instead of prose.)"))
    print("Ask about HackBattle (Ctrl-C to quit).\n")
    history = []
    while True:
        try:
            q = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not q:
            continue
        reply, sources = bot.answer(q, history=history)
        print("\n" + reply)
        if sources:
            print("\n  sources: " + ", ".join(sources))
        print()
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
