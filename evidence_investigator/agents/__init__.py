"""The four specialist agents named in the Unit 6 design proposal.

Each agent has one responsibility and communicates only through the shared
state, the role-specialised, centrally coordinated structure that Guo et al.
(2024) identify as the common pattern in LLM-based multi-agent systems. Each
agent is a plain callable taking state and returning a partial update, so it
can be unit-tested without LangGraph.
"""

from .critic import CriticAgent
from .planner import PlannerAgent
from .reporter import ReportingAgent
from .retrieval import RetrievalAgent

__all__ = ["CriticAgent", "PlannerAgent", "ReportingAgent", "RetrievalAgent"]
