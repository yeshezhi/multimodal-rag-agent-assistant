import subprocess
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .rag import RagService


class AgentState(TypedDict, total=False):
    question: str
    route: str
    answer: str


class EnterpriseAgent:
    """Small, explicit LangGraph tool-routing agent for the first production workflow."""

    def __init__(self, rag: RagService) -> None:
        self.rag = rag
        graph = StateGraph(AgentState)
        graph.add_node("route", self.route)
        graph.add_node("gpu_status", self.gpu_status)
        graph.add_node("knowledge_base", self.knowledge_base)
        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", lambda state: state["route"], {"gpu_status": "gpu_status", "knowledge_base": "knowledge_base"})
        graph.add_edge("gpu_status", END)
        graph.add_edge("knowledge_base", END)
        self.graph = graph.compile()

    @staticmethod
    def route(state: AgentState) -> AgentState:
        question = state["question"].lower()
        gpu_words = ("gpu", "显卡", "显存", "nvidia-smi", "服务器状态", "cuda")
        return {"route": "gpu_status" if any(word in question for word in gpu_words) else "knowledge_base"}

    @staticmethod
    def gpu_status(_: AgentState) -> AgentState:
        command = ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        if result.returncode:
            return {"answer": "GPU 状态工具调用失败。"}
        return {"answer": f"已调用 gpu_status 工具：\n{result.stdout.strip()}"}

    def knowledge_base(self, state: AgentState) -> AgentState:
        response = self.rag.chat(state["question"], top_k=3)
        return {"answer": response.answer}

    def invoke(self, question: str) -> dict:
        return self.graph.invoke({"question": question})

    def invoke_image(self, image: bytes, question: str) -> dict:
        answer, citations, matches = self.rag.multimodal_rag(image, question, top_k=3)
        return {"route": "image_multimodal", "answer": answer, "citations": citations, "matches": matches}
