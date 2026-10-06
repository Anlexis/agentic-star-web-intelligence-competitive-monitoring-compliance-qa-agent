"""AgentCore Platform v1.0"""

# Entry-point exposure. config/agent.yaml names the agent by dotted path
# ("src.graph.graph.WebIntelligenceComplianceQAAgent"), and importers reach it
# through this package, so the package __init__ re-exports the real graph class.
from .graph import WebIntelligenceComplianceQAAgent

__all__ = ["WebIntelligenceComplianceQAAgent"]
