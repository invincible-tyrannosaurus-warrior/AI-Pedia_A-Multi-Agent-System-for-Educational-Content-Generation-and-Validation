"""
Agent Registry Module

This module provides a central registry for all available agents in the system.
It allows dynamic registration, retrieval, and capability discovery of agents,
facilitating the Task Manager's orchestration process.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Standard interface for agent pipelines
# Function signature: 
# def run_pipeline(instruction: str, output_dir: Path, assets: list = None, **kwargs) -> dict
AgentPipelineFunc = Callable[..., Dict[str, Any]]

class AgentRegistry:
    """
    Central registry for managing agent capabilities and execution entry points.
    """
    
    def __init__(self) -> None:
        self._agents: Dict[str, AgentPipelineFunc] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self, 
        name: str, 
        agent_func: AgentPipelineFunc,
        description: str = "",
        capabilities: Optional[List[str]] = None
    ) -> None:
        """
        Register a new agent with the system.
        
        Args:
            name: Unique identifier for the agent (e.g., 'coder', 'video')
            agent_func: The entry point function to execute the agent's pipeline
            description: Human-readable description of what the agent does
            capabilities: List of keywords describing the agent's abilities
        """
        if name in self._agents:
            logger.warning(f"Overwriting existing agent: {name}")
            
        self._agents[name] = agent_func
        self._metadata[name] = {
            "description": description,
            "capabilities": capabilities or []
        }
        logger.info(f"Registered agent: {name}")
    
    def get(self, name: str) -> AgentPipelineFunc:
        """
        Retrieve the execution function for a specific agent.
        
        Args:
            name: Agent identifier
            
        Returns:
            The callable pipeline function
            
        Raises:
            ValueError: If the agent is not found
        """
        if name not in self._agents:
            raise ValueError(f"Unknown agent: {name}. Available: {list(self._agents.keys())}")
        return self._agents[name]
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a specific agent."""
        return self._metadata.get(name, {})
    
    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered agents and their metadata.
        
        Returns:
            Dictionary mapping agent names to their metadata
        """
        return self._metadata.copy()

# Global registry instance
registry = AgentRegistry()
