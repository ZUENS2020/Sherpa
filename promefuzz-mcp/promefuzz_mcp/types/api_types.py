"""
Type definitions for PromeFuzz MCP.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class APIFunction:
    """Represents an API function."""

    header: str
    """Header file where the API function is located."""

    name: str
    """Name of the API function."""

    loc: str
    """Location of the API function (as an identifier)."""

    decl_loc: str
    """Declaration location (in the header) of the API function."""

    def __str__(self) -> str:
        return f"{self.name} at {self.loc.split('/')[-1]} in {Path(self.header).name}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "header": self.header,
            "name": self.name,
            "loc": self.loc,
            "decl_loc": self.decl_loc,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "APIFunction":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class APICollection:
    """Collection of API functions."""

    funcs: list[APIFunction] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Get the count of API functions."""
        return len(self.funcs)

    @property
    def function_names(self) -> list[str]:
        """Get all function names."""
        return list(set(func.name for func in self.funcs))

    @property
    def function_locations(self) -> list[str]:
        """Get all function locations."""
        return [func.loc for func in self.funcs]

    def get_by_location(self, loc: str) -> Optional[APIFunction]:
        """Get API function by location."""
        for func in self.funcs:
            if func.loc == loc:
                return func
        return None

    def get_by_name(self, func_name: str) -> list[APIFunction]:
        """Get API functions by name."""
        return [func for func in self.funcs if func.name == func_name]

    def get_locations_by_name(self, func_name: str) -> list[str]:
        """Get locations of API functions with the same name."""
        return [func.loc for func in self.funcs if func.name == func_name]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "funcs": [f.to_dict() for f in self.funcs],
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "APICollection":
        """Create from dictionary."""
        return cls(funcs=[APIFunction.from_dict(f) for f in data.get("funcs", [])])


@dataclass
class FunctionInfo:
    """Information about a function."""

    name: str
    """Function name."""

    location: str
    """Function location."""

    signature: str
    """Function signature."""

    return_type: str
    """Return type."""

    param_types: list[str] = field(default_factory=list)
    """Parameter types."""

    heldby_class: Optional[str] = None
    """Class that holds this method."""

    impl_range: Optional[tuple[int, int]] = None
    """Implementation line range (start, end)."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "location": self.location,
            "signature": self.signature,
            "return_type": self.return_type,
            "param_types": self.param_types,
            "heldby_class": self.heldby_class,
            "impl_range": self.impl_range,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FunctionInfo":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CallGraphNode:
    """Call graph node."""

    function_name: str
    """Function name."""

    function_location: str
    """Function location."""

    caller: list["CallGraphNode"] = field(default_factory=list)
    """List of functions that call this function."""

    callee: list["CallGraphNode"] = field(default_factory=list)
    """List of functions that this function calls."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "function_location": self.function_location,
            "caller": [c.to_dict() for c in self.caller],
            "callee": [c.to_dict() for c in self.callee],
        }


@dataclass
class CallGraph:
    """Call graph."""

    nodes: list[CallGraphNode] = field(default_factory=list)
    """List of call graph nodes."""

    edges: list[tuple[str, str]] = field(default_factory=list)
    """List of edges (caller_loc, callee_loc)."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": self.edges,
        }


@dataclass
class RelevanceResult:
    """Relevance calculation result."""

    func_loc_a: str
    """Location of function A."""

    func_loc_b: str
    """Location of function B."""

    relevance: float
    """Relevance score."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "func_loc_a": self.func_loc_a,
            "func_loc_b": self.func_loc_b,
            "relevance": self.relevance,
        }


@dataclass
class ComplexityResult:
    """Complexity analysis result."""

    func_location: str
    """Function location."""

    complexity_score: float
    """Complexity score."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "func_location": self.func_location,
            "complexity_score": self.complexity_score,
        }


@dataclass
class IncidentalRelation:
    """Incidental relation between API functions."""

    source_func: str
    """Source function location."""

    target_func: str
    """Target function location (incidentally called)."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source_func": self.source_func,
            "target_func": self.target_func,
        }
