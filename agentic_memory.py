Here is a reusable Python code for an agentic memory system, designed based on the provided comprehensive search results.

```python
# agentic_memory.py

import time
import json
from collections import deque
from typing import List, Dict, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt

# Best Practice: Use structured formats for memory entries 
class MemoryEntry:
    """A structured class for individual memory entries."""
    def __init__(self, content: Any, timestamp: float, metadata: Optional[Dict] = None):
        self.content = content
        self.timestamp = timestamp
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> Dict:
        """Serializes the memory entry to a dictionary."""
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        """Deserializes a dictionary to a MemoryEntry object."""
        return cls(
            content=data.get("content"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )

# --- Memory Type Implementations ---

class ShortTermMemory:
    """
    Simulates Short-Term Memory (STM) with a limited, temporary capacity.
    Holds a small amount of information for immediate use .
    """
    def __init__(self, capacity: int = 10):
        """
        Initializes the Short-Term Memory.
        Args:
            capacity (int): The maximum number of recent memories to store.
        """
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def add(self, content: Any, metadata: Optional[Dict] = None):
        """Adds a new memory to the short-term buffer."""
        entry = MemoryEntry(content, time.time(), metadata)
        self.memory.append(entry)

    def get_recent(self) -> List[MemoryEntry]:
        """Retrieves all memories currently in the short-term buffer."""
        return list(self.memory)

    def clear(self):
        """Clears all memories from the short-term buffer."""
        self.memory.clear()

class LongTermMemory:
    """
    Base class for long-term memory stores.
    Responsible for storing vast amounts of information over extended periods .
    This implementation uses a simple in-memory list. For production systems,
    this can be replaced with persistent storage like a database .
    """
    def __init__(self):
        self.memory: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry):
        """Adds a memory entry to long-term storage."""
        self.memory.append(entry)

    def retrieve(self, query: str) -> List[MemoryEntry]:
        """
        Retrieves memories based on a simple keyword search.
        In a real-world scenario, this would involve semantic search with embeddings .
        """
        # This is a placeholder for a more sophisticated search (e.g., vector search)
        results = [
            entry for entry in self.memory
            if query.lower() in str(entry.content).lower()
        ]
        return results

    def save(self, filepath: str):
        """Saves the memory to a JSON file for persistence."""
        with open(filepath, 'w') as f:
            json.dump([entry.to_dict() for entry in self.memory], f, indent=4)

    def load(self, filepath: str):
        """Loads memory from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.memory = [MemoryEntry.from_dict(item) for item in data]
        except FileNotFoundError:
            print(f"Warning: Memory file not found at {filepath}. Starting with empty memory.")
            self.memory = []

class SemanticMemory(LongTermMemory):
    """
    Stores general world knowledge, facts, and concepts .
    Inherits from LongTermMemory and can be extended for factual data.
    """
    def add_fact(self, fact: str, category: str, tags: Optional[List[str]] = None):
        """Adds a structured fact to semantic memory."""
        metadata = {"type": "semantic", "category": category, "tags": tags or []}
        entry = MemoryEntry(fact, time.time(), metadata)
        self.add(entry)

class ProceduralMemory(LongTermMemory):
    """
    Stores skills, procedures, and how-to knowledge .
    Inherits from LongTermMemory and can be extended for procedural steps.
    """
    def add_procedure(self, procedure_name: str, steps: List[str], domain: str):
        """Adds a structured procedure to memory."""
        content = {"name": procedure_name, "steps": steps}
        metadata = {"type": "procedural", "domain": domain}
        entry = MemoryEntry(content, time.time(), metadata)
        self.add(entry)

class GraphMemory:
    """

    Represents memory as a network of interconnected information (nodes and edges).
    This enables more contextually relevant search results by analyzing relationships
    between entities . Uses networkx for graph operations.
    """
    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, node_id: str, attributes: Optional[Dict] = None):
        """Adds a node (entity) to the graph memory."""
        self.graph.add_node(node_id, **(attributes or {}))

    def add_edge(self, node1_id: str, node2_id: str, relationship: str):
        """Adds an edge (relationship) between two nodes."""
        if node1_id not in self.graph or node2_id not in self.graph:
            raise ValueError("Both nodes must exist in the graph before adding an edge.")
        self.graph.add_edge(node1_id, node2_id, relationship=relationship)

    def retrieve_neighbors(self, node_id: str) -> List[str]:
        """Finds all nodes directly connected to a given node."""
        if node_id not in self.graph:
            return []
        return list(self.graph.neighbors(node_id))

    def find_path(self, start_node: str, end_node: str) -> Optional[List[str]]:
        """Finds the shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source=start_node, target=end_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def visualize(self, filename: str = "memory_graph.png"):
        """
        Generates a visual representation of the graph memory .
        Saves the graph as a PNG image.
        """
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.graph, k=0.5)
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', node_size=2000,
                edge_color='gray', font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(self.graph, 'relationship')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Agentic Graph Memory")
        plt.savefig(filename)
        plt.close()
        print(f"Graph memory visualization saved to {filename}")


# --- Main System Class ---

class AgenticMemorySystem:
    """
    A comprehensive, hierarchical agentic memory system.
    This system integrates different memory types to provide a robust framework
    for AI agents, following best practices for agentic memory design .
    """
    def __init__(self, stm_capacity: int = 10, ltm_filepath: Optional[str] = None):
        """
        Initializes the complete memory system.
        Args:
            stm_capacity (int): The capacity of the short-term memory.
            ltm_filepath (str, optional): Path to save/load long-term memory.
        """
        print("Initializing Agentic Memory System...")
        # Hierarchical Memory Structure 
        self.short_term_memory = ShortTermMemory(capacity=stm_capacity)
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()
        self.graph_memory = GraphMemory()

        self.ltm_filepath = ltm_filepath
        if self.ltm_filepath:
            self.semantic_memory.load(f"{self.ltm_filepath}_semantic.json")
            self.procedural_memory.load(f"{self.ltm_filepath}_procedural.json")

    def remember(self, memory_type: str, **kwargs):
        """
        Stores information in the appropriate memory type.
        Example:
            remember('short_term', content='User said hello')
            remember('semantic', fact='Paris is the capital of France', category='Geography')
            remember('procedural', procedure_name='Make coffee', steps=['...'], domain='Cooking')
        """
        if memory_type == 'short_term':
            self.short_term_memory.add(kwargs.get('content'), kwargs.get('metadata'))
        elif memory_type == 'semantic':
            self.semantic_memory.add_fact(kwargs.get('fact'), kwargs.get('category'), kwargs.get('tags'))
        elif memory_type == 'procedural':
            self.procedural_memory.add_procedure(kwargs.get('procedure_name'), kwargs.get('steps'), kwargs.get('domain'))
        else:
            print(f"Warning: Unknown memory type '{memory_type}'")

    def recall(self, query: str) -> Dict[str, Any]:
        """
        Retrieves relevant information from all memory types.
        This function demonstrates contextual memory management by searching across
        different memory stores to build a comprehensive context.
        """
        print(f"\n--- Recalling information for query: '{query}' ---")
        context = {
            "short_term": [entry.to_dict() for entry in self.short_term_memory.get_recent()],
            "semantic": [entry.to_dict() for entry in self.semantic_memory.retrieve(query)],
            "procedural": [entry.to_dict() for entry in self.procedural_memory.retrieve(query)],
        }
        return context

    def persist_memories(self):
        """
        Saves all long-term memories to their respective files if a path is configured.
        """
        if self.ltm_filepath:
            print("Persisting long-term memories...")
            self.semantic_memory.save(f"{self.ltm_filepath}_semantic.json")
            self.procedural_memory.save(f"{self.ltm_filepath}_procedural.json")
            print("Memories saved.")
        else:
            print("Warning: ltm_filepath not set. Cannot persist memories.")

    def prune_memories(self, max_age_seconds: int):
        """
        Best Practice: Regularly remove outdated or irrelevant memories .
        This is a simple time-based pruning example.
        """
        current_time = time.time()
        
        def filter_old(memory_list):
            return [mem for mem in memory_list if (current_time - mem.timestamp) < max_age_seconds]

        self.semantic_memory.memory = filter_old(self.semantic_memory.memory)
        self.procedural_memory.memory = filter_old(self.procedural_memory.memory)
        print("Pruned old memories.")


# --- Example Usage ---

if __name__ == "__main__":
    # Initialize the memory system with a file path for persistence
    memory_system = AgenticMemorySystem(ltm_filepath="agent_memory_data")

    print("\n--- Populating Memories ---")
    # 1. Short-Term Memory Example
    memory_system.remember('short_term', content="User's name is Alex.")
    memory_system.remember('short_term', content="Alex is interested in Python programming.")
    print("Added to Short-Term Memory.")

    # 2. Semantic Memory Example
    memory_system.remember('semantic', fact="Python is a high-level, interpreted programming language.", category="Computer Science", tags=["python", "programming"])
    memory_system.remember('semantic', fact="Paris is the capital of France.", category="Geography", tags=["paris", "france", "capitals"])
    print("Added to Semantic Memory.")

    # 3. Procedural Memory Example
    coffee_steps = ["Grind coffee beans", "Boil water", "Pour water over grounds", "Wait 4 minutes", "Press and serve"]
    memory_system.remember('procedural', procedure_name="How to make French Press coffee", steps=coffee_steps, domain="Cooking")
    print("Added to Procedural Memory.")

    # 4. Graph-Based Memory Example
    graph_mem = memory_system.graph_memory
    graph_mem.add_node("Alex", attributes={"interest": "Python"})
    graph_mem.add_node("Python", attributes={"type": "Programming Language"})
    graph_mem.add_node("Paris", attributes={"type": "City"})
    graph_mem.add_node("France", attributes={"type": "Country"})
    graph_mem.add_edge("Alex", "Python", relationship="is interested in")
    graph_mem.add_edge("Paris", "France", relationship="is capital of")
    print("Added to Graph Memory.")
    graph_mem.visualize() # Save a visualization of the graph

    # --- Interaction and Recall Examples ---

    # Example 1: Recall information about the user
    alex_context = memory_system.recall("Alex")
    print("\nRecall results for 'Alex':")
    print(json.dumps(alex_context, indent=2))
    
    # Example 2: Recall information about a technical topic
    python_context = memory_system.recall("Python")
    print("\nRecall results for 'Python':")
    print(json.dumps(python_context, indent=2))

    # Example 3: Recall a procedure
    coffee_context = memory_system.recall("coffee")
    print("\nRecall results for 'coffee':")
    print(json.dumps(coffee_context, indent=2))

    # Example 4: Using Graph Memory
    print("\n--- Graph Memory Queries ---")
    alex_neighbors = graph_mem.retrieve_neighbors("Alex")
    print(f"Entities related to 'Alex': {alex_neighbors}")
    path = graph_mem.find_path("Alex", "Python")
    print(f"Path between 'Alex' and 'Python': {path}")

    # --- Persistence Example ---
    # Save the current state of long-term memories to files
    memory_system.persist_memories()

    # Create a new instance to demonstrate loading from files
    print("\n--- Demonstrating Persistence ---")
    new_memory_system = AgenticMemorySystem(ltm_filepath="agent_memory_data")
    loaded_context = new_memory_system.recall("France")
    print("\nRecall results for 'France' from new instance:")
    print(json.dumps(loaded_context, indent=2))
```