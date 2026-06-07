# ============================================================================
# PS9 - City Surveillance Agent
# Greedy Best First Search (GBFS) for the Route Inspection problem
# (cover every lane/edge exactly once - an Eulerian path/circuit).
#
# Group: G041
# Course: MTech AIML - AIMLCZG557/AECLZG557
#
# The drone must traverse every lane (edge) between landmarks (vertices)
# without repeating any lane, while landmarks MAY be visited more than once.
# A solution that covers every edge exactly once exists only when the graph
# is connected and has either 0 or exactly 2 odd-degree vertices (Euler's
# theorem). GBFS expands the state (current_landmark, used_lanes) whose
# heuristic value is the smallest - i.e. the state closest to the goal of
# covering every lane.
#
# Data structures used (with overflow/underflow handling as required):
#   * PriorityQueue (min-heap) - the GBFS frontier
#   * Graph (adjacency list)   - the city map
# ============================================================================

import heapq
import time
import tracemalloc
from collections import defaultdict
from typing import Any, List, Tuple, Optional


# ----------------------------------------------------------------------------
# DATA STRUCTURE: Priority Queue (min-heap) with full/empty messages
# ----------------------------------------------------------------------------
class PriorityQueue:
    """
    Bounded min-heap priority queue used as the GBFS frontier.

    Items are stored as (priority, sequence_id, payload). The min-heap pops
    the item with the LOWEST priority first; the sequence_id breaks ties in
    favour of the most recently inserted state (DFS-like exploration on
    equal heuristic values, which works well for Eulerian walks).
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("PriorityQueue capacity must be a positive integer.")
        self.capacity = capacity
        self._heap: List[Tuple] = []
        self._seq = 0   # monotonically decreasing tiebreaker -> LIFO on ties

    def is_empty(self) -> bool:
        """Return True if the queue holds no elements."""
        return len(self._heap) == 0

    def is_full(self) -> bool:
        """Return True if the queue has reached its capacity."""
        return len(self._heap) >= self.capacity

    def push(self, priority: float, payload: Any) -> bool:
        """
        Insert an item with the given priority.
        Prints a message and returns False when the queue is full (overflow).
        """
        if self.is_full():
            print(f"[PriorityQueue OVERFLOW] Cannot push item: queue is full "
                  f"(capacity = {self.capacity}).")
            return False
        self._seq -= 1   # smaller seq -> newer item pops first on tied priority
        heapq.heappush(self._heap, (priority, self._seq, payload))
        return True

    def pop(self) -> Optional[Tuple[float, Any]]:
        """
        Remove and return the (priority, payload) with the lowest priority.
        Prints a message and returns None when the queue is empty (underflow).
        """
        if self.is_empty():
            print("[PriorityQueue UNDERFLOW] Cannot pop: queue is empty.")
            return None
        priority, _, payload = heapq.heappop(self._heap)
        return priority, payload

    def size(self) -> int:
        """Return the current number of elements in the queue."""
        return len(self._heap)


# ----------------------------------------------------------------------------
# DATA STRUCTURE: Graph (undirected, weighted, adjacency list)
# ----------------------------------------------------------------------------
class Graph:
    """City map: landmarks are vertices and lanes are undirected edges."""

    def __init__(self):
        self.adj = defaultdict(list)      # vertex -> list of (neighbour, weight)
        self.vertices = set()             # set of all landmarks
        self.edges = set()                # set of unique undirected edges
        self.edge_count = 0

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> bool:
        """
        Insert an undirected lane between landmarks u and v.
        Performs basic error handling and rejects invalid/duplicate lanes.
        """
        if not u or not v:
            print("[Graph ERROR] Cannot add edge: empty landmark name.")
            return False
        if u == v:
            print(f"[Graph ERROR] Self-loop '{u}-{v}' is not allowed.")
            return False

        edge = tuple(sorted([u, v]))
        if edge in self.edges:
            print(f"[Graph WARNING] Duplicate lane '{u}-{v}' ignored.")
            return False

        self.vertices.add(u)
        self.vertices.add(v)
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))
        self.edges.add(edge)
        self.edge_count += 1
        return True

    def get_neighbors(self, vertex: str) -> List[Tuple[str, float]]:
        """Return the list of (neighbour, weight) for a landmark."""
        return self.adj.get(vertex, [])

    def get_degree(self, vertex: str) -> int:
        """Return the degree (number of incident lanes) of a landmark."""
        return len(self.adj.get(vertex, []))

    def is_valid_vertex(self, vertex: str) -> bool:
        """Return True if the landmark exists in the map."""
        return vertex in self.vertices

    def odd_degree_vertices(self) -> List[str]:
        """Return all landmarks with an odd number of incident lanes."""
        return [v for v in self.vertices if self.get_degree(v) % 2 == 1]


def build_chennai_map() -> Graph:
    """
    Build the Chennai tourist-landmark map shown in the problem statement.

    The graph below is the one drawn in the assignment PDF (and matches the
    sample output path). Weights are illustrative distances (km).
    """
    g = Graph()
    canonical = [
        ("Marina Beach",      "Mahabalipuram",     2.5),
        ("Mahabalipuram",     "Vandaloor Zoo",     3.0),
        ("Vandaloor Zoo",     "Kovalam Beach",     4.0),
        ("Kovalam Beach",     "Muttukadu",         1.5),
        ("Muttukadu",         "Government Museum", 2.8),
        ("Government Museum", "Kovalam Beach",     2.0),
        ("Kovalam Beach",     "Mahabalipuram",     2.2),
        ("Mahabalipuram",     "Government Museum", 3.5),
        ("Government Museum", "Marina Beach",      3.0),
    ]
    for u, v, w in canonical:
        g.add_edge(u, v, w)
    return g


# ----------------------------------------------------------------------------
# ALGORITHM: canonical Greedy Best First Search
# ----------------------------------------------------------------------------
class GreedyBestFirstSearch:
    """
    Greedy Best First Search over the state space
        state = (current_landmark, frozenset of used_lanes)
    using a priority queue keyed by the heuristic h(state). The first state
    popped whose used_lanes covers EVERY lane exactly once is the answer.
    """

    def __init__(self, graph: Graph, start_vertex: str):
        self.graph = graph
        self.start_vertex = start_vertex
        self.total_edges = graph.edge_count
        self.intermediate_paths: List[str] = []
        self.nodes_explored = 0
        self.peak_frontier_size = 0

    def heuristic(self, visited_edges: frozenset, current: str) -> float:
        """
        h(state) - estimate of how far this state is from the goal.

        Primary component  : number of lanes still to be covered
                             (E - |used|). Lower => closer to goal, which is
                             exactly the GBFS convention.
        Secondary component: a small term that prefers states whose CURRENT
                             landmark still has many unused incident lanes
                             - this keeps the search flexible and avoids
                             walking into stranded lanes.
                             Weight is small enough that the primary term
                             always dominates.
        """
        remaining_total = self.total_edges - len(visited_edges)
        unused_at_current = 0
        for nbr, _ in self.graph.get_neighbors(current):
            edge = tuple(sorted([current, nbr]))
            if edge not in visited_edges:
                unused_at_current += 1
        # Smaller is better. We subtract a tiny fraction so that on ties
        # in remaining_total, states with MORE unused options at the current
        # landmark win.
        return remaining_total - unused_at_current / (self.total_edges + 1.0)

    def find_path(self) -> List[str]:
        """
        Run GBFS and return the patrol path covering every lane exactly once,
        or an empty list when no such walk exists from the given start.
        """
        if not self.graph.is_valid_vertex(self.start_vertex):
            print(f"[ERROR] Starting point '{self.start_vertex}' is not on the map.")
            return []

        num_vertices = len(self.graph.vertices)
        num_edges = self.total_edges
        # Generous upper bound on reachable states: V * 2^E.
        capacity = max(1024, num_vertices * (1 << num_edges))
        frontier = PriorityQueue(capacity=capacity)

        initial_state = (self.start_vertex, frozenset(), (self.start_vertex,))
        frontier.push(self.heuristic(frozenset(), self.start_vertex), initial_state)

        while not frontier.is_empty():
            self.nodes_explored += 1
            self.peak_frontier_size = max(self.peak_frontier_size, frontier.size() + 1)

            popped = frontier.pop()
            if popped is None:
                break
            _, (current, used, path) = popped

            # Record an intermediate path snapshot for the report.
            self.intermediate_paths.append(" -> ".join(path))

            # Goal test: have we covered every lane?
            if len(used) == self.total_edges:
                return list(path)

            # Expand: push one successor per unused incident lane.
            for neighbour, _ in self.graph.get_neighbors(current):
                edge = tuple(sorted([current, neighbour]))
                if edge in used:
                    continue
                new_used = used | {edge}
                new_path = path + (neighbour,)
                priority = self.heuristic(new_used, neighbour)
                frontier.push(priority, (neighbour, new_used, new_path))

        # Queue exhausted without finding a complete walk.
        return []


# ----------------------------------------------------------------------------
# AGENT: input/output orchestration
# ----------------------------------------------------------------------------
class SurveillanceAgent:
    """Reads the starting point, runs GBFS, and writes outputPS9.txt."""

    def __init__(self, output_file: str = "outputPS9.txt"):
        self.output_file = output_file
        self.graph = build_chennai_map()
        self.start_point: Optional[str] = None

    def read_starting_point(self) -> bool:
        """
        Read the starting landmark interactively from the user.

        The prompt accepts terminal typing AND stdin redirection, so that
            python PS9_City_Surveillance_Agent.py < inputPS9.txt
        also works for batch testing.
        """
        try:
            value = input("Enter Starting point: ").strip()
        except EOFError:
            print("[ERROR] No starting point provided on standard input.")
            return False
        except Exception as exc:                # basic error handling
            print(f"[ERROR] Could not read starting point: {exc}")
            return False

        if not value:
            print("[ERROR] Starting point cannot be empty.")
            return False

        if not self.graph.is_valid_vertex(value):
            print(f"[ERROR] '{value}' is not a landmark on the map. "
                  f"Valid landmarks: {sorted(self.graph.vertices)}")
            return False

        self.start_point = value
        return True

    def run(self) -> bool:
        """Run GBFS while measuring real time and memory, then write report."""
        tracemalloc.start()
        start_time = time.perf_counter()

        gbfs = GreedyBestFirstSearch(self.graph, self.start_point)
        path = gbfs.find_path()

        elapsed = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if not path:
            print("[ERROR] No valid patrol path could be produced.")
            return False

        return self._write_report(gbfs, path, elapsed, current_mem, peak_mem)

    def _write_report(self, gbfs: GreedyBestFirstSearch, path: List[str],
                      elapsed: float, current_mem: int, peak_mem: int) -> bool:
        """Write a structured report to outputPS9.txt."""
        try:
            with open(self.output_file, "w") as f:
                f.write("CITY SURVEILLANCE AGENT - PATROL PATH REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Starting point : {self.start_point}\n")
                f.write(f"Total landmarks: {len(self.graph.vertices)}\n")
                f.write(f"Total lanes    : {self.graph.edge_count}\n")
                lanes_covered = len(path) - 1 if path else 0
                f.write(f"Lanes covered  : {lanes_covered}\n")
                f.write(f"All lanes covered exactly once: "
                        f"{lanes_covered == self.graph.edge_count}\n\n")

                f.write("INTERMEDIATE PATHS:\n")
                for idx, snap in enumerate(gbfs.intermediate_paths, 1):
                    f.write(f"  Step {idx:2d}: {snap}\n")
                f.write("\n")

                f.write("FINAL PATH:\n")
                f.write(" -> ".join(path) + "\n\n")

                f.write("SPACE AND TIME COMPLEXITY (MEASURED, NOT THEORETICAL):\n")
                f.write(f"  Actual execution time : {elapsed:.8f} seconds\n")
                f.write(f"  Current memory in use : {current_mem} bytes\n")
                f.write(f"  Peak memory allocated : {peak_mem} bytes\n")
                f.write(f"  States expanded       : {gbfs.nodes_explored}\n")
                f.write(f"  Peak frontier size    : {gbfs.peak_frontier_size}\n")
            print(f"Results written to {self.output_file}")
            return True
        except Exception as exc:                # basic error handling
            print(f"[ERROR] Could not write output file: {exc}")
            return False

    def display_results(self) -> None:
        """Echo the generated report to the console."""
        try:
            with open(self.output_file, "r") as f:
                print(f.read())
        except FileNotFoundError:
            print(f"[ERROR] Output file '{self.output_file}' not found.")


def main() -> None:
    """Program entry point."""
    print("CITY SURVEILLANCE AGENT - GREEDY BEST FIRST SEARCH")
    print("=" * 60)

    agent = SurveillanceAgent()
    print(f"Loaded city map: {len(agent.graph.vertices)} landmarks, "
          f"{agent.graph.edge_count} lanes.")

    odd = agent.graph.odd_degree_vertices()
    if len(odd) == 0:
        print("Eulerian CIRCUIT exists (every landmark has even degree). "
              "Any starting point is valid.")
    elif len(odd) == 2:
        print(f"Eulerian PATH exists. Valid starting points: {sorted(odd)}.")
    else:
        print(f"[NOTE] Graph has {len(odd)} odd-degree landmarks; an Eulerian "
              f"walk may not exist, so some lanes could remain uncovered.")

    if not agent.read_starting_point():
        print("Failed to read starting point. Exiting.")
        return

    if not agent.run():
        print("Failed to run surveillance. Exiting.")
        return

    print()
    agent.display_results()


if __name__ == "__main__":
    main()
