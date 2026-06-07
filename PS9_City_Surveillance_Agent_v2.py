# ============================================================================
# PS9 - City Surveillance Agent
# Greedy Best First Search (GBFS) for the Route Inspection problem
# (cover every lane/edge exactly once - an Eulerian path/circuit).
#
# Group: G041
# Course: MTech AIML - AIMLCZG557/AECLZG557
#
# The drone must traverse every lane (edge) between landmarks (vertices)
# without repeating any lane, while landmarks may be visited more than once.
# A solution that covers all edges exactly once exists only when the graph
# has an Eulerian path/circuit (0 or 2 odd-degree vertices). The GBFS
# heuristic guides the choice of the next lane at every step, and an explicit
# Stack is used to backtrack so that ALL lanes are guaranteed to be covered.
# ============================================================================

import time
import tracemalloc
from collections import defaultdict
from typing import List, Tuple, Optional


# ----------------------------------------------------------------------------
# DATA STRUCTURE: Stack (with full/empty messages and basic error handling)
# ----------------------------------------------------------------------------
class Stack:
    """
    A bounded LIFO stack used by the search to backtrack along the walk.

    Every insert (push) and delete (pop) operation reports an appropriate
    message when the stack is full or empty, as required by the problem
    statement.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Stack capacity must be a positive integer.")
        self.capacity = capacity
        self._items: List = []

    def is_empty(self) -> bool:
        """Return True if the stack holds no elements."""
        return len(self._items) == 0

    def is_full(self) -> bool:
        """Return True if the stack has reached its capacity."""
        return len(self._items) == self.capacity

    def push(self, item) -> bool:
        """
        Insert an item on top of the stack.
        Prints a message and returns False if the stack is full (overflow).
        """
        if self.is_full():
            print(f"[Stack OVERFLOW] Cannot push '{item}': stack is full "
                  f"(capacity = {self.capacity}).")
            return False
        self._items.append(item)
        return True

    def pop(self):
        """
        Remove and return the top item.
        Prints a message and returns None if the stack is empty (underflow).
        """
        if self.is_empty():
            print("[Stack UNDERFLOW] Cannot pop: stack is empty.")
            return None
        return self._items.pop()

    def peek(self):
        """Return (without removing) the top item, or None if empty."""
        if self.is_empty():
            print("[Stack EMPTY] Cannot peek: stack is empty.")
            return None
        return self._items[-1]

    def size(self) -> int:
        """Return the current number of elements in the stack."""
        return len(self._items)


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

    # --- The following helper is for testing/printing only -------------------
    # def __str__(self) -> str:
    #     out = "Adjacency list:\n"
    #     for v in sorted(self.vertices):
    #         out += f"  {v}: {self.adj[v]}\n"
    #     return out


# ----------------------------------------------------------------------------
# ALGORITHM: Greedy Best First Search guided Eulerian traversal
# ----------------------------------------------------------------------------
class GreedyBestFirstSearch:
    """
    Cover every lane exactly once using a GBFS-guided walk.

    At each landmark the agent looks at all not-yet-used lanes and uses a
    heuristic to greedily pick the most promising next lane. An explicit
    Stack lets the agent backtrack when it reaches a landmark with no unused
    lanes, splicing sub-tours together (Hierholzer-style) so that EVERY lane
    is eventually covered. Backtracking never re-uses a lane.
    """

    def __init__(self, graph: Graph, start_vertex: str):
        self.graph = graph
        self.start_vertex = start_vertex
        self.visited_edges = set()
        self.total_edges = graph.edge_count
        self.intermediate_paths: List[str] = []   # snapshots for the report
        self.nodes_explored = 0

    def heuristic(self, neighbour: str) -> float:
        """
        Heuristic h(n): estimate of how 'useful' moving to `neighbour` is.

        h(n) = (unused lanes still incident to neighbour)
               + 1 / (1 + number of lanes already used at neighbour)

        Fewer remaining unused lanes at a candidate => closer to finishing
        that landmark, so a LOWER value is preferred (greedy best-first).
        The second term breaks ties in favour of landmarks we have touched
        less, keeping the walk flexible.
        """
        unused_incident = 0
        used_incident = 0
        for nbr, _ in self.graph.get_neighbors(neighbour):
            edge = tuple(sorted([neighbour, nbr]))
            if edge in self.visited_edges:
                used_incident += 1
            else:
                unused_incident += 1
        return unused_incident + 1.0 / (1 + used_incident)

    def _select_next(self, current: str) -> Optional[Tuple[str, tuple]]:
        """Pick the unused lane from `current` with the best (lowest) h(n)."""
        best_next, best_edge, best_h = None, None, float("inf")
        for neighbour, _ in self.graph.get_neighbors(current):
            edge = tuple(sorted([current, neighbour]))
            if edge in self.visited_edges:
                continue
            h = self.heuristic(neighbour)
            if h < best_h:
                best_h, best_next, best_edge = h, neighbour, edge
        if best_next is None:
            return None
        return best_next, best_edge

    def find_path(self) -> List[str]:
        """
        Build a walk that covers every lane exactly once.

        Returns the ordered list of landmarks visited (the final patrol path).
        Uses a Stack for backtracking; the resulting walk is an Eulerian
        path/circuit when one exists.
        """
        if not self.graph.is_valid_vertex(self.start_vertex):
            print(f"[ERROR] Starting point '{self.start_vertex}' not on the map.")
            return []

        # Stack capacity: a walk over E edges visits at most E+1 landmarks.
        stack = Stack(capacity=self.total_edges + 1)
        circuit: List[str] = []

        stack.push(self.start_vertex)
        while not stack.is_empty():
            self.nodes_explored += 1
            current = stack.peek()
            selection = self._select_next(current)

            if selection is not None:
                # Greedily move along the best unused lane.
                next_vertex, edge = selection
                self.visited_edges.add(edge)
                stack.push(next_vertex)
                # Record an intermediate path snapshot: the active trail from
                # the start landmark to the current position (bottom -> top).
                self.intermediate_paths.append(" -> ".join(stack._snapshot()))
            else:
                # Dead-end: backtrack, committing this landmark to the circuit.
                circuit.append(stack.pop())

        circuit.reverse()
        return circuit

    def get_edges_covered(self) -> int:
        """Return how many distinct lanes have been traversed."""
        return len(self.visited_edges)

    def all_edges_covered(self) -> bool:
        """Return True if every lane was covered exactly once."""
        return len(self.visited_edges) == self.total_edges


# Give the Stack a private snapshot helper used only for the report output.
def _stack_snapshot(self):
    return list(self._items)
Stack._snapshot = _stack_snapshot


# ----------------------------------------------------------------------------
# AGENT: input/output orchestration
# ----------------------------------------------------------------------------
class SurveillanceAgent:
    """Reads the map and starting point, runs GBFS, and writes the report."""

    def __init__(self, input_file: str = "inputPS9.txt",
                 output_file: str = "outputPS9.txt"):
        self.input_file = input_file
        self.output_file = output_file
        self.graph = Graph()
        self.start_point: Optional[str] = None

    def read_input(self) -> bool:
        """
        Read the map and starting point from the input file.

        Format (fields are comma-separated, which supports multi-word
        landmark names such as "Marina Beach"; plain whitespace separation is
        also accepted when landmark names contain no spaces):
            Line 1 : starting landmark
            Line 2 : number of lanes E
            Next E : <landmark1>, <landmark2>, <weight>
        """
        try:
            with open(self.input_file, "r") as f:
                lines = [ln.rstrip("\n") for ln in f.readlines()]
        except FileNotFoundError:
            print(f"[ERROR] Input file '{self.input_file}' not found.")
            return False
        except Exception as exc:                       # basic error handling
            print(f"[ERROR] Could not read input file: {exc}")
            return False

        # Drop trailing blank lines.
        while lines and lines[-1].strip() == "":
            lines.pop()

        if len(lines) < 2:
            print("[ERROR] Input must have a starting point and a lane count.")
            return False

        self.start_point = lines[0].strip()
        if not self.start_point:
            print("[ERROR] Starting point (line 1) is empty.")
            return False

        try:
            num_edges = int(lines[1].strip())
        except ValueError:
            print("[ERROR] Line 2 must be an integer (number of lanes).")
            return False

        if len(lines) - 2 < num_edges:
            print(f"[ERROR] Declared {num_edges} lanes but found "
                  f"{len(lines) - 2}.")
            return False

        for i in range(2, 2 + num_edges):
            raw = lines[i].strip()
            if not raw:
                print(f"[ERROR] Lane on line {i + 1} is empty.")
                return False
            # Comma-separated fields support multi-word landmark names
            # (e.g. "Marina Beach"); fall back to whitespace for single-word
            # names so older input files keep working.
            if "," in raw:
                parts = [p.strip() for p in raw.split(",") if p.strip()]
            else:
                parts = raw.split()
            if len(parts) < 2:
                print(f"[ERROR] Lane on line {i + 1} needs two landmarks.")
                return False
            u, v = parts[0], parts[1]
            weight = 1.0
            if len(parts) >= 3:
                try:
                    weight = float(parts[2])
                except ValueError:
                    print(f"[WARNING] Bad weight on line {i + 1}; using 1.0.")
            self.graph.add_edge(u, v, weight)

        if not self.graph.is_valid_vertex(self.start_point):
            print(f"[ERROR] Starting point '{self.start_point}' is not part "
                  f"of any lane.")
            return False
        return True

    def run(self) -> bool:
        """Run GBFS while measuring real time and memory, then write report."""
        # --- Measure ACTUAL space and time (not theoretical) ----------------
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

    def _write_report(self, gbfs, path, elapsed, current_mem, peak_mem) -> bool:
        try:
            with open(self.output_file, "w") as f:
                f.write("CITY SURVEILLANCE AGENT - PATROL PATH REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Starting point : {self.start_point}\n")
                f.write(f"Total landmarks: {len(self.graph.vertices)}\n")
                f.write(f"Total lanes    : {self.graph.edge_count}\n")
                f.write(f"Lanes covered  : {gbfs.get_edges_covered()}\n")
                f.write(f"All lanes covered exactly once: "
                        f"{gbfs.all_edges_covered()}\n\n")

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
                f.write(f"  Landmarks explored    : {gbfs.nodes_explored}\n")
            print(f"Results written to {self.output_file}")
            return True
        except Exception as exc:                       # basic error handling
            print(f"[ERROR] Could not write output file: {exc}")
            return False

    def display_results(self):
        """Echo the generated report to the console."""
        try:
            with open(self.output_file, "r") as f:
                print(f.read())
        except FileNotFoundError:
            print(f"[ERROR] Output file '{self.output_file}' not found.")


def main():
    """Program entry point."""
    print("CITY SURVEILLANCE AGENT - GREEDY BEST FIRST SEARCH")
    print("=" * 60)

    agent = SurveillanceAgent()
    if not agent.read_input():
        print("Failed to read input. Exiting.")
        return

    print(f"Loaded {len(agent.graph.vertices)} landmarks and "
          f"{agent.graph.edge_count} lanes.")
    odd = agent.graph.odd_degree_vertices()
    if len(odd) not in (0, 2):
        print(f"[NOTE] Graph has {len(odd)} odd-degree landmarks; an Eulerian "
              f"path may not exist, so some lanes could remain uncovered.")

    if not agent.run():
        print("Failed to run surveillance. Exiting.")
        return

    print()
    agent.display_results()


if __name__ == "__main__":
    main()
