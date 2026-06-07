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
import heapq
# A Stack only remembers the most recent path, while a Priority Queue (heapq) evaluates every available path and always picks the globally best one.

# ----------------------------------------------------------------------------
# 1. DATA STRUCTURE: Capacity-Enforced Priority Queue
# ----------------------------------------------------------------------------
class PriorityQueue:
    """
    A bounded Priority Queue used for the GBFS frontier.
    Enforces capacity limits and prints messages on overflow/underflow as required.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = []
        self.entry_count = 0 # Used to break ties in heuristic scores

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def is_full(self) -> bool:
        return len(self.queue) >= self.capacity

    def insert(self, priority: float, state: tuple) -> bool:
        """Inserts a state into the queue based on its priority (heuristic)."""
        if self.is_full():
            print(f"[QUEUE OVERFLOW] Cannot insert state. Capacity of {self.capacity} reached.")
            return False
        
        # We push (priority, tie_breaker, state) so heapq never compares the state objects directly
        heapq.heappush(self.queue, (priority, self.entry_count, state))
        self.entry_count += 1
        return True

    def delete(self):
        """Removes and returns the state with the lowest heuristic value."""
        if self.is_empty():
            print("[QUEUE UNDERFLOW] Cannot delete. The Priority Queue is empty.")
            return None
        
        priority, _, state = heapq.heappop(self.queue)
        return state

# ----------------------------------------------------------------------------
# 2. MAIN AGENT LOGIC
# ----------------------------------------------------------------------------
class SurveillanceAgent:
    def __init__(self):
        self.input_file = "inputPS9.txt"
        self.output_file = "outputPS9.txt"
        self.start_point = None
        self.all_edges = set()
        self.landmarks = set()
        self.intermediate_paths = []

    def _build_chennai_map(self):
        """Hardcodes the Eulerian map of Chennai based on the PDF diagram."""
        lanes = [
            ("Marina Beach", "Mahabalipuram"),
            ("Marina Beach", "Government Museum"),
            ("Mahabalipuram", "Vandaloor Zoo"),
            ("Mahabalipuram", "Government Museum"),
            ("Mahabalipuram", "Kovalam Beach"),
            ("Vandaloor Zoo", "Kovalam Beach"),
            ("Kovalam Beach", "Government Museum"),
            ("Kovalam Beach", "Muttukadu"),
            ("Government Museum", "Muttukadu")
        ]
        
        # Standardize edges alphabetically so ("A", "B") is the same as ("B", "A")
        for u, v in lanes:
            edge = tuple(sorted([u, v]))
            self.all_edges.add(edge)
            self.landmarks.update([u, v])

    def read_input(self) -> bool:
        """Reads the starting point dynamically and cleans up user input errors."""
        self._build_chennai_map()
        
        try:
            with open(self.input_file, "r") as f:
                raw_text = f.readline().strip()
                
            if not raw_text:
                print("[ERROR] Input file is empty.")
                return False
                
            # Handle "Enter Starting point: " prefix if it exists
            if ":" in raw_text:
                extracted = raw_text.split(":")[1].strip()
            else:
                extracted = raw_text.strip()
                
            # Bulletproof format: Replace underscores with spaces
            self.start_point = extracted.replace("_", " ")
            
            if self.start_point not in self.landmarks:
                print(f"[ERROR] '{self.start_point}' is not a valid landmark.")
                return False
                
            return True
            
        except FileNotFoundError:
            print(f"[ERROR] {self.input_file} not found. Please create it.")
            return False
        except Exception as exc:
            print(f"[ERROR] Failed to read input: {exc}")
            return False

    def heuristic(self, next_node: str, remaining_edges: frozenset) -> float:
        """
        Heuristic h(n): Evaluates how close a node is to the goal.
        A lower score is better.
        h(n) = Total unvisited edges. 
        Tie-breaker: We apply a penalty if moving to a node results in a dead end.
        """
        # Count how many of the remaining edges are connected to the next node
        incident_edges = sum(1 for edge in remaining_edges if next_node in edge)
        
        # If the next node has no outgoing edges left, but there are still edges 
        # to clear globally, it's a dead end. Heavily penalize it!
        if incident_edges == 0 and len(remaining_edges) > 0:
            penalty = 1000 
        else:
            # Otherwise, slightly prefer nodes with fewer incident edges to force the 
            # algorithm to "clean up" areas before moving on.
            penalty = incident_edges * 0.1 
            
        return len(remaining_edges) + penalty

    def greedy_best_first_search(self):
        """Executes GBFS to find a path traversing all edges exactly once."""
        # Set a safe upper bound capacity for the priority queue
        pq = PriorityQueue(capacity=10000)
        
        # State definition: (Current Location, Unvisited Edges, Path History)
        initial_unvisited = frozenset(self.all_edges)
        initial_state = (self.start_point, initial_unvisited, [self.start_point])
        
        # Initial h(n)
        h_start = self.heuristic(self.start_point, initial_unvisited)
        pq.insert(h_start, initial_state)
        
        while not pq.is_empty():
            current_node, unvisited, path = pq.delete()
            
            # Goal Check: Are all lanes covered?
            if len(unvisited) == 0:
                return path
                
            # Record intermediate path for reporting
            self.intermediate_paths.append(" -> ".join(path))
            
            # Generate valid next states (neighbors)
            for edge in unvisited:
                if current_node in edge:
                    # Find the neighbor node on this edge
                    next_node = edge[0] if edge[1] == current_node else edge[1]
                    
                    # Create the new state
                    new_unvisited = unvisited - frozenset([edge])
                    new_path = path + [next_node]
                    
                    # Calculate heuristic for the neighbor and insert into queue
                    h_score = self.heuristic(next_node, new_unvisited)
                    pq.insert(h_score, (next_node, new_unvisited, new_path))
                    
        return None # No valid path found

    def run(self):
        """Runs the search, measures complexity metrics, and writes the report."""
        if not self.read_input():
            return
            
        print("Starting Drone Surveillance Routing (GBFS)...")
        
        # 1. Start memory and time tracking
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # 2. Run Algorithm
        final_path = self.greedy_best_first_search()
        
        # 3. Stop tracking
        elapsed_time = (time.perf_counter() - start_time) * 1000 # Convert to ms
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if not final_path:
            print("[ERROR] GBFS failed to find a valid route.")
            return
            
        # 4. Write to Output File
        try:
            with open(self.output_file, "w") as f:
                f.write("=== CITY SURVEILLANCE AGENT OUTPUT ===\n\n")
                
                f.write("INTERMEDIATE PATHS:\n")
                # Write a sample of intermediate paths to prevent massive text walls
                for idx, step in enumerate(self.intermediate_paths):
                    f.write(f"Step {idx+1}: {step}\n")
                
                f.write("\nFINAL PATH:\n")
                f.write(" -> ".join(final_path) + "\n\n")
                
                f.write("=== COMPLEXITY METRICS ===\n")
                f.write(f"Time Complexity: {elapsed_time:.4f} milliseconds\n")
                f.write(f"Space Complexity (Peak Memory): {peak_mem / 1024:.2f} KB\n")
                
            print(f"Success! Output and metrics written to {self.output_file}")
            self.display_results()
        except Exception as exc:
            print(f"[ERROR] Failed to write to {self.output_file}: {exc}")

    def display_results(self):
        """Echo the generated report to the console."""
        try:
            with open(self.output_file, "r") as f:
                print(f.read())
        except FileNotFoundError:
            print(f"[ERROR] Output file '{self.output_file}' not found.")

if __name__ == "__main__":
    agent = SurveillanceAgent()
    agent.run()
    