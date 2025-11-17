import random
import networkx as nx

class MDSIteratedGreedy:
    
    def __init__(self, graph, destruction_rate=0.3, max_no_improvement=50):
        self.graph = graph
        self.destruction_rate = destruction_rate
        self.max_no_improvement = max_no_improvement
        self.vertices = set(graph.nodes())
        self.adjacency = {}
        
        for v in self.vertices:
            self.adjacency[v] = set(graph.neighbors(v))
        
        self.support_vertices, self.leaf_vertices = self._find_support_and_leaf_vertices()
        
    def _find_support_and_leaf_vertices(self):
        leaf_vertices = set()
        support_vertices = set()
        
        for v in self.vertices:
            neighbors = self.adjacency[v]
            if len(neighbors) == 1:
                leaf_vertices.add(v)
                support_vertex = next(iter(neighbors))
                support_vertices.add(support_vertex)
        
        return support_vertices, leaf_vertices
    
    def _get_dominated_vertices(self, solution):
        dominated = set(solution)
        for v in solution:
            dominated.update(self.adjacency[v])
        return dominated
    
    def _is_dominating_set(self, solution):
        dominated = self._get_dominated_vertices(solution)
        return dominated == self.vertices
    
    def _remove_redundant_vertices(self, solution):
        minimal_solution = set(solution)
        
        for v in list(minimal_solution):
            temp_solution = minimal_solution - {v}
            
            if self._is_dominating_set(temp_solution):
                minimal_solution = temp_solution
        
        return minimal_solution
    
    def greedy_insertion_procedure(self, partial_solution=None):
        solution = set(self.support_vertices) if partial_solution is None else partial_solution
        dominated = self._get_dominated_vertices(solution)
        undominated = self.vertices - dominated
        
        available_vertices = self.vertices - self.leaf_vertices - solution
        
        while undominated:
            best_vertex = None
            best_coverage = -1
            
            for v in available_vertices:
                new_coverage = len(self.adjacency[v] & undominated)
                if v in undominated:
                    new_coverage += 1
                
                if new_coverage > best_coverage:
                    best_coverage = new_coverage
                    best_vertex = v
            
            if best_vertex is None:
                break
                
            solution.add(best_vertex)
            available_vertices.remove(best_vertex)
            
            dominated = self._get_dominated_vertices(solution)
            undominated = self.vertices - dominated
        
        solution = self._remove_redundant_vertices(solution)
        return solution
    
    def random_destruction(self, solution):
        destruction_size = max(1, int(len(solution) * self.destruction_rate))
        solution_list = list(solution)
        
        if destruction_size >= len(solution_list):
            destroyed_solution = set(random.sample(solution_list, 1))
        else:
            removed_vertices = set(random.sample(solution_list, destruction_size))
            destroyed_solution = solution - removed_vertices
        
        return destroyed_solution
    
    def greedy_reconstruction(self, partial_solution):
        solution = set(partial_solution)
        
        return self.greedy_insertion_procedure(solution)
    
    def solve(self):
        current_solution = self.greedy_insertion_procedure()
        best_solution = set(current_solution)
        best_size = len(best_solution)
        
        iterations_without_improvement = 0
        
        while iterations_without_improvement < self.max_no_improvement:
            destroyed_solution = self.random_destruction(current_solution)
            new_solution = self.greedy_reconstruction(destroyed_solution)
            new_size = len(new_solution)
            
            if new_size < best_size:
                best_solution = set(new_solution)
                best_size = new_size
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            
            current_solution = new_solution
            
        
        return best_solution


def create_test_graph():
    graph = nx.Graph()
    """
        A -> 0
        B -> 1
        C -> 2
        D -> 3
        E -> 4
        G -> 5
        H -> 6
        I -> 7
        J -> 8
    """
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (4, 5), (4, 8),
        (5, 7), (8, 6), (6, 7)
    ]
    graph.add_edges_from(edges)
    return graph

def main():
    graph = create_test_graph()
    print(f"Graph with {graph.number_of_nodes()} vertices and {graph.number_of_edges()} edges")
    
    ig_solver = MDSIteratedGreedy(
        graph=graph,
        destruction_rate=0.3,
        max_no_improvement=50
    )
    
    result = ig_solver.solve()
    
    print(f"Solution: {sorted(result)}")
    
    dominated = set(result)
    for v in result:
        dominated.update(graph.neighbors(v))
    
    print(f"All vertices dominated: {dominated == set(graph.nodes())}")


if __name__ == "__main__":
    main()
