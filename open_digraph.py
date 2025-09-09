import random
class node :
    def _init_(self, identity, label, parents, children):
        
        """
        identity: int; its unique id in the graph
        label: string;
        parents: int->int dict; maps a parent nodes id to its multiplicity
        children: int-> int dict; map a child nodes id to its multiplicity
        """

        self.id = identity
        self.label = label
        self.parents = parents
        self.children = children
    
    def _str_(self):
        """Return a string representation of the node."""
        return f"Node:\n Id : {self.id}, Label : {self.label}"

    def _repr_(self):
        """Return a string """
        return f"Node: \n Id: {self.id}, Label : {self.label} \n Parents : {self.parents} \n Children : {self.children}"
    
    def copy(self):
        """
        Create a copy of the node.

        Returns:
        - node: Copy of the current node.
        """
        n  = node()
        n._init_(self.id, self.label,self.parents.copy(), self.children.copy())
        return n
    
    def get_id(self):
        """
        Return the id of the node.
        """
        return self.id
    
    def get_label(self):
        """
        Return the label of the node.
        """
        return self.label

    def get_parents(self):
        """
        Return the parents of the node.
        """
        return self.parents
    
    def get_children(self):
        """
        Return the children of the node.
        """
        return self.children
    
    def set_id(self, newId):
        """
        Set a new id for the node.

        Parametres: 
        - newId: int
        """
        self.id = newId
    
    def set_label(self, newLabel):
        """
        Set a new label for the node.
        Parametres: 
        - newLabel: string
        """
        self.label = newLabel
        print(self.label)
    
    def add_parents(self, newP, multiplicity):
        """
        Add a parent node with multiplicity.
        Parametres: 
        - newP: int
        - multiplicity: int
        """
        self.parents[newP] = multiplicity

    def add_child_id(self, newC, multiplicity):
        """
        Add a child node with multiplicity.
        Parametres: 
        - newC: int
        - multiplicity: int
        """
        self.children[newC] = multiplicity

    def remove_parent_once (self, parents):
        """
        remove 1 occurence of the id

        Parametres: 
        - parents: int

        """
        multiplicity = self.parents[parents]
        while multiplicity > 0:
            del self.parents[parents]
            multiplicity = multiplicity - 1

    def remove_children_once (self, children):
        """
        remove 1 occurence of the id
        Parametres: 
        - children: int

        """
        multiplicity = self.children[children]
        while multiplicity > 0:
            del self.children[children]
            multiplicity = multiplicity - 1
    
    def remove_parent_id(self, parents):
        """
        remove all occurences of the id

        Parametres: 
        - parents: int

        """
        del self.parents[parents]
        
    
    def remove_children_id(self, children):
        """
        remove all occurences of the id
        Parametres: 
        - children: int

        """
        del self.children[children]
        
        
    def indegree(self):
        """
        Returns the indegree of the node
        """
        return len(self.parents)
    
    def outdegree(self):
        """
        Returns the outdegree of the node 
        """
        return len(self.children)

    def degree(self):
        """
        Returns the degree of the node (total number of edges incident on the node).
        """
        return len(self.children) + len(self.parents)


class open_digraph : #for open directed graph
    def _init_(self, inputs, outputs, nodes):

        """
        inputs: int list; the ids of the input nodes
        outputs: int list; the ids of the output nodes
        nodes: node iter;
        """

        self.inputs = inputs
        self.outputs = outputs
        self.nodes = {node.id:node for node in nodes} #self.nodes: <int, node>dict

    def _str_(self):
       """
        Return a string representation of the open graph.
       """
       return f"Open Graph: \n Inputs: {self.inputs} \n Outputs: {self.outputs} \n Nodes: {self.nodes}"
        
    def _repr_(self):
        nodes = "\n".join(node._repr_ (n) for n in self.nodes)
        return f"Open Graph: \n Inputs: {self.inputs} \n Outputs: {self.outputs} \n Nodes: {nodes} "

    @classmethod
    def empty():
        """
        Create an empty open directed graph.
        """
        val = open_digraph()
        open_digraph._init_(val , inputs = [], outputs = [], nodes = {})
        return val
    
    def copy(self):
        """
        Create a copy of the open directed graph.

        Returns:
        - d: open_digraph (Copy of the current graph.)
        """ 
        d = open_digraph()
        open_digraph._init_( d, self.inputs.copy(), self.outputs.copy(), self.get_nodes())
        return d
    
    def get_input_ids(self):
        """
        Return the ids of the input nodes.
        """
        return self.inputs
    
    def get_output_ids(self):
        """
        Return the ids of the output nodes.
        """
        return self.outputs

    def get_id_node_map(self):
        """
        Return a dictionary mapping node ids to node instances.
        """
        return self.nodes
       
    
    def get_nodes(self):
       """
       Return a list of all nodes.
       """
       return  [n for n in self.nodes.values()]
    
    def get_nodes_ids(self):
        """
        Return a list of all node ids.
        """
        return [n for n in self.nodes.keys()]
    
    def get_nodes_by_id(self, id):
        """
        Return the node instance with the given id.
        Parametres: 
        - id: int
        """
        return self.nodes[id]
    
    def get_nodes_by_ids(self, ids):
        """
        Return a list of node instances with the given ids.
        Parametres: 
        - ids: int
        """
        return [self.nodes.get(k) for k in ids]
    
    def set_inputs(self, Input):
        """
        Set a new list of input ids.
        Parametres: 
        - Imput: int list
        """
        self.inputs = Input

    def set_outputs(self, Output):
        """
        Set a new list of output ids.
        Parametres: 
        - Puts: int list
        """
        self.outputs = Output

    def add_input_id(self, nexInput):
        """
        Add a new input id.
        Parametres: 
        - nexInput: int 
        """
        self.inputs.append(nexInput)

    def add_output_id(self, nexOutput):
        """
        Add a new output id.
        Parametres: 
        - nexOutput: int list
        """
        self.outputs.append(nexOutput)

    def new_id(self):
        """
        Return an unused ID in the graph.
        """
        used_ids = set(self.inputs + self.outputs + list(self.nodes.keys()))
        unused_id = 1
        while unused_id in used_ids:
            unused_id = unused_id + 1
        return unused_id

    def add_edge(self, src, tgt):
        """
        which adds an edge from the src id node to the tgt id node.
        Parametres: 
        - src: int 
        - tgt: int
        """
        if src not in self.nodes or tgt not in self.nodes:
            raise ValueError("Les nœuds source et cible n'existent pas dans le graphe.")
        self.nodes[src].add_child_id(tgt, multiplicity=1)
        self.nodes[tgt].add_parents(src, multiplicity=1)


    def add_edges(self, edges):
        """
        edges is a list of pairs of node ids, and adds an edge between each of these pairs
        """
        for src, tgt in edges:
            self.add_edge(src, tgt)


    def add_node(self, label='', parents=None, children=None):
        """
        adds a node (with label) to the graph (using a new id), and links it with the parent and child id nodes (with their respective multiplicities).
        If the default values for parents and/or children are None, assign them an empty dictionary. Return the id of the new node
        """
        new_node_id = self.new_id()
        new_node = node()
        print(children)
        if parents == None or parents == {}:
            parents = {}
        else :
            for (id_par, mult) in parents.items():
                parent = self.nodes[id_par]
                parent.add_child_id(new_node_id, mult)

        if children == {} or children == None :
            children = {}
        else: 
            for (id_child, mult) in children.items():
                child = self.nodes[id_child]
                child.add_parents(new_node_id, mult)

        new_node._init_(new_node_id, label, parents, children)
        self.nodes[new_node_id] = new_node

        return new_node_id
    

    def remove_edge(self, src, tgt):
        """
        Remove a single edge

        Parameters:
        - src: int   
        - tgt: int
        """
        if src in self.nodes.keys() :
            del self.nodes[src]
        else:
            if tgt in self.nodes.keys():
                del self.nodes[tgt]
    
    def remove_parallel_edges(self, src, tgt):
        """
        Remove all the edges from src to tgt

        Parameters:
        - src: int
        - tgt: int 
        """

        if src in self.nodes.keys() and tgt in self.nodes.keys():
            while src in self.nodes[tgt].parents:
                self.nodes[tgt].parents.remove(src)

            while tgt in self.nodes[src].children:
                self.nodes[src].children.remove(tgt)

    def remove_node_by_id(self, node_id):
        """
        Remove the edges associates with the specified node

        Parameters:
        - node_id: int;     
        """


        node = self.nodes[node_id]
        if node.get_parents() != {}:
                for parent_id in node.get_parents().kays():
                    parent = self.nodes[parent_id]
                    parent.remove_children_id(node_id)
        if node.get_children() != {}:
                for child_id in node.get_children().kays():
                    child = self.nodes[child_id]
                    child.remove_parent_id(node_id)

        if node_id in self.outputs:
            del self.outputs[node_id]
        if node_id in self.inputs:
            del self.inputs[node_id]

        del self.nodes[node_id]   


                
    def remove_edges(self, edges):
        """
        Remove multiple edges from the graph.

        Parameters:
        - edges: tuple;
        """
        for src, tgt in edges:
            self.remove_edge (src, tgt)
    
    def remove_several_parallel_edges(self, edges):
        """
        Remove parallel edges for multiple edge pairs from the graph.

        Parameters:
        - edges: tuple
        """
        for src, tgt in edges:
            self.remove_parallel_edges(src, tgt)
    
    def remove_node_by_id(self, node_id):
        """
        Remove the edges associates with the specified node

        Parameters:
        - node_id: int;     
        """
        node = self.nodes[node_id]
        if node.get_parents() != {}:
            for parent_id in node.get_parents().keys():
                parent = self.nodes[parent_id]
                parent.remove_children_id(node_id)
        if self.nodes[node_id].parents != {}:
            for child_id in node.get_children().keys():
                child = self.nodes[child_id]
                child.remove_parent_id(node_id)
        
        if node_id in self.outputs:
            del self.outputs[node_id]
        if node_id in self.inputs:
            del self.inputs[node_id]
            
        del self.nodes[node_id]

    ###### Modification de exo 1 Tp 11
    def is_well_formed(self):
        """
            Fonction that verifay if a graph is well formed
        """
        
        for i in self.inputs :
            if i not in self.nodes.keys():
                return False
            if len(self.nodes[i].get_parents()) != 0  :
                return False
            if len(self.nodes[i].get_children()) != 1 and self.nodes[i].get_children().values() == 1:
                return False
            
      
        for i in self.outputs :
            if i not in self.nodes.keys():
                return False
            if len(self.nodes[i].get_parents()) != 1 and self.nodes[i].get_parents().values() == 1:
                return False
            if len(self.nodes[i].get_children()) != 0:
                return False

 
        for node_id, node in self.nodes.items():
            for child_id, multiplicity in node.get_children().items():
                child_node = self.nodes[child_id]
                if node_id not in child_node.get_parents() or child_node.get_parents()[node_id] != multiplicity:
                    return False   
     

        for node_id, node in self.nodes.items():
            if node_id != node.get_id(): return False

       
        return True
    

    def add_input_node(self, id_children, label = ''):
        """
        Add a new input node pointing to the specified node.

        Parameters:
        - id_perent: int;
        - label: str; (default is an empty string)
        """

        if id_children not in self.nodes.keys():
            raise ValueError("The parent node does not exist.")

        new_node = node()
        node._init_(new_node, identity = self.new_id(), label= label, parents={}, children={id_children:1})
        
        self.nodes[new_node.id] = new_node
        self.inputs.append(new_node.id)
        self.nodes[id_children].add_parents(new_node.id, 1)


    def add_output_node(self, id_parent, label = ''):
        """
        Add a new output node pointing to the specified node.

        Parameters:
        - parent_id: int; 
        - label: str;(default is an empty string).
        """
        if id_parent not in self.nodes:
            raise ValueError("The parent node does not exist.")

        new_node = node()
        node._init_(new_node, identity = self.new_id(), label= label, parents={id_parent:1}, children={})
        
        self.nodes[new_node.id] = new_node
        self.outputs.append(new_node.id)
        self.nodes[id_parent].add_child_id(new_node.id, 1)

    def dictionnary_id_graph(self):

            """
            Returns a dictionary associating each node id with a unique integer.
            """
            node_indices = {node_id: i for i, node_id in enumerate(self.nodes)}
            return node_indices


    def random_int_list(n, bound):
            """
            generates a list of size n containing (random) integers between 0 and bound

            Parameters:
            - n: int;
            - bound: int;
            """
            return [random.randint(0, bound - 1) for _ in range(n)]
        
  
    def random_int_matrix (n, bound, null_diag=False):
            """
            generates a matrix nxn

            Parameters:
            - n: int;
            - bound: int;
            - null_diag: bool; 
            """
            matrix = [open_digraph.random_int_list(n, bound) for _ in range(n)]

            if (null_diag):
                for i in range (n):
                    matrix[i][i] = 0
            
            return matrix


    def random_symmetric_int_matrix(n, bound, null_diag = False):
            """
            generates a symmetric matrix with the option of making the diagonal zero
            
            Parameters:
            - n: int;
            - bound: int;
            - null_diag: bool; 
            """

            matrix = [open_digraph.random_int_list(n, bound) for _ in range(n)]

            for i in range (n):
                for j in range (n):
                    matrix[i][j] = matrix[j][i]

            if null_diag:
                for i in range(n):
                    matrix[i][i] = 0

            return matrix
        

    def random_oriented_int_matrix(n, bound, null_diag=False):
            """
            generates a oriented matrix

            Parameters:
            - n: int;
            - bound: int;
            - null_diag: bool; 
            """
            matrix =  [open_digraph.random_int_list(n, bound) for _ in range(n)]

            for i in range(n):
                for j in range(i+1, n):
                    if matrix[i][j] > 0 and matrix[j][i] > 0:
                        if random.choice([True, False]):
                            matrix[i][j] = 0
                        else:
                            matrix[j][i] = 0
            if null_diag:
                for i in range(n):
                    matrix[i][i] = 0

            return matrix


    def random_dag_int_matrix(n, bound, null_diag=False):
            """
            generates a DAG matrix

            Parameters:
            - n: int;
            - bound: int;
            -null_diag: bool;
            """

            matrix = [[0] * n for i in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if random.choice([True, False]):
                        matrix[i][j] = random.randint(0, bound - 1)

            if null_diag:
                for i in range(n):
                    matrix[i][i] = 0

            for i in range(n):
                for j in range(i + 1, n):
                    if matrix[i][j] > 0 and matrix[j][i] > 0:
                        if random.choice([True, False]):
                            matrix[i][j] = 0
                        else:
                            matrix[j][i] = 0

            return matrix

 
    def graph_from_adjacency_matrix(matrix):
        """
        Generate an open directed graph from the given adjacency matrix.

        Parameters:
        - matrix (list of lists): The adjacency matrix representing the graph.

        Returns:
        - open_digraph: The open directed graph generated from the adjacency matrix.
        """
        g = open_digraph()
        lst_node = []
        for i in range(len(matrix)):
            n = node()
            node._init_(n, i, str(i), {}, {})
            lst_node.append(n)

        for i in range(0, len(matrix)):
            for j in range(0,len(matrix[i])):
                if matrix[i][j]!=0:
                    lst_node[i].add_child_id(j, matrix[i][j])
                    lst_node[j].add_parents(i, matrix[i][j])
        
        
        inputs= []
        outputs = []
        for i in range(len(lst_node)):
            if len(lst_node[i].get_children()) ==1 and lst_node[i].get_children().values() == 1 and len(lst_node[i].get_parents()) == 0  :
                inputs.append(i)
            if len(lst_node[i].get_parents()) ==1 and lst_node[i].get_parents().values() == 1 and len(lst_node[i].get_children()) == 0  :
                outputs.append(i)
        
        open_digraph._init_(g, inputs, outputs, lst_node )
        return g
    
    @classmethod
    def random(cls, n, bound, inputs=0, outputs=0, form="free"):
        """
        Generate a random graph according to the specified constraints.

        Parameters:
        - n: int;
        - bound: int; 
        - inputs: int; 
        - outputs: int; 
        - form: string;

        Returns:
        Graph: a random graph generated according to the constraints.
        """
       
        
        if form == "free":
            matrix = open_digraph.random_int_matrix(n, bound)
        elif form == "loop":
            matrix = open_digraph.random_int_matrix(n, bound, True)
        elif form == "undirected":
            matrix = open_digraph.random_symmetric_int_matrix(n, bound)
        elif form == "oriented":
            matrix = open_digraph.random_oriented_int_matrix(n, bound)
        elif form == "DAG":
            matrix = open_digraph.random_dag_int_matrix(n, bound)
        elif form == "loop-free undirected":
            matrix = open_digraph.random_symmetric_int_matrix(n, bound, True)

      
        g = open_digraph.graph_from_adjacency_matrix(matrix)
        
        if inputs + outputs < n:
            if inputs > 0:
                if inputs > n:
                    raise ValueError("Number of input nodes cannot exceed total number of nodes.")
                input_nodes = random.sample(range(n), inputs)
                for i in input_nodes:
                    g.add_input_node(i, str(i))
            
            if outputs > 0:
                if outputs > n:
                    raise ValueError("Number of output nodes cannot exceed total number of nodes.")
                output_nodes = random.sample(range(n), outputs)
                for i in output_nodes:
                    g.add_output_node(i, str(i))
        else:
            raise ValueError("Number of output + inputs nodes cannot exceed total number of nodes.")
        return g

    
    def  adjacency_matrix(self):
        """
        Generate the adjacency matrix of the graph.

        Returns:
        - list of lists: The adjacency matrix
        """
        matrix = []
        for n in self.nodes.values():
            ligne = [0]*(len(self.nodes))
            for (id_child, multy) in n.get_children().items():
                ligne[id_child] = multy
            matrix.append(ligne)
        return matrix
    
    @classmethod

    def from_dot_file(cls, filename):
        """
        Create an open_digraph from a .dot file.
        
        Parameters:
            - filename: str; path to the .dot file.
        
        Returns:
            open_digraph: an open_digraph object created from the .dot file.
        """
        graph = open_digraph()

        with open(filename, 'r') as file:
            lines = file.readlines()

        node_lines = [line.strip() for line in lines if '->' in line]

        for line in node_lines:
            # Extract source and target node IDs
            source, targets = line.split('->')
            source = source.strip()
            targets = targets.strip()

            # Extract multiplicity from the line
            multiplicity = 1
            if '[' in line:
                multiplicity = int(line.split('[')[1].split(']')[0])

            # Add the edge to the graph
            source_id = int(source)
            target_ids = [int(target) for target in targets.split(';')]
            for target_id in target_ids:
                graph.add_edge(source_id, target_id, multiplicity)

        return graph


    def  save_as_dot_file(self, path, verbose=False):
            """
            saves the graph in .dot format, at the location specified by path 
            Parameters :
            - path : string
            - verbose : boolean
            """

            fichier = open(path, 'w')
            fichier.write ("digraph G { \n")
            for node_id, node in self.nodes.items():
                label = node.label 
                if verbose: 
                    fichier.write (f'    v{node_id} [label= "{label}"];\n')
                else:
                    fichier.write (f'    [label="{label}"];\n')
                
            for node_id, node in self.nodes.items():
                for child_id, _ in node.children.items():
                    fichier.write(f'    v{node_id} -> v{child_id};\n')

            fichier.write("}\n")
            fichier.close()

            
    
    def min_id(self):
        """
        Return the min index of graph nodes
        """
        
        ##return min(node["index"] for node in self.nodes) if self.nodes else -1
        return min(self.nodes.keys())
    
    def max_id(self):
        """
        Return the max indew of graph nodes
        """

        ##return max(node["index"] for node in self.nodes) if self.nodes else 0
        return max(self.nodes.keys())
    
    def shift_indices(self, n):
        """
        adds an integer n (possibly negative) to all the indices in the graph
        """
        self.inputs = [i+n for i in self.inputs]
        self.outputs = [i+n for i in self.outputs]
        for node_id in self.nodes.keys():
            self.nodes[node_id].id += n
        
   
   
    def iparallel(self, g):
        """
        appends, in parallel, g to self, thereby modifying self, g is not modified
        
        Parameters:
        - g : graph
        """
        new_nodes  = self.nodes.copy()
        for (id, n) in self.nodes.items():
            if id in g.get_nodes_ids():
                #find the max index M in self
                M = self.max_id()
                #find the min index m in g
                m = g.min_id()

                new_id = id - m + M + 1
                n.set_id(new_id)
               
                new_nodes[new_id] = n
                del new_nodes[id]

                if id in self.inputs:
                    self.inputs.remove(id)
                    self.inputs.append(new_id)

                if id in self.outputs:
                    self.outputs.remove(id)
                    self.outputs.append(new_id)
                
                for (child_id, multi) in n.get_children().items():
                    new_nodes[child_id].remove_parent_id(id)
                    new_nodes[child_id].add_parents(new_id, multi)
                
                for (parent_id, multi) in n.get_parents().items():
                    new_nodes[parent_id].remove_children_id(id)
                    new_nodes[parent_id].add_child_id(new_id, multi)

        self.nodes = new_nodes

        self.nodes.update(g.get_id_node_map())
        self.inputs.extend(g.get_input_ids())
        self.outputs.extend(g.get_output_ids())          

        
        
    @classmethod
    def parallel (cls, g1, g):
        """
        Parameters:
        -g : graph
        Return:
        - g1 : graph parallel
        """
        g2 = g1.copy()
        g2.iparallel(g)
        return g2
        

    def icompose(self, f):
        """
        Perform sequential composition of self and f (inputs of self connected to outputs of f).

        Parameters:
        - f: open_digraph

        Raises:
        - ValueError: If the number of inputs from self and outputs from f do not match.
        """
        
        if len(self.inputs) != len(f.get_output_ids()):
            raise ValueError("Number of inputs from self does not match the number of outputs from f.")

       
        new_nodes  = self.nodes.copy()

        for (id, n) in self.nodes.items():
            if id in f.get_nodes_ids():
                #find the max index M in self
                M = self.max_id()
                #find the min index m in g
                m = f.min_id()

                new_id = id - m + M + 1
                n.set_id(new_id)
               
                new_nodes[new_id] = n
                del new_nodes[id]


                if id in self.inputs:
                    self.inputs.remove(id)
                    self.inputs.append(new_id)

                if id in self.outputs:
                    self.outputs.remove(id)
                    self.outputs.append(new_id)
                
                for (child_id, multi) in n.get_children().items():
                    new_nodes[child_id].remove_parent_id(id)
                    new_nodes[child_id].add_parents(new_id, multi)
                
                for (parent_id, multi) in n.get_parents().items():
                    new_nodes[parent_id].remove_children_id(id)
                    new_nodes[parent_id].add_child_id(new_id, multi)
        self.nodes = new_nodes
        self.nodes.update(f.get_id_node_map())
        self.inputs.extend(f.get_output_ids())

    @classmethod
    def compose(cls, g1, g):
        """
        Return a new graph that is the composition of self and g, without modifying them.

        Parameters:
        - g: graph

        Returns:
        - open_digraph: A new graph representing the composition of self and g.
        """
        g2 = g1.copy()
        g2.icompose(g)
        return g2
    
    @classmethod
    def identity(cls, n):
        """
        creates an open-digraph representing the identity over n children
        
        Parameters:
        -n : int
        """
        identity_graph = open_digraph.empty()
        
        for _ in range (n):
            node_id = identity_graph.add_node()
            identity_graph.add_input_id(node_id)
            identity_graph.add_output_id(node_id)
        
        return identity_graph
    
    def connected_components (self):
        """
        Find the connected components of a graph
        
        Returns:
        - dictionary; int:int the connnected_components
        - int; the number of connected components
        """
        component_dict = {}
        component_nb = 0
        node_visited = []
            
        #function to find the different connected components in the graph
        def search_component(node_id, component):
            node_visited.append(node_id)
            component_dict[node_id] = component
            for child_id in self.nodes[node_id].get_children().keys():
                if child_id not in node_visited:
                    search_component(child_id, component)
            
        #for each node we research the connected component
        for node_id in self.nodes.keys():
            #if the node has not already been visited we search from this node the connected_component 
            if node_id not in node_visited:
                search_component(node_id, component_nb)
                component_nb += 1
        
        #we return the dictionnary with all the connected_component and the number of connected_component the graph have          
        return component_dict, component_nb
                    
    def list_of_open_digraph_connected_component (self, g):
        """
        Creates a list of open_digraph, each corresponding to a connected component of the starting graph
        
        Paramters :
        -g: open_digraph; the starting graph
        
        Returns : 
        - list of open_digraph: representing the connected components of the graph
        
        """ 
        #research connected components of the graph
        c_dict, c_nb = g.connected_components()
        
        #create a list to store the connected components
        connected_components_graph = []
        for i in range(c_nb):
            # Extract nodes belonging to the current connected component
            node = [self.nodes[id] for (id, k) in c_dict.items() if k == i ]
            id_nodes = [n.get_id() for n in node]

            # Filter input and output nodes based on the component nodes
            input = [id for id in self.inputs if id in id_nodes]
            output  = [id for id in self.outputs if id in id_nodes]

            # Create a new open_digraph for the connected component
            graph = open_digraph()
            graph._init_(input, output, node)
            
            connected_components_graph.append(graph)
    
        return connected_components_graph
        
    def dijkstra (self, src, direction = None, tgt = None):
        """
        Dijkstra's algorithm
        Parameters:
        - src : the node from whoch we calculate the distances
        - direction : int {None, 1, -1}
        - tgt : node
        """
        Q = [src]
        dist = {src : 0}
        prev = {}
        
        while Q:
            u = min (Q, key = dist.get)
            Q.remove(u)
            neighbours = set()
            if direction is None:
               
                neighbours_c = u.get_children()
                neighbours_p = u.get_parents()
                
                neighbours = neighbours.union(neighbours_c)
                neighbours = neighbours.union(neighbours_p)
                
            elif direction == -1:
                neighbours = u.get_parents()
            elif direction == 1:
                neighbours = u.get_children()
            else:
                neighbours = []
            
            for v in neighbours:
                if v not in dist:
                    Q.append(v)
                #mise à jour de la distance si un chemin plus court trouvé    
                if v not in dist or dist[v] > dist [u] + 1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    
                    #si le noeud cible est atteint on arrête
                    if v == tgt:
                        return dist, prev
        return dist, prev
                
    def shortest_path(self, u, v, direction = None):
        _ , prev = self.dijkstra(u, v, direction)
        path = []
        while v in prev:
            path.insert(0, v)
            v = prev[v]
        path.insert(0, u)
        return path
        
    def common_ancestor (self, n1, n2):
        """
        Given two nodes, returns a dictionary which
        associates each common ancestor of the two nodes with its distance from each of
        the two nodes
        """
        dist1, prev1 = self.dijkstra(n1)
        dist2, prev2 = self.dijkstra(n2)
        c_ancestors = {}
        for node in self.nodes:
            if node != n1 and node != n2 and node in prev1 and node in prev2:
                distance_n1 = dist1[node]
                distance_n2 = dist2[node]
                c_ancestors[node] = (distance_n1, distance_n2)
        return c_ancestors
    
    def topo_sort(self):
        """
        method that implements this topological sorting.
        """
        sorted_nodes = []
        visited = set()
        
        #si le graph est cyclique on relève une erreur
        if self.is_cyclic() :
            raise ValueError("Graph is cyclic")
        else:
            #sinon on recherche le chemin
            def parcourt_nodes(node):
                if node not in visited:
                    
                    for child in node.get_children():
                        parcourt_nodes(child)

                    visited.add(node)
                    sorted_nodes.append(node)
                    
            #pour chaque noeuds qui n'a pas de parents
            for node in self.nodes :
                if node.get_parents() == {}:
                    parcourt_nodes(node)
            return sorted_nodes

    def depth_node (self, node):
        """
        Returns the depth of the given node
        """
        if node.id in self.inputs:
            return 0
        else:
            parents = node.parents.keys()
            max_parent_depth = max(self.depth(self.nodes[parent_id]) for parent_id in parents)
            return max_parent_depth + 1
    
    def depth_graph(self):
        """
        Returns the depth of the graph
        """
        max_depth = 0
        for node in self.nodes:
            node_depth = self.depth_node(node)
            if node_depth > max_depth:
                max_depth = node_depth
        return max_depth
    
    def longest_path (self, start, end):
        """
        Calculate the longest path and distance from node u to node v
        """
        dist = {node_id: 0 for node_id in self.nodes}
        prev = {node_id: None for node_id in self.nodes}
        for depth, node_set in enumerate(self.topo_sort()):
            if start in node_set:
                break

        for c in self.topological_sets[depth:]:
            for node_id in c:
                if node_id == end:
                    return dist[end], self._construct_path(prev, end)
                for parent_id in self.nodes[node_id].get_parents():
                    if parent_id in dist:
                        if dist[parent_id] + 1 > dist[node_id]:
                            dist[node_id] = dist[parent_id] + 1
                            prev[node_id] = parent_id

        return -1, []

    def construct_path(self, prev, end):
        path = []
        c = end
        while c is not None:
            path.append(c)
            c = prev[c]
        return list(reversed(path))

        

class  bool_circ(open_digraph):
    def _init_(self, inputs, outputs, nodes):

        """
        inputs: int list; the ids of the input nodes
        outputs: int list; the ids of the output nodes
        nodes: node iter;
        """
        super()._init_(inputs, outputs,nodes)
        if not self.is_well_formed_bool():
            raise ValueError("The provided graph is not a valid Boolean circuit.")
        
    def is_cyclic(self):
        """
        Determines whether the directed graph contains any cycles.
    
        Returns:
        bool: True if the graph contains cycles, False otherwise.
        """
        visited = set()
        def node_cyclic(node_id):
            if node_id in visited:
                return True
            else:
                visited.add(node_id)
                for child_id in self.nodes[node_id].get_children().keys():
                    if node_cyclic(child_id): 
                        return True
                visited.remove(node_id)
            return False
        
        for node in self.nodes.keys():
            if node_cyclic(node): return False
        return False
    
    def is_well_formed_bool(self):
   
        ## verifie que un node ait commme label &, |, ~, '', '0', '1', '^'
        lst_label = ['&', '|', '~', '0', '1', '', '^']
        for node in self.nodes.values():
            if node.get_label() not in lst_label:
                return False
        
        
        for node in self.nodes.values():
            if node.get_label() == '' and node.indegree() != 1:
                return False
        

        return self.is_well_formed() and not self.is_cyclic()
    
    def copy(self):
        """
        Create a copy of the open directed graph.

        Returns:
        - d: open_digraph (Copy of the current graph.)
        """ 
        d = bool_circ()
        bool_circ._init_( d, self.inputs.copy(), self.outputs.copy(), self.get_nodes().copy())
        return d

    ########################### TP 9
    
    #### question 1 et changemant 3 on a changer la fonction add edadg pour que cela il ajout les enfent e paraents
    def parse_parentheses(self, s):
        g = open_digraph()
        current_node = node(0, '', [], [])
        i = 0
        s2 = ''
        variables= set()
        for c in s:
            self.add_node(current_node)
            i += 1
            if c == '(':
                current_node.set_label(s2)
                n = node(i, '', [], [])
                g.add_edge(n, current_node)
                current_node = n
                s2 = ''
            elif c == ')':
                current_node.set_label(s2)
                n = node(i, '', [], [])
                g.add_edge(n, current_node)
                current_node = n
                s2 = ''
            else:
                s2 += c
                if c.isalpha() and c not in variables:
                    variables.append(c)
        return g, c
    
    ###### quetion 4 
    def parse_parentheses(*formulas):
        g = open_digraph()
        current_node = node()
        s2 = ''
        variables = set()
    
        for formula in formulas:
            for char in formula:
                if char == '(':
                    current_node.set_label(s2)
                    n = node()
                    current_node.add_child_id(n.get_id())
                    n.add_parents(current_node.get_id())
                    current_node = n
                    s2 = ''
                elif char == ')':
                    current_node.set_label(s2)
                    n = node()
                    current_node.add_parents(n.get_id())
                    n.add_child_id(current_node.get_id())
                    current_node = n
                    s2 = ''
                else:
                    s2 += char
                    if char.isalpha():
                        variables.add(char)
    
        return g, list(variables)


    
    #### question 2 
    def fusion_node(self, nodeid1, nodeid2):
        if nodeid1 not in self.nodes.keys() or nodeid2 not in self.nodes.keys():
            raise ValueError("L'un des node n'est pas present dans le graphe")
        
        node1 = self.nodes[nodeid1]
        node2 = self.nodes[nodeid2]

        label = input("Enter label for the merged node: ")
        
        parents = node1.get_parents().copy()
        parents.update(node2.get_parents())

        children = node1.get_children().copy()
        children.update(node2.get_children())

        self.remove_node(nodeid1)
        self.remove_node(nodeid2) 

        node1 = node()
        node1._init_(nodeid1, label, parents, children)

        self.add_node(node1)

        return node1
    


    ##################### Tp 10
    
    ##Exercice 1 & 2

    def random_bool_circ(self, nb_nodes, nb_inputs, nb_outputs):
        """
        Genère un circuit booléen aléatoirement à partir d'un graphe de taille donnée
        
        Paramètre :
        - nb_nodes : int; nombre de noeuds dans le graph
        - nb_inputs : int; nombre de inputs à générer (supérieur ou égal à 1)
        - nb_outpus : int; nombre de outputs à générer (supérieur ou égal à 1)
        
        Return :
        - bool_circ :  le circuit booléen généré aléatoirement à partir du graph de taille donnée
        """
        
        #Etape 1 : on génère un graphe dirigé acyclique sans inputs, outputs
        
        graph = open_digraph().random(n=nb_nodes, bound=100, inputs=nb_inputs, outputs=nb_outputs, form="free")
        
        #Etape 2 
        ### revoir la car au debut tout les node n'ont ni de parant di de enfent donc si on fait ça tout les node seront des input et output
        
        for node_id, node in graph.nodes.items():
            
            #on ajoute un input vers chaque noeud sans parent
            if node.indegree() == 0 :
                graph.inputs.append(node_id)
            
            #on ajoute un output depuis chaque noeud sans enfant
            if node.outdegree() == 0 :
                graph.outputs.append(node_id)

        #Etape 3
        
        for u_id, u in graph.nodes.items() :
            degI = u.indegree()
            degO = u.outdegree()
            
            if degI == 1 and degO == 1 :
                u.set_label(random.choice(['^', '~']))
            
            elif degI > 1 and degO == 1 :
                u.set_label(random.choice(['&', '|']))
            
            elif degI > 1 and degO > 1 :
               op_node = node(identity = self.max_id() + 1, label = random.choice(['&', '|']), parents = {}, children = {})
               cp_node = node(identity = self.max_id() + 2, label = '', parents = {}, children = {})
            
            #les parents de u deviennent les parents de Uop
            for parent_id, _ in u.get_parents().items() :
                
                #REVOIR POUR LA MULTIPLICITE 
                
                op_node.add_parents(parent_id, 1)
            
            #les enfants de u deviennent les enfants de Ucp
            for child_id, _ in u.get_children().items() :
                
                #REVOIR POUR LA MULTIPLICITE
                
                cp_node.add_children(child_id, 1)
            
            #une flèche entre op_node et cp_node 
            op_node.add_children(cp_node.get_id(), 1)
            cp_node.add_parents(op_node.get_id(), 1)
            
            #les deux noeuds remplacent le noeud u
            graph.remove_node_by_id[u_id]
            graph.add_node(op_node)
            graph.add_node(cp_node)
            
        return graph
    
    ##Exercice 3
    
    def half_adder (self, taille_registre):
        """
        construise le circuit half_Adder en fonction de la taille du registre donné
        
        Paramètre :
        - taille_registre : int; 
        
        Return:
        - bool_circ : circuit de half-Adder
        """

        graph = open_digraph()
        
        for i in range (taille_registre*2):
            #on fait le noeud de la somme et celui de la retenu
            somme = node(idendity = i, label = '^', parents={}, children ={})
            carry = node(identity = i+1, label='', parents={}, children={})
            #on les relie entres eux
            somme.add_child_id(i+1, 1)
            carry.add_parents(i, 1)
            #on les ajoute au graph
            graph.add_node(somme)
            graph.add_node(carry)
        
        return graph
        
    def adder (self, taille_registre):
        """
        Construit le circuit Adder en fonction de la taille du registre donné
        
        Paramètre :
        - taille_registre : int; 
        
        Return:
        - bool_circ : circuit de Adder
        """
        graph = open_digraph()
        

    ######################### TP 11

    ### Exercice 2
    def registre_entier(k, taille = 8):
        """
        Construit un graphe à partir de la representation binaire d'un nomlbre

        Paramètre:
        - k :int
        - taillr : la taille du registre

        Return:
        - bool_circ
        """
        graph  = bool_circ.empty()
        nb_bin = bin(k)[2:].zfill(taille)


        for i in range(taille):
            node_label = nb_bin[i]
            node = node(i, node_label, {}, {None})
            graph.add_node(node)
            graph.add_input_id(i)

        return graph
    
    ### Exercice 2
    def copies(self, node_id):
        n = self.nodes[node_id] 
        parents = n.get_parents().copy()
        children = n.get_children().copy()
        self.add_node(n.get_lebel(), parents, children)
        
    
    def porte_non(self, node):
        if node.indegree() == 1 :
            for id_p, m in node.get_parents().items(): 
                node_p = id_p
                mult = m
            parent = self.nodes[node_p]
            if parent.indegree() == 0 and mult ==  1 :
                if parent.get_label() == '1':
                    node.set_label('0')   
                   
                elif node.get_label() == '0' : 
                    node.set_label('1')
                return node_p
        return None
        

    def port_et(self, node):
        nodes_to_add = []
        for (node_id, mult) in node.get_parents().items():
            node_p = self.nodes[node_id]
            if node_p.get_label() == '0' and mult == 1 and node_p.indegree() == 0:
                for id, mult in node.get_parents().items():
                    if id != node_id:
                        nodes_to_add.append(('', {id:mult}, {}))
                node.set_label('0')
                return node_id, nodes_to_add
            if  node_p.get_label() == '1' and node_p.indegree() == 0 and mult == 1:
                return node_id, None
        return None, None
    
    def port_ou(self, node):
        nodes_to_add = []
        for (node_id, mult) in node.get_parents().items():
            node_p = self.nodes[node_id]
            if node_p.get_label() == '1' and mult == 1 and node_p.indegree() == 0:
                for id, mult in node.get_parents().items():
                    if id != node_id:
                        nodes_to_add.append(('', {id:mult}, {}))
                node.set_label('0')
                return node_id, nodes_to_add
            if  node_p.get_label() == '0' and node_p.indegree() == 0 and mult == 1:
                return node_id, None
        return None, None
    
    def port_ou_exclusif(self, node):
        for (id_p, mult) in node.get_parents().items():
            parent = self.nodes[id_p]
            if parent.get_label() == '0' and parent.indegree() == 0 and mult == 1:
                return id_p, None

            elif parent.get_label() == '1' and parent.indegree() == 0 and mult == 1:
                children =  node.get_children()
                for child_id in children.keys():
                    child = self.nodes[child_id]
                    child.remove_parent_id(node.get_id())
                    node.remove_children_id(child_id)
                if children == {}: children = None
                return id_p, [('~', {node.get_id() : 1}, children)]
        return None, None
                
                
    def evaluate(self):
        lst_remove = []
        lst_add =[]
        g2 = self.copy()
        for node in self.nodes.values():
            if node.get_label() == '~':
                node1 = self.porte_non(node)
                lst_remove.append(node1)

            elif node.get_label() == '&':
                ## On verifi si c'est un element neutre
                if node.indegree() == 0 and node.outdegree()== 1:
                    node.set_label('1') 
                else:
                    node_rmove , node_add= self.port_et(node)
                    if node_add != None: lst_add.extend(node_add)
                    lst_remove.append(node_rmove)
                  
            elif node.get_label() == '|':
                ## On verifi si c'est un element neutre
                if node.indegree() == 0 and node.outdegree()== 1:
                    node.set_label('1') 
                else:
                    node_rmove , node_add= self.port_ou(node)
                    if node_add != None: lst_add.extend(node_add)
                    lst_remove.append(node_rmove)

            elif node.get_label() == '^':
                ## On verifi si c'est un element neutre
                if node.indegree() == 0 and node.outdegree()== 1:
                    node.set_label('0') 
                else:
                    node_rmove , node_add= self.port_ou_exclusif(node)
                    if node_add != None: lst_add.extend(node_add)
                    lst_remove.append(node_rmove)

        for node_id in lst_remove:
            if node_id != None:
                self.remove_node_by_id(node_id)
        for (label, parent, child) in lst_add:
            self.add_node(label, parent, child)

    ############# Tp12
    ############# Exercice 1

    def hamming_encodeur (self):
        """
        Méthode pour construire le circuit d'encodeur de Hamming (7,4)
        """
        
        if len(self.inputs) < 4:
            raise ValueError("On a besion de 4 bit de input.")

        k = 0
        z = 1
        for i in range(3):
            if i == 3: k += 1
            if i ==2 : z = 2
            n1 = self.add_node('^', {self.inputs[k] : 1, self.inputs[z]: 1, self.inputs[3]: 1}, {})
            self.add_output_id(n1)
            self.add_output_id(self.inputs[i])
        self.add_output_id(self.inputs[3])
        

    def hamming_decoder(self):
        """
        Méthode pour construire le circuit de decodeur de Hamming (1,7)
        """
        if len(self.inputs) < 4 :
            raise ValueError("On a besion de 4 bit de input.")

        lst_node = []
        k = 0
        z = 1
        for i in range(3):
            if i == 3: k += 1
            if i ==2 : z = 2
            n1 = self.add_node('^', {self.inputs[k] : 1, self.inputs[z]: 1, self.inputs[3]: 1}, {})
            lst_node.append(n1)

        for j in range(3):
            node = lst_node[j]
            n2 = self.add_node('',{node : 1}, {})
            lst_node.append(n2)
        
        for i in range(3, 6):
            node = lst_node[i]
            n2 = self.add_node('~',{node : 1}, {})
            lst_node.append(n2)

        for i in range(4):
            ### Creation du node
            n2 = self.add_node('&',{}, {})
            lst_node.append(n2)
        k = 0
        for node_id in lst_node:
            node  = self.nodes[node_id]
            
            if node.get_label() == '&':  
                for i in range(3):
                   if 6 - k != 3 + i: 
                        n1 = self.nodes[lst_node[i+3]]    
                        node.add_parents(lst_node[i+3], 1)
                        n1.add_child_id(node_id, 1)

                if k < 3:
                    node.add_parents(lst_node[6 + k], 1)
                    n2 = self.nodes[lst_node[6 + k]]
                    n2.add_child_id(node_id, 1)
                    k+=1


        for i in range(4):
            n3 = self.add_node('^', {self.inputs[i] : 1,}, {})
            self.nodes[n3].add_parents(lst_node[9 + i], 1)
            self.nodes[lst_node[9 + i]].add_child_id(n3, 1)

    
    def associativite_OR (self):
        """
        Applique l'associativité de la porte logique OR
        
        """
        lst = []
        for node_id, node in self.nodes.items():
            #pour chaque porte logique OR trouvée
            if node.get_label() == '^':
                #on regarde si ses enfants sont également des portes logiques OR
                child = node.get_children()
                parents = node.get_parents()

                for c_id, mult in child.items():
                    #si c'est le cas, alors on associe les deux OR
                    c_node = self.nodes[c_id]
                    if c_node.get_label() == '^'  and node.outdegree() == 1:
                        #on lui rajoute les parents de la première porte puis on supprime cette porte
                        for p_id , mult in parents.items():
                            c_node.add_parents(p_id, mult)
                            self.nodes[p_id].add_child_id(c_id, mult)
                        lst.append(node_id)
                        #on sort de la boucle une fois terminé avec cett première association
                        break
        for i in lst:
            self.remove_node_by_id(i)

    def associativite_Copie (self):
        """
        Applique la transformation pour la copie d'un noeud
        
        """
        lst = []
        #pour tous les noeuds du graph on recherche les copies et on les supprime
        for node_id, node in self.nodes.items():
            if node.get_label == '':
                for c_id, mult in node.get_children():
                    c_node = self.nodes[c_id]
                    if c_node.get_label() == '' and c_node.indegree() == 1 and mult == 1:
        
                        for child_id, mult2 in c_node.get_children():
                            child = self.nodes[child_id]
                            child.add_parents(node_id, mult2)
                            node.add_child_id(child_id, mult2)
                        lst.append(c_id)
                        break
        for i in lst:
            self.remove_node_by_id(i)
        
    
    def involution_OR (self):
        """
        Applique l'involution de la porte logique OR

        """
        
        lst_node = []
        for node_id, node in self.nodes.items():
            if node.get_label() == '^':
                p = node.get_parents()
                #on regarde la multiplicite de chaque parents
                for p_id, p_m in p.items():
                    #si il y a un nombre paire d'arrêtes
                    if p_m % 2 == 0 and self.nodes[p_id].get_label() == '':
                        lst_node.append((p_id, node_id))

                       
        
        #on supprime toutes les arrêtes
        for np, n in lst_node:
            self.nodes[n].remove_parent_id(np)
            self.nodes[np].remove_children_id(n)
                    
                        

    
    def effacement (self):
        """
        Applique l'effacement

        """
        lst = []
        lst_remove = []
        for node_id, node in self.nodes.items():
            if node.get_label() in ['&', '|', '~','^'] and node.outdegree() == 1:
                for c, m in node.get_children().items():
                    chil_id = c
                    mult = m
                child = self.nodes[chil_id]
                if m == 1 and child.get_label == '' and child.outdegree() == 0:
                    for p_id, mult in node.get_parents().items():
                        lst.append(('', {p_id : mult}, {}))
                    lst_remove.append(node_id)
                    lst_remove.append(child)
        
        for label, parent, children in lst:
            self.add_node(label, parent, children)
            
        for n in lst_remove:
            self.remove_node_by_id(n)
                                
                
                
            
    
    def NON_XOR (self):
        """
        Applique le non à travers la porte logique OR
        
        Paramètre:
        - node : node; le node à changer
        
        """
        lst_remove =[]
        lst_add = []
        for node_id, node in self.nodes.items():
            #pour chaque porte logique NON trouvée
            if node.get_label() == '~':
                #on regarde si ses enfants sont des portes logiques OR
                child = node.get_children()
                for c_id, m in child.items() :
                    #si on a une porte OR, alors on donne ses parents à la porte Non, puis on change les labels des deux portes
                    c_node = self.nodes[c_id]
                    if c_node.get_label() == '^' and m ==1 and c_node.outdegree() == 1:
                        parents = c_node.get_parents()
                        for p_id, mult in parents.items():
                            if p_id != node_id:
                                lst_remove.append((c_id, p_id))
                                lst_add.append((node_id, p_id, mult))
                        node.set_label('^')
                        c_node.set_label('~')
                        
            break
        for c, p in lst_remove:
            self.nodes[c].remove_parent_id(p)
            self.nodes[p].remove_children_id(c)
        for c, p, m in lst_add:
            self.nodes[p].add_child_id(c,m)
            self.nodes[p].add_parents(p, m)

            
    def NON_Copie (self):
        """
        Applique le non à travers la copie
        
        """
        lst_remove = []
        lst_add = []
        for node_id, node in self.nodes.items():
            #lorsqu'on a l'opération logique NON
            if node.get_label() == '~' and node.outdegree() == 1:
                child = node.get_children()
                for c_id, m in child.items():
                    c_node = self.nodes[c_id]
                    if c_node.get_label() == '' and c_node.indegree() == 1:
                        lst_remove.append(c_id)
                        for c, m  in c_node.get_children().items():
                            lst_add.append(('~', {node_id : 1},{ c : m}))
                        node.set_label('')
        for c in lst_remove:
            self.remove_node_by_id(c)
        for label, parents,chidren  in lst_add:
            self.add_node(label, parents, chidren)
                
                
    
    def involution_NON (self, node):
        """
        Applique l'involution du non
 
        """
        lst_remove = []
        lst_add = []
        #pour tous les NON on regarde si on en trouve pas un dans ses enfants
        for node_id, node in self.nodes.items():
            if node.get_label() == '~' and node.outdegree() == 1 and node.intdegree() == 1:
                child = node.get_children()
                for p , m in  node.get_parents().items(): 
                    parent = p
                for c_id, m in child.items():
                    c_node =self.nodes[c_id]
                    #si on a deux Non d'affilé, on les supprime
                    if c_node.get_label() == '~' and c_node.outdegree == 1 and c_node.indegree() == 1:
                        lst_remove.append(c_node)
                        lst_remove.append(node_id)
                        for p , m in c_node.get_parents().items(): lst_add.append((parent, 1))
                        break
        for n in lst_remove:
            self.remove_node_by_id(c)
           
        for c, p, m in lst_add:
            self.nodes[p].add_child_id(c,m)
            self.nodes[p].add_parents(p, m)



    
        
        
    

    

            
                



        
    



        

        


        
    
        
                
