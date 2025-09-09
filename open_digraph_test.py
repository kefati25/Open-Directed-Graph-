import sys
import os
root = os.path.normpath(os.path.join(__file__, ))
sys.path.append(root) # allows us to fetch files from the project root
import unittest
from modules.open_digraph import *


class InitTest(unittest.TestCase):
   def test_init_node(self):
      n0 = node()
      node._init_(n0, identity=0, label='i', parents={}, children={1: 1})
      self.assertEqual(n0.id, 0)
      self.assertEqual(n0.label, 'i')
      self.assertEqual(n0.parents, {})
      self.assertEqual(n0.children, {1:1})
      self.assertIsInstance(n0, node)

   def test_init_digraph(self):
      n0 = node()
      n1 = node()
      n2 = node()
      node._init_(n0, 0, 'A', {}, {1: 1})
      node._init_(n1, 1, 'B', {0: 1}, {2: 1})
      node._init_(n2, 2, 'C', {1: 1}, {})
      graph = open_digraph()
      open_digraph._init_(graph, inputs=[0], outputs=[2], nodes=[n0, n1, n2])
      print(graph)
   
   def test_copy_node(self):
      n0 = node()
      node._init_(n0,0, 'i', {}, {1:1})
      n1 = node()
      n1 = node.copy(n0)
      self.assertIsNot(n0.copy( ),n0)
      self.assertEqual(n0.id, n1.id)
      self.assertEqual(n0.label, n1.label)
      self.assertEqual(n0.parents, n1.parents)
      self.assertEqual(n0.children, n1.children)

   def test_get_node(self):
      n0 = node()
      node._init_(n0,0, 'A', {}, {1: 1})
      self.assertEqual(n0.id, n0.get_id())
      self.assertEqual(n0.label, n0.get_label())
      self.assertEqual(n0.children, n0.get_children())
      self.assertEqual(n0.parents, n0.get_parents())

   def test_copy_digraph(self):
      n0 = node()
      n1 = node()
      n2 = node()
      node._init_(n0, 0, 'A', {}, {1: 1})
      node._init_(n1, 1, 'B', {0: 1}, {2: 1})
      node._init_(n2, 2, 'C', {1: 1}, {})
      graph = open_digraph()
      open_digraph._init_(graph, inputs=[0], outputs=[2], nodes=[n0, n1, n2])
      graph2 = graph.copy()
      self.assertIsNot(graph2, graph)
      self.assertEqual(graph2.inputs, graph.inputs)
      self.assertEqual(graph2.outputs, graph.outputs)
      self.assertEqual(graph2.nodes, graph.nodes)

   def test_empty(self):
      graph = open_digraph()
      open_digraph._init_(graph, inputs=[], outputs=[], nodes=[ ])
      graph2 = open_digraph.empty()
      self.assertEqual(graph2.inputs, graph.inputs)
      self.assertEqual(graph2.outputs, graph.outputs)
      self.assertEqual(graph2.nodes, graph.nodes)

   def test_get_digraph(self):
      n0 = node()
      n1 = node()
      n2 = node()
      node._init_(n0, 0, 'A', {}, {1: 1})
      node._init_(n1, 1, 'B', {0: 1}, {2: 1})
      node._init_(n2, 2, 'C', {1: 1}, {})
      graph = open_digraph()
      open_digraph._init_(graph, inputs=[0], outputs=[2], nodes=[n0, n1, n2])
      self.assertEqual(graph.get_input_ids(), graph.inputs)
      self.assertEqual(graph.get_output_ids(), graph.outputs)
      self.assertDictEqual(graph.get_id_node_map(), graph.nodes)
      self.assertEqual(graph.get_nodes_by_id(n0.id), n0)
      self.assertEqual(graph.get_nodes(), [n0, n1, n2])

   def test_is_wall_graph(self):
      n1 = node()
      n2 = node()
      n3 = node()
      n4 = node()
      node._init_(n1, identity=0, label='n1', parents={}, children={1: 1})
      node._init_(n2, identity=1, label='n2', parents={0: 1}, children={2: 1})
      node._init_(n3, identity=2, label='n3', parents={1: 1}, children={})
      node._init_(n4, identity=3, label='n4', parents={}, children={1:3})

      graph = open_digraph()
      open_digraph._init_(graph, inputs=[0], outputs=[2], nodes=[n1, n2, n3])
      ## les inputs sont males fait 
      graph1 = open_digraph()
      open_digraph._init_(graph1, inputs=[1], outputs=[2], nodes=[n1, n2, n3])

      graph2 = open_digraph()
      open_digraph._init_(graph2, inputs=[3], outputs=[2], nodes=[n1, n2, n3, n4])
      ## les outputs sont males fait 
      graph3 = open_digraph()
      open_digraph._init_(graph3, inputs=[0], outputs=[0], nodes=[n1, n2, n3])
      ## il y a un id dans l'output qui n'est pas dans les neud
      graph4 = open_digraph()
      open_digraph._init_(graph4, inputs=[0], outputs=[2], nodes=[n2, n3])

      self.assertTrue(graph.is_well_formed())
      self.assertFalse(graph1.is_well_formed())
      self.assertFalse(graph2.is_well_formed())
      self.assertFalse(graph3.is_well_formed())
      self.assertFalse(graph4.is_well_formed())
   
   def test_add_remove_edge(self):
      n1 = node()
      n2 = node()
      n3 = node()
      node._init_(n1, identity=0, label='n1', parents={}, children={})
      node._init_(n2, identity=1, label='n2', parents={0: 1}, children={2: 1})
      node._init_(n3, identity=2, label='n3', parents={1: 1}, children={})
      graph = open_digraph()
      open_digraph._init_(graph, inputs=[0], outputs=[2], nodes=[n1, n2, n3])

      graph.add_edge(0, 2)
      self.assertTrue(graph.is_well_formed())
      graph.remove_edge(1, 2)
      self.assertTrue(graph.is_well_formed())

   def test_add_remove_node(self):
      n1 = node()
      n2 = node()
      n3 = node()
      node._init_(n1, identity=0, label='n1', parents={}, children={1: 1})
      node._init_(n2, identity=1, label='n2', parents={0: 1}, children={2: 1})
      node._init_(n3, identity=2, label='n3', parents={1: 1}, children={})
      graph = open_digraph()
      open_digraph._init_(graph, inputs=[0], outputs=[2], nodes=[n1, n2, n3])

      graph.add_node(label='new_node', parents={}, children={})
      self.assertTrue(graph.is_well_formed())

      graph.remove_node_by_id(0)
      self.assertTrue(graph.is_well_formed())

   def test_add_inpout_output(self):
      n1 = node()
      n2 = node()
      n3 = node()
      node._init_(n1, identity=0, label='n1', parents={}, children={1: 1})
      node._init_(n2, identity=1, label='n2', parents={0: 1}, children={2: 1})
      node._init_(n3, identity=2, label='n3', parents={1: 1}, children={})
      graph = open_digraph()
      open_digraph._init_(graph, inputs=[0], outputs=[2], nodes=[n1, n2, n3])

      graph.add_input_node(2)
      self.assertTrue(graph.is_well_formed())

   def test_random_int_matrix(self):
        n = 5
        bound = 10
        result = open_digraph.random_int_matrix(n, bound)
        self.assertEqual(len(result), n)
        print('[')
        for row in result:
            self.assertEqual(len(row), n)
            print('    [', end='')
            for num in row:
               print(num, end=', ')
            print(']')
        print(']')

   def test_random_symmetric_int_matrix(self):
        n = 5
        bound = 10
        result = open_digraph.random_symmetric_int_matrix(n, bound)
        self.assertEqual(len(result), n)
        print('[')
        for row in result:
            self.assertEqual(len(row), n)
            print('    [', end='')
            for num in row:
               print(num, end=', ')
            print(']')
        print(']')

   def test_random_oriented_int_matrix(self):
        n = 5
        bound = 10
        result = open_digraph.random_oriented_int_matrix(n, bound)
        self.assertEqual(len(result), n)
        print('[')
        for row in result:
            self.assertEqual(len(row), n)
            print('    [', end='')
            for num in row:
               print(num, end=', ')
            print(']')
        print(']')

   def test_random_dag_int_matrix(self):
        n = 5
        bound = 10
        result = open_digraph.random_dag_int_matrix(n, bound)
        self.assertEqual(len(result), n)
        print('[')
        for row in result:
            self.assertEqual(len(row), n)
            print('    [', end='')
            for num in row:
               print(num, end=', ')
            print(']')
        print(']')

   def test_graph_from_adjacency_matrix(self):
        matrix = [
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 2, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ]
        g = open_digraph.graph_from_adjacency_matrix(matrix)
        self.assertTrue(g.is_well_formed())
        node_lst = g.get_nodes()
        self.assertDictEqual(node_lst[0].get_children(), {1:1, 2:1})
        self.assertDictEqual(node_lst[1].get_children(), {3:1, 4:2})
        self.assertDictEqual(node_lst[2].get_children(), {3:2})
        self.assertDictEqual(node_lst[3].get_children(), {0:1, 4:1})
        self.assertDictEqual(node_lst[4].get_children(), {})


        self.assertDictEqual(node_lst[0].get_parents(), {3:1})
        self.assertDictEqual(node_lst[1].get_parents(), {0:1})
        self.assertDictEqual(node_lst[2].get_parents(), {0:1})
        self.assertDictEqual(node_lst[3].get_parents(), {1:1, 2:2})
        self.assertDictEqual(node_lst[4].get_parents(), {1:2, 3:1})

        self.assertListEqual(g.inputs, [])
        self.assertListEqual(g.outputs, [])
      
   def test_random(self):
      g = open_digraph.random(5, 3, 1, 1)
      self.assertTrue(g.is_well_formed())
      out = g.get_output_ids()
      inp = g.get_input_ids()
      node = g.get_nodes()
      self.assertTrue(len(out), 1)
      self.assertTrue(len(inp), 1)
      self.assertTrue(len(node), 5)
      
   def test_adjacency_matrix(self):
      print(" fontion                        \n")
      matrix = [
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 2, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ]
      g = open_digraph.graph_from_adjacency_matrix(matrix)
      matrix2 = g.adjacency_matrix()
      self.assertTrue(len(matrix), len(matrix2))
      for i in range(len(matrix)):
         self.assertTrue(len(matrix[i]), len(matrix2[i]))
         self.assertListEqual(matrix[i], matrix2[i])
         

   def test_dictionnary_id_graph(self):
      n1 = node()
      n2 = node()
      n3 = node()
      node._init_(n1, identity=0, label='n1', parents={}, children={1: 1})
      node._init_(n2, identity=1, label='n2', parents={0: 1}, children={2: 1})
      node._init_(n3, identity=2, label='n3', parents={1: 1}, children={})
      graph = open_digraph()
      open_digraph._init_(graph, inputs=[0], outputs=[2], nodes=[n1, n2, n3])
      indices = graph.dictionnary_id_graph()
      print (indices)

   def test_save_as_dot_file(self):
      n1 = node()
      n2 = node()
      n3 = node()
      node._init_(n1, identity=0, label='n1', parents={}, children={1: 1})
      node._init_(n2, identity=1, label='n2', parents={0: 1}, children={2: 1})
      node._init_(n3, identity=2, label='n3', parents={1: 1}, children={})
      graph = open_digraph()
      open_digraph._init_(graph, inputs=[0], outputs=[2], nodes=[n1, n2, n3])
      
      path = "modules"
      
      graph_dot = graph.save_as_dot_file(path, True)


    
if __name__ == '__main__': # the following code is called only when
 unittest.main() 
 # precisely this file is run