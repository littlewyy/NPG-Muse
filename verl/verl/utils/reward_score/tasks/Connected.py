from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
import networkx as nx
from collections import Counter
import random
import pickle
import re
import signal
import functools
import json
import numpy as np
import walker
import math
import time
import pandas as pd
from tqdm import tqdm

from .base import *
class Connected_Task(NPTask):
    def __init__(self, data_loc='dataset', task_name='Connected'):
        super(Connected_Task, self).__init__(data_loc, task_name)
        self.examples = []
    
    def check_solution(self, problem_id=None, response=None, graph=None, problem_text=None):
        
        
        
        
        
        if graph:
            
            
            
            
            g = graph
        else:
            if problem_id not in self.problem_set:
                
                return -1
            g = self.problem_set[problem_id]['graph']
            

        pattern = re.compile(r'\[(.*?)\]')
        p = pattern.findall(response)
        # print("Found patterns:", p)
        
        if p:
            matches = p[-1]
            matches = matches.split(",")
            name_list = [name.strip() for name in matches]
            # print("Name list:", name_list)
            
            node_list = []
            for name in name_list:
                
                node = find_node_by_name(g, name)
                
                if node is None:
                    continue
                node_list.append(node)
            
            
            if not node_list:
                
                return -2
                
            
            components = list(nx.connected_components(g))
            node_to_component = {}
            for i, comp in enumerate(components):
                for node in comp:
                    node_to_component[node] = i
            
            found_components = set()
            for node in node_list:
                if node in node_to_component:
                    found_components.add(node_to_component[node])
            
            if len(found_components) == len(node_list):
                
                return len(found_components)
            else:
                
                return -2
        
        return -1
        
    
    
    
    
    
    
    
    
    
    

    def is_feasible(self, g, node_list): 
        
        if not node_list:
            return False
            
        
        components = list(nx.connected_components(g))
        
        
        node_to_component = {}
        for i, comp in enumerate(components):
            for node in comp:
                node_to_component[node] = i
                
        
        found_components = set()
        for node in node_list:
            if node in node_to_component:
                comp_idx = node_to_component[node]
                if comp_idx in found_components:
                    
                    return False
                found_components.add(comp_idx)
        
        
        return True
        
    def generate_problem(self, graph):
        description = []
        description.append("You are required to identify all connected components in the given social network and output one representative node from each component.")
        description.append("Within a connected component, any node can be reached from any other node through the edges in the graph. Different connected components are isolated from each other.")
        description.append('\n**Problem to Solve**\n')
        description.append("- Names in the network: " + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()]))
        description.append('- Fiendship connections: ' + ", ".join([f"{graph.nodes[u]['name']} to {graph.nodes[v]['name']}" for u,v,data in graph.edges(data=True)]))
        description.append("Identify all connected components in this network. Note that for each connected component, you should only output one of its nodes.")
        description.append('Present your answer in the following format: [UserA, UserB, UserC, UserD, ...]')
        return '\n'.join(description)
    
    def generate_example(self, graph, path):
        example = []
        example.append('- Names in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()])+'.')
        relations = ", ".join([f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u,v,data in graph.edges(data=True)])
        example.append(f"- Friendship connections: {relations}.")
        answer = ", ".join([graph.nodes[node]['name'] for node in path])
        example.append(f"The answer including one representative element from each connected component in the given social network: [{answer}]")
        return '\n'.join(example)
    
    def generate_dataset(self, count=500):             
        G = pickle.load(open('./source/social_network_union.pkl', 'rb'))
        all_walks = walker.random_walks(G, n_walks=1, walk_len = 1000, start_nodes=range(G.number_of_nodes()), alpha=0.2)
        for difficulty in ['easy', 'hard']:
            self.problem_set = []
            min_nodes, max_nodes = (4, 14) if difficulty == 'easy' else (15, 30)
            
            while len(self.problem_set) < count:
                node_size = sample_node_size(min_nodes, max_nodes)
                c = Counter()
                for e in range(random.randint(1, 1+node_size//5)):
                    c.update(all_walks[random.randint(0, G.number_of_nodes()-1)])
  
                node_list = [k for k, v in c.most_common(node_size)]
                if len(node_list) < node_size:
                    continue       
                H = nx.induced_subgraph(G, node_list).copy()
                
                exact_answer, path = self.exact_solver(H)  
                if len(self.examples) < 100:
                    self.examples.append(self.generate_example(H, path))
                    continue
                    
                self.problem_set.append({
                    'id' : len(self.problem_set),
                    'problem_text' : self.generate_problem(H),
                    'graph': H,
                    'path': path,
                    'exact_answer': exact_answer,
                })
            self.save_dataset(difficulty)

    @staticmethod
    def exact_solver(graph): 
        connected_num = nx.number_connected_components(graph)
        components = nx.connected_components(graph)
        representative_nodes = [list(comp)[0] for comp in components]
        return connected_num, representative_nodes