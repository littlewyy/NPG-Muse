import json
import math
import re
from typing import Dict, Any
import networkx as nx  

from .tasks.MCS import MCS_Task
from .tasks.TSP import TSP_Task
from .tasks.GED import GED_Task
from .tasks.MCP import MCP_Task
from .tasks.MVC import MVC_Task
from .tasks.MIS import MIS_Task
from .tasks.Connected import Connected_Task
from .tasks.Diameter import Diameter_Task
from .tasks.Neighbor import Neighbor_Task
from .tasks.Distance import Distance_Task

def build_networkx_graph(graph_data):
    """转换JSON图数据为NetworkX图。"""
    if isinstance(graph_data, str):
        try:
            graph_data = json.loads(graph_data)  
        except:
            
            return None

    try:
        if not isinstance(graph_data, list):
            G = nx.Graph()
            
            
            for node in graph_data.get("nodes", []):
                attrs = {}
                
                if "label" in node:
                    attrs["label"] = node["label"]
                if "name" in node:
                    attrs["name"] = node["name"]
                G.add_node(node["id"], **attrs)
            
            
            for link in graph_data.get("links", []):
                attrs = {}
                
                if "label" in link:
                    attrs["label"] = link["label"]
                if "relation" in link:
                    attrs["relation"] = link["relation"]
                G.add_edge(link["source"], link["target"], **attrs)
            
            
            return G
        else:  
            return [build_networkx_graph(g) for g in graph_data]        
    except Exception as e:
        
        return None
    

def graph_reward_complicated(completions, extra_info, task_name):
    """Reward function for connected components task."""
    content = completions
    example = extra_info 
    
    reward = 0.0
    print("---GRAPH REWARD DEBUG---")
    task_instance = globals()[f"{task_name}_Task"]()

    graph_json = example.get("graph", "{}")
    
    
    graph = None
    
    try:
        graph = build_networkx_graph(graph_json) 
    except Exception as e:
        
        graph = None
    
    
    problem_text = example.get("problem_text", "")

    
    small_better = ["Distance", "MVC", "GED", "TSP"]
    big_better = ["Neighbor", "Connected", "Diameter", "MCP", "MIS", "MCS"]
    
    
    res = 0.0
    if graph is None:
        reward = 0.0
        print("graph is None!")
        print("Reward:", reward)
        print("---END GRAPH REWARD---\n")
        return reward 
    else:
        try:
            res = task_instance.check_solution(
                problem_id=None,
                response=content,
                graph=graph,
                problem_text=problem_text
            )
            print("Current result:", res)
        except Exception as e:
            print(f"Error in check_solution: {e}")
            reward = 0.0
            print("Reward:", reward)
            print("---END GRAPH REWARD---\n")
            return reward 
  
    expected = example.get("exact_answer", None)
    if expected is not None and str(expected).strip() != "":
        expected = int(float(expected))
        print(f"Expected answer: {expected}")
   
        if res == expected:
            reward = 2.0
        elif res > 0: 
            if task_name in small_better:
                if task_name != "TSP":
                    reward = (expected / res) * 0.5
                else:
                    reward = 0.5 * ((expected / res) ** 2) 
            else:
                reward = (res / expected) * 0.5
        else: 
            
            reward = -1.0 
        print("Reward:", reward)
        print("---END GRAPH REWARD---\n")
        return reward
    else:
        
        
        print(f"No exact answer, return res only: {res}")
        print("---END GRAPH REWARD---\n")
        return res

def graph_reward_binary(completions, extra_info, task_name):
    content = completions
    example = extra_info 
    
    reward = 0.0
    print("---GRAPH REWARD DEBUG---")
    task_instance = globals()[f"{task_name}_Task"]()

    graph_json = example.get("graph", "{}")
    
    
    graph = None
    
    try:
        graph = build_networkx_graph(graph_json) 
    except Exception as e:
        
        graph = None
    
    
    problem_text = example.get("problem_text", "")

    
    small_better = ["Distance", "MVC", "GED", "TSP"]
    big_better = ["Neighbor", "Connected", "Diameter", "MCP", "MIS", "MCS"]
    
    
    res = 0.0
    if graph is None:
        reward = 0.0
        print("graph is None!")
        print("Reward:", reward)
        print("---END GRAPH REWARD---\n")
        return reward 
    else:
        try:
            res = task_instance.check_solution(
                problem_id=None,
                response=content,
                graph=graph,
                problem_text=problem_text
            )
            print("Current result:", res)
        except Exception as e:
            print(f"Error in check_solution: {e}")
            reward = 0.0
            print("Reward:", reward)
            print("---END GRAPH REWARD---\n")
            return reward 
  
    expected = example.get("exact_answer", None)
    if expected is not None and str(expected).strip() != "":
        expected = int(float(expected))
        print(f"Expected answer: {expected}")
   
        if res == expected:
            reward = 1.0
        else:             
            reward = 0.0
        print("Reward:", reward)
        print("---END GRAPH REWARD---\n")
        return reward
    else:        
        print(f"No exact answer, return res only: {res}")
        print("---END GRAPH REWARD---\n")
        return res

def graph_reward_ratio_quality(completions, extra_info, task_name):
    """Reward function for connected components task."""
    content = completions
    example = extra_info 
    
    reward = 0.0
    print("---GRAPH REWARD DEBUG---")
    task_instance = globals()[f"{task_name}_Task"]()

    graph_json = example.get("graph", "{}")
    
    
    graph = None
    
    try:
        graph = build_networkx_graph(graph_json) 
    except Exception as e:
        
        graph = None
    
    
    problem_text = example.get("problem_text", "")

    
    small_better = ["Distance", "MVC", "GED", "TSP"]
    big_better = ["Neighbor", "Connected", "Diameter", "MCP", "MIS", "MCS"]
    
    
    res = 0.0
    if graph is None:
        reward = 0.0
        print("graph is None!")
        print("Reward:", reward)
        print("---END GRAPH REWARD---\n")
        return reward 
    else:
        try:
            res = task_instance.check_solution(
                problem_id=None,
                response=content,
                graph=graph,
                problem_text=problem_text
            )
            print("Current result:", res)
        except Exception as e:
            print(f"Error in check_solution: {e}")
            reward = 0.0
            print("Reward:", reward)
            print("---END GRAPH REWARD---\n")
            return reward 
  
    expected = example.get("exact_answer", None)
    if expected is not None and str(expected).strip() != "":
        expected = int(float(expected))
        print(f"Expected answer: {expected}")
        # 正确则为1.0；部分正确为0~0.5；不合法为0.0
        if res == expected:
            reward = 1.0
        elif res > 0: 
            if task_name in small_better:
                reward = (expected / res) * 0.5
            else:
                reward = (res / expected) * 0.5
        else: 
            reward = 0.0 
        print("Reward:", reward)
        print("---END GRAPH REWARD---\n")
        return reward
    else:
        print(f"No exact answer, return res only: {res}")
        print("---END GRAPH REWARD---\n")
        return res

def format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?$" 
    content = completions
    match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
    return 1.0 if match else 0.0

from string_repetition import StringRepetitionDetector

def detect_repeat(solution_str: str):
    detector = StringRepetitionDetector(
        min_length=20,     
        min_repeats=5      
    )
    result = detector.detect_string(solution_str, parallel=True)
    result = result.has_repetition 
    return result

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 extra_info: Dict[str, Any],
                 task_name: str,
                 reward_type: str,
                ) :
    print("\n" + "="*80)
    
    # print(f"\n[Problem Text]\n{extra_info.get('problem_text')}")
    # print(f"\n[Model Response]\n{solution_str}")
    print(f"\n[Ground Truth]\n{extra_info.get('exact_answer')}")

    if reward_type == "complicated":
        format_score = format_reward(solution_str) 
        format_correct = True if format_score == 1.0 else False

        if format_correct:
            format_score_scaled = 1
        else:
            format_score_scaled = 0

        answer_score = graph_reward_complicated(solution_str, extra_info, task_name)
        if answer_score == 2.0:
            answer_score_scaled = 1
        else:
            answer_score_scaled = 0

        repeat_appear = detect_repeat(solution_str)

        if repeat_appear:
            repeat_score = -1.0 
            repeat_score_scaled = 1.0 
        else:
            repeat_score = 0.0    
            repeat_score_scaled = 0.0
        
        total_score = format_score + answer_score + repeat_score
        
        print(f"  Format Reward: {format_score}")
        print(f"  Answer Reward: {answer_score}")
        print(f"  Repeat Reward: {repeat_score}")
        print(f"  Total Reward: {total_score}")
        
        print("="*80 + "\n")
        
        output = {
            "score": total_score,
            "answer_accuracy": answer_score_scaled, 
            "format_accuracy": format_score_scaled, 
            "repeat_proportion": repeat_score_scaled
        }

        return output

    elif reward_type == "binary":
        answer_score = graph_reward_binary(solution_str, extra_info, task_name)
        total_score = answer_score
        print(f"  Answer Reward: {answer_score}")
        print(f"  Total Reward: {total_score}")
        print("="*80 + "\n")
        
        output = {
            "score": total_score,
            "answer_accuracy": answer_score, 
        }
        return output
    
    elif reward_type == "binary_format":        
        answer_score = graph_reward_binary(solution_str, extra_info, task_name) # answer: 0~1
        if answer_score == 1.0:
            answer_score_scaled = 1
        else:
            answer_score_scaled = 0
        
        format_score = format_reward(solution_str) * 0.5 # format: 0~0.5
        format_correct = True if format_score > 0 else False

        if format_correct:
            format_score_scaled = 1
        else:
            format_score_scaled = 0

        total_score = answer_score + format_score
        print(f"  Answer Reward: {answer_score}")
        print(f"  Format Reward: {format_score}")
        print(f"  Total Reward: {total_score}")
        print("="*80 + "\n")
        
        output = {
            "score": total_score,
            "answer_accuracy": answer_score_scaled, 
            "format_accuracy": format_score_scaled, 
        }
        return output
    
    elif reward_type == "binary_format_repeat":
        answer_score = graph_reward_binary(solution_str, extra_info, task_name) # answer: 0~1
        if answer_score == 1.0:
            answer_score_scaled = 1
        else:
            answer_score_scaled = 0
        
        format_score = format_reward(solution_str) * 0.5 # format: 0~0.5
        format_correct = True if format_score > 0 else False

        if format_correct:
            format_score_scaled = 1
        else:
            format_score_scaled = 0
        
        repeat_appear = detect_repeat(solution_str) # repeat: -0.5/0
        if repeat_appear:
            repeat_score = -0.5 
            repeat_score_scaled = 1.0 
        else:
            repeat_score = 0.0    
            repeat_score_scaled = 0.0
        
        total_score = answer_score + format_score + repeat_score
        print(f"  Answer Reward: {answer_score}")
        print(f"  Format Reward: {format_score}")        
        print(f"  Repeat Reward: {repeat_score}")
        print(f"  Total Reward: {total_score}")
        print("="*80 + "\n")
        
        output = {
            "score": total_score,
            "answer_accuracy": answer_score_scaled, 
            "format_accuracy": format_score_scaled, 
            "repeat_proportion": repeat_score_scaled
        }
        return output
    
    elif reward_type == "ratio_quality_format_repeat":

        answer_score = graph_reward_ratio_quality(solution_str, extra_info, task_name) # answer: 0~1
        if answer_score == 1.0:
            answer_score_scaled = 1
        else:
            answer_score_scaled = 0
        
        format_score = format_reward(solution_str) * 0.5 # format: 0~0.5
        format_correct = True if format_score > 0 else False

        if format_correct:
            format_score_scaled = 1
        else:
            format_score_scaled = 0
        
        repeat_appear = detect_repeat(solution_str)
        if repeat_appear:
            repeat_score = -0.5 
            repeat_score_scaled = 1.0 
        else:
            repeat_score = 0.0    
            repeat_score_scaled = 0.0
        
        total_score = format_score + answer_score + repeat_score
        print(f"  Format Reward: {format_score}")
        print(f"  Answer Reward: {answer_score}")
        print(f"  Repeat Reward: {repeat_score}")
        print(f"  Total Reward: {total_score}")
        print("="*80 + "\n")
        
        output = {
            "score": total_score,
            "format_accuracy": format_score_scaled, 
            "answer_accuracy": answer_score_scaled, 
            "repeat_proportion": repeat_score_scaled
        }
        return output