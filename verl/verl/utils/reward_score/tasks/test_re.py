import re

def extract_nodes(text):
    
    
    pattern = r"Please determine the shortest path between\s+(.*?)\s+and\s+(.*?)\s+in this network."
    
    match = re.search(pattern, text)
    
    
    if match:
        node1 = match.group(1)
        node2 = match.group(2)
        return node1, node2
    else:
        return None, None



test_text = "Please determine the shortest path between Texas and United States in this network."
node1, node2 = extract_nodes(test_text)

