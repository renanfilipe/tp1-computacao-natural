from treelib import Node, Tree
import pandas as pd
import numpy as np
import random

CHANCE_OF_BEING_TERMINAL = 0.5

def generate_tree(tree: Tree, current_height: int, max_height: int, parent_node_name: str):
	if current_height == max_height:
		return

	left_node_name = parent_node_name + "L"
	right_node_name = parent_node_name + "R"

	if parent_node_name == "":
		parent_node_name = "Root"

	# left child
	tree.create_node(left_node_name, left_node_name, parent=parent_node_name)
	generate_tree(tree, current_height+1, max_height, left_node_name)

	# right child
	tree.create_node(right_node_name, right_node_name, parent=parent_node_name)
	generate_tree(tree, current_height + 1, max_height, right_node_name)

def start():
	data = pd.read_csv('datasets/synth1/synth1-train.csv', header=None)
	functions = ['+', '-', '*', '/']
	terminals = [-1.235928614240245, -1.3641055948703154 ]

	seed = "bunitao, tao, tao ,tao, tao"
	random.seed(seed)
	# heigth = random.randint(2, 7)
	heigth = 5
	random_terminals_size = pow(2, heigth) - 2
	random_terminals = [random.uniform(0.0, 10.0) for _ in range(random_terminals_size)]
	terminals = [*terminals, *random_terminals]

	# tree = Tree()
	# tree.create_node("Root", "Root")  # root node
	# generate_tree(tree, 0, 5, "")
	# tree.show()
	# print(tree.depth())
