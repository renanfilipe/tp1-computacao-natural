from treelib import Tree
import pandas as pd
import random

CHANCE_OF_BEING_TERMINAL = 0.5
RANDOM_TERMINALS_SIZE = 50


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


def df_to_list(df: pd.DataFrame):
	data = [df[y].tolist() for y in range(len(df.columns))]
	data = [item for x in data for item in x]
	return data


def start():
	data = pd.read_csv('datasets/synth1/synth1-train.csv', header=None)
	x_set = data.drop(data.columns[len(data.columns)-1], axis=1).copy()
	y_set = data[len(data.columns)-1].copy()

	functions = ['+', '-', '*', '/']
	terminals = df_to_list(x_set)

	seed = 1
	random.seed(seed)
	random_terminals = [random.uniform(-5, 5) for _ in range(RANDOM_TERMINALS_SIZE)]
	terminals = [*terminals, *random_terminals]

	# heigth = random.randint(2, 7)
	heigth = 5
