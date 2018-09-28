from treelib import Node, Tree
import pandas as pd
import random

CHANCE_OF_BEING_TERMINAL = 70
RANDOM_TERMINALS_SIZE = 50
functions = []
terminals = []


class NodeData(object):
	def __init__(self, node_type, node_value):
		self.node_type = node_type
		self.node_value = str(node_value)


def generate_tree(tree: Tree, current_height: int, max_height: int, parent_node_name: str = ""):
	if current_height == max_height:
		return

	left_node_name = parent_node_name + "L"
	right_node_name = parent_node_name + "R"

	if parent_node_name == "":
		parent_node_name = "Root"

	# left child
	left_node_data = set_node_data()
	tree.create_node(left_node_name, left_node_name, parent=parent_node_name, data=left_node_data)

	if left_node_data.node_type == "function":
		generate_tree(tree, current_height+1, max_height, left_node_name)

	# right child
	right_node_data = set_node_data()
	tree.create_node(right_node_name, right_node_name, parent=parent_node_name, data=right_node_data)

	if right_node_data.node_type == "function":
		generate_tree(tree, current_height + 1, max_height, right_node_name)


def set_node_data() -> NodeData:
	data = {"type": None, "value": None}
	data["type"] = "function" if random.randrange(100) > CHANCE_OF_BEING_TERMINAL else "terminal"
	data["value"] = random.choice(functions) if data["type"] == "function" else random.choice(terminals)
	return NodeData(data["type"], data["value"])


def df_to_list(df: pd.DataFrame) -> list:
	data = [df[y].tolist() for y in range(len(df.columns))]
	data = [item for x in data for item in x]
	return data


def resolve_tree(tree: Tree, node_name: str):
	node = tree.get_node(node_name)
	children = node.fpointer
	if not children:
		return node.data.node_value

	node_name = "" if node.tag == "Root" else node.tag

	left_node_name = node_name + "L"
	right_node_name = node_name + "R"

	left_value = resolve_tree(tree, left_node_name)
	if right_node_name in children:
		right_value = resolve_tree(tree, right_node_name)
		full_string = left_value + " " + node.data.node_value + " " + right_value
		return str(eval(full_string))
	else:
		return left_value


def start():
	data = pd.read_csv('datasets/synth1/synth1-train.csv', header=None)
	x_set = data.drop(data.columns[len(data.columns)-1], axis=1).copy()
	y_set = data[len(data.columns)-1].copy()

	global functions
	functions = ['+', '-', '*', '/']

	global terminals
	terminals = df_to_list(x_set)

	seed = 10
	random.seed(seed)
	random_terminals = [random.uniform(-5, 5) for _ in range(RANDOM_TERMINALS_SIZE)]
	terminals = [*terminals, *random_terminals]

	# heigth = random.randint(2, 7)
	max_height = 7

	tree = Tree()
	root_node_data = set_node_data()
	tree.create_node("Root", "Root", data=root_node_data)
	if root_node_data.node_type == "function":
		generate_tree(tree, 0, max_height)

	tree.show()

	resolve_tree(tree, "Root")
	print(tree.children("L"))