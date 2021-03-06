from treelib import Node, Tree
import pandas as pd
import random
import math
import copy

NUMBER_OF_GENERATIONS = 100
CHANCE_OF_BEING_TERMINAL = 70
CHANCE_OF_CROSSOVER = 700
CHANCE_OF_MUTATION = 5
CHANCE_OF_TOURNAMENT = 50
NUMBER_OF_CONSTANTS = 20
NUMBER_OF_INDIVIDUALS = 200
MAX_HEIGHT = 7
ELITISM = True
TOURNAMENT_SIZE = 2

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
	left_node_data = set_node_data(current_height)
	tree.create_node(left_node_name, left_node_name, parent=parent_node_name, data=left_node_data)

	if left_node_data.node_type == "function":
		generate_tree(tree, current_height+1, max_height, left_node_name)

	# right child
	right_node_data = set_node_data(current_height)
	tree.create_node(right_node_name, right_node_name, parent=parent_node_name, data=right_node_data)

	if right_node_data.node_type == "function":
		generate_tree(tree, current_height + 1, max_height, right_node_name)


def set_node_data(current_height: int) -> NodeData:
	data = {"type": None, "value": None}
	data["type"] = "function" if random.randrange(100) > CHANCE_OF_BEING_TERMINAL and current_height < MAX_HEIGHT - 1 else "terminal"
	data["value"] = random.choice(functions) if data["type"] == "function" else random.choice(terminals)
	return NodeData(data["type"], data["value"])


def df_to_list(df: pd.DataFrame) -> list:
	data = [df[y].tolist() for y in range(len(df.columns))]
	data = [item for x in data for item in x]
	return data


def resolve_tree(tree: Tree, node_name: str) -> str:
	node = tree.get_node(node_name)
	children = node.fpointer
	if not children:
		return node.data.node_value

	node_name = "" if node.tag == "Root" else node.tag

	left_node_name = node_name + "L"
	right_node_name = node_name + "R"

	left_value = resolve_tree(tree, left_node_name)
	right_value = resolve_tree(tree, right_node_name)
	try:
		return str(eval(left_value + " " + node.data.node_value + " " + right_value))
	except ZeroDivisionError:
		print("zero division error on tree")
		return str((eval(left_value + " " + node.data.node_value + " 1")))


def estimate_fitness(tree_value: float, list_of_outputs: pd.Series) -> float:
	numerator = 0
	denominator = 0
	mean = list_of_outputs.mean()
	for output in list_of_outputs:
		numerator += math.pow(output - tree_value, 2)
		denominator += math.pow(output - mean, 2)
	try:
		return math.sqrt(numerator/denominator)
	except ZeroDivisionError:
		print("zero division error on fitness")
		return 0


def generate_individual() -> dict:
	tree = Tree()
	tree.create_node("Root", "Root")
	while True:
		root_node_data = set_node_data(0)
		tree.get_node("Root").data = root_node_data
		if root_node_data.node_type == "function":
			generate_tree(tree, 0, MAX_HEIGHT)
			break
	return {
		"tree": tree,
		"value": float(resolve_tree(tree, "Root")),
		"fitness": None
	}


def operator_crossover(tree_a_obj: dict, tree_b_obj: dict, y_set: pd.Series) -> tuple:
	tree_a = tree_a_obj["tree"]
	tree_b = tree_b_obj["tree"]
	tree_a_nodes = list(tree_a.nodes.keys())
	tree_b_nodes = list(tree_b.nodes.keys())
	possible_nodes = list(set(tree_a_nodes).intersection(tree_b_nodes))
	crossover_node_tag = random.choice(possible_nodes)
	while crossover_node_tag == "Root":
		crossover_node_tag = random.choice(possible_nodes)

	child_ab_obj = {"tree": None, "value": None, "fitness": None}
	child_ab_obj["tree"] = copy.deepcopy(tree_a)
	child_ab_obj["tree"].remove_node(crossover_node_tag)
	child_ab_obj["tree"].paste(tree_a.get_node(crossover_node_tag).bpointer, tree_b.subtree(crossover_node_tag))
	child_ab_obj["value"] = float(resolve_tree(child_ab_obj["tree"], "Root"))

	child_ba_obj = {"tree": None, "value": None, "fitness": None}
	child_ba_obj["tree"] = copy.deepcopy(tree_b)
	child_ba_obj["tree"].remove_node(crossover_node_tag)
	child_ba_obj["tree"].paste(tree_b.get_node(crossover_node_tag).bpointer, tree_a.subtree(crossover_node_tag))
	child_ba_obj["value"] = float(resolve_tree(child_ba_obj["tree"], "Root"))

	if ELITISM:
		child_ab_obj["fitness"] = estimate_fitness(child_ab_obj["value"], y_set)
		child_ba_obj["fitness"] = estimate_fitness(child_ba_obj["value"], y_set)
		result_list = sorted([tree_a_obj, tree_b_obj, child_ab_obj, child_ba_obj], key=lambda k: k['fitness'])
		return result_list[0], result_list[1]

	return child_ab_obj, child_ba_obj


def operator_mutation(tree_obj: dict, y_set: pd.Series):
	tree = tree_obj["tree"]
	tree_nodes = list(tree.nodes.keys())
	node_to_be_mutated = random.choice(tree_nodes)
	node = tree.get_node(node_to_be_mutated)
	list_to_be_used = functions if node.data.node_type == "function" else terminals
	new_value = random.choice(list_to_be_used)
	while new_value == node.data.node_value:
		new_value = random.choice(list_to_be_used)

	child_obj = {"tree": None, "value": None, "fitness": None}
	child_obj["tree"] = copy.deepcopy(tree)
	child_obj["tree"].get_node(node_to_be_mutated).data.node_value = str(new_value)
	child_obj["value"] = float(resolve_tree(child_obj["tree"], "Root"))

	if ELITISM:
		child_obj["fitness"] = estimate_fitness(child_obj["value"], y_set)
		return child_obj if child_obj["fitness"] <= tree_obj["fitness"] else tree_obj

	return child_obj


def tournament(population: list):
	participants = random.sample(population, k=TOURNAMENT_SIZE)
	participants = sorted(participants, key=lambda k: k['fitness'])
	return participants[0]


def read_data(file_path: str) -> tuple:
	data = pd.read_csv(file_path, header=None)
	x_set = data.drop(data.columns[len(data.columns) - 1], axis=1).copy()
	y_set = data[len(data.columns) - 1].copy()
	return x_set, y_set


def remove_equals(population: list, old_population) -> list:
	old_population = sorted(old_population, key=lambda k: k['fitness'])
	new_population = []
	for i in range(1, len(population)):
		if population[i]["fitness"] != population[i-1]["fitness"]:
			new_population.append(population[i])

	index = len(old_population) - 1
	while len(new_population) <= len(population):
		new_population.append(old_population[index])
		index -= 1
	return new_population


def start():
	# read data from file
	x_set, y_set = read_data('datasets/synth1/synth1-train.csv')

	# define set of functions and terminals
	global functions
	global terminals
	functions = ['+', '-', '*', '/']
	terminals = df_to_list(x_set)
	seed = 10
	random.seed(seed)
	random_constants = [random.uniform(-1, 1) for _ in range(NUMBER_OF_CONSTANTS)]
	terminals = [*terminals, *random_constants]

	generations = []
	# generate starting population
	population = []
	for x in range(NUMBER_OF_INDIVIDUALS):
		population.append(generate_individual())

	# evaluate the fitness of each individual
	for individual in population:
		individual["fitness"] = estimate_fitness(individual["value"], y_set)

	for i in range(NUMBER_OF_GENERATIONS):
		# selection
		new_population = []
		# tournament selection
		# new_population.append(tournament(population))

		# applying operators
		did_crossover = False
		for i in range(0, NUMBER_OF_INDIVIDUALS):
			if did_crossover:
				did_crossover = False
				continue
			if random.randrange(1000) < CHANCE_OF_CROSSOVER:
				individual_a, individual_b = operator_crossover(population[i], population[(i+1)%NUMBER_OF_INDIVIDUALS], y_set)
				new_population.extend((individual_a, individual_b))
				did_crossover = True
			elif random.randrange(1000) < CHANCE_OF_MUTATION:
				new_population.append(operator_mutation(population[i], y_set))
			else:
				new_population.append(population[i])

		new_population = sorted(new_population, key=lambda k: k['fitness'])
		# new_population = remove_equals(new_population, population)
		generations.append(new_population)
		population = copy.deepcopy(new_population)
		random.shuffle(population)
	print()
