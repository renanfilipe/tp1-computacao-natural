from treelib import Node, Tree
import pandas as pd
from .not_random import NotRandom
import math
import copy
import matplotlib.pyplot as plt

CHANCE_TERMINAL = 0.7
CHANCE_CONSTANT = 0.15
CHANCE_FUNCTION = 0.15

CHANCE_CROSSOVER = 700
CHANCE_MUTATION = 5
CONSTANTS = 50

GENERATIONS = 30
INDIVIDUALS = 50
MAX_HEIGHT = 7
ELITISM = True
TOURNAMENT = False
TOURNAMENT_SIZE = 10

random = NotRandom()


class NodeData(object):
	def __init__(self, node_type, node_value):
		self.node_type = node_type
		self.node_value = str(node_value)
		self.terminal_value = str(node_value)


def read_data(file_path: str) -> tuple:
	data = pd.read_csv(file_path, header=None)
	x_set = data.drop(data.columns[len(data.columns) - 1], axis=1).copy()
	y_set = data[len(data.columns) - 1].copy()
	return list(x_set.values), y_set


def generate_individual(possible_nodes_values: dict) -> dict:
	tree = Tree()
	tree.create_node("Root", "Root")
	root_node_data = set_node_data(0, possible_nodes_values)
	tree.get_node("Root").data = root_node_data
	generate_tree(tree, 0, MAX_HEIGHT, "", possible_nodes_values)

	return {
		"fitness": 0,
		"tree": tree,
		"value": []
	}


def set_node_data(current_height: int, possible_nodes_values: dict) -> NodeData:
	data = {"type": None, "value": None}
	if current_height == 0:
		data["type"] = "function"
	elif current_height < MAX_HEIGHT - 1:
		data["type"] = random.choice(
			a=["terminal", "constant", "function"],
			p=[CHANCE_TERMINAL, CHANCE_CONSTANT, CHANCE_FUNCTION]
		)
	else:
		data["type"] = random.choice(
			a=["terminal", "constant"],
			p=[CHANCE_TERMINAL / (CHANCE_TERMINAL + CHANCE_CONSTANT), CHANCE_CONSTANT / (CHANCE_TERMINAL + CHANCE_CONSTANT)],
		)
	data["value"] = random.choice(possible_nodes_values[data["type"]])
	return NodeData(data["type"], data["value"])


def generate_tree(tree: Tree, current_height: int, max_height: int, parent_node_name: str, possible_nodes_values: dict):
	if current_height == max_height:
		return

	left_node_name = parent_node_name + "L"
	right_node_name = parent_node_name + "R"

	if parent_node_name == "":
		parent_node_name = "Root"

	# left child
	left_node_data = set_node_data(current_height, possible_nodes_values)
	tree.create_node(left_node_name, left_node_name, parent=parent_node_name, data=left_node_data)

	if left_node_data.node_type == "function":
		generate_tree(tree, current_height+1, max_height, left_node_name, possible_nodes_values)

	# right child
	right_node_data = set_node_data(current_height, possible_nodes_values)
	tree.create_node(right_node_name, right_node_name, parent=parent_node_name, data=right_node_data)

	if right_node_data.node_type == "function":
		generate_tree(tree, current_height + 1, max_height, right_node_name, possible_nodes_values)


def calculate_fitness(tree_obj: dict, terminal_set: list, x_set: list, y_set: pd.Series):
	tree_obj.update({"fitness": 0, "tree": tree_obj["tree"], "value": []})
	for i in range(len(y_set)):
		tree_obj["value"].append(float(resolve_tree(tree_obj["tree"], "Root", terminal_set, x_set[i])))
	tree_obj["fitness"] = estimate_fitness(tree_obj["value"], y_set)
	return tree_obj


def resolve_tree(tree: Tree, node_name: str, terminal_set: list, x_set: list) -> str:
	node = tree.get_node(node_name)

	if node.data.node_type == "terminal":
		terminal_index = terminal_set.index(node.data.terminal_value)
		node.data.node_value = str(x_set[terminal_index])

	children = node.fpointer
	if not children:
		return node.data.node_value

	node_name = "" if node.tag == "Root" else node.tag

	left_node_name = node_name + "L"
	right_node_name = node_name + "R"

	left_value = resolve_tree(tree, left_node_name, terminal_set, x_set)
	right_value = resolve_tree(tree, right_node_name, terminal_set, x_set)
	try:
		return str(eval(left_value + " " + node.data.node_value + " " + right_value))
	except ZeroDivisionError:
		return str((eval(left_value + " " + node.data.node_value + " 1")))


def estimate_fitness(tree_value: list, outputs: pd.Series) -> float:
	numerator = 0
	denominator = 0
	mean = outputs.mean()
	for i in range(len(outputs)):
		numerator += math.pow(outputs[i] - tree_value[i], 2)
		denominator += math.pow(outputs[i] - mean, 2)
	try:
		return math.sqrt(numerator/denominator)
	except ZeroDivisionError:
		return 0


def operator_crossover(tree_a_obj: dict, tree_b_obj: dict, terminal_set: list, x_set: list, y_set: pd.Series) -> tuple:
	tree_a = tree_a_obj["tree"]
	tree_b = tree_b_obj["tree"]
	tree_a_nodes = list(tree_a.nodes.keys())
	tree_b_nodes = list(tree_b.nodes.keys())
	possible_nodes = []
	for item in tree_a_nodes:
		if item in tree_b_nodes and item not in possible_nodes:
			possible_nodes.append(item)
	for item in tree_b_nodes:
		if item in tree_a_nodes and item not in possible_nodes:
			possible_nodes.append(item)
	# print(possible_nodes)
	possible_nodes.pop(possible_nodes.index("Root"))
	crossover_node_tag = random.choice(possible_nodes)

	child_ab_obj = dict({"fitness": 0, "tree": None, "value": []})
	child_ab_obj["tree"] = copy.deepcopy(tree_a)
	child_ab_obj["tree"].remove_node(crossover_node_tag)
	child_ab_obj["tree"].paste(tree_a.get_node(crossover_node_tag).bpointer, tree_b.subtree(crossover_node_tag))
	child_ab_obj = calculate_fitness(child_ab_obj, terminal_set, x_set, y_set)

	child_ba_obj = dict({"fitness": 0, "tree": None, "value": []})
	child_ba_obj["tree"] = copy.deepcopy(tree_b)
	child_ba_obj["tree"].remove_node(crossover_node_tag)
	child_ba_obj["tree"].paste(tree_b.get_node(crossover_node_tag).bpointer, tree_a.subtree(crossover_node_tag))
	child_ba_obj = calculate_fitness(child_ba_obj, terminal_set, x_set, y_set)

	if ELITISM:
		result_list = sorted([tree_a_obj, tree_b_obj, child_ab_obj, child_ba_obj], key=lambda k: k['fitness'])
		return result_list[0], result_list[1]

	return child_ab_obj, child_ba_obj


def operator_mutation(tree_obj: dict, possible_nodes_values: dict, x_set: list, y_set: pd.Series):
	tree = tree_obj["tree"]
	tree_nodes = list(tree.nodes.keys())
	mutated_node_name = random.choice(tree_nodes)
	node = tree.get_node(mutated_node_name)
	new_value = random.choice(possible_nodes_values[node.data.node_type])

	child_obj = dict()
	child_obj["tree"] = copy.deepcopy(tree)
	child_obj["tree"].get_node(mutated_node_name).data.node_value = str(new_value)
	child_obj = calculate_fitness(child_obj, possible_nodes_values["terminal"], x_set, y_set)

	if ELITISM:
		return child_obj if child_obj["fitness"] >= tree_obj["fitness"] else tree_obj

	return child_obj


def tournament(population: list):
	participants = random.choice(
			a=population,
			size=TOURNAMENT_SIZE,
			replace=False
		)
	participants = sorted(participants, key=lambda k: k['fitness'])
	return participants[0]


def find_best_fitness(population: list):
	best_fitness = population[0]["fitness"]
	for individual in population:
		best_fitness = min(best_fitness, individual["fitness"])
	return best_fitness


def start():
	# read data from file
	x_set, y_set = read_data('datasets/synth1/synth1-train.csv')

	# define set of functions and terminals
	possible_nodes_values = {
		"constant": [random.uniform(-1, 1) for _ in range(CONSTANTS)],
		"terminal": ["X1", "X2"],
		"function": ["+", "-", "*", "/"]
	}

	generations = []
	# generate starting population
	population = []
	for k in range(INDIVIDUALS):
		# print("individuo", k, "---------------")
		population.append(generate_individual(possible_nodes_values))

	# evaluate the fitness of each individual
	i = 0
	for ind in population:
		# print("fitness", i, "---------------")
		calculate_fitness(ind, possible_nodes_values["terminal"], x_set, y_set)
		i += 1

	for j in range(GENERATIONS):
		# selection
		new_population = []
		# tournament selection
		if TOURNAMENT:
			# print("torneio", "---------------")
			new_population.append(tournament(population))

		# applying operators
		# print("operadores", "---------------")
		did_crossover = False
		for i in range(0, INDIVIDUALS):
			if did_crossover:
				did_crossover = False
				continue
			if random.randrange(1000) < CHANCE_CROSSOVER:
				ind_a, ind_b = operator_crossover(
					population[i], population[(i+1) % INDIVIDUALS], possible_nodes_values["terminal"], x_set, y_set
				)
				new_population.extend((ind_a, ind_b))
				did_crossover = True
			elif random.randrange(1000) < CHANCE_MUTATION:
				new_population.append(operator_mutation(population[i], possible_nodes_values, x_set, y_set))
			else:
				new_population.append(population[i])

		generations.append(new_population)
		population = copy.deepcopy(new_population)
		generations[j] = sorted(generations[j], key=lambda k: k['fitness'])
		print("--------------------------", j, generations[j][0]["fitness"])
	best_ind_fitness = [x[0]["fitness"] for x in generations]
	plt.plot(best_ind_fitness)
	plt.xlabel("Number of generations")
	plt.ylabel("NRMSE")
	plt.title("Best Individual")
	plt.show()

	average_fitness = []
	for generation in generations:
		average_fitness.append(sum([ind["fitness"] for ind in generation]) / len(generation))
	plt.plot(average_fitness)
	plt.xlabel("Number of generations")
	plt.ylabel("NRMSE")
	plt.title("Average Fitness of the Population")
	plt.show()
	print()
