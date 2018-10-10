from treelib import Node, Tree
import pandas as pd
from .not_random import NotRandom
import math
import copy
import matplotlib.pyplot as plt


FILE_TRAIN = 'datasets/synth1/synth1-train.csv'
FILE_TEST = 'datasets/synth1/synth1-test.csv'

CHANCE_TERMINAL = 0.7
CHANCE_CONSTANT = 0.15
CHANCE_FUNCTION = 0.15

CHANCE_CROSSOVER = 700
CHANCE_MUTATION = 5
CONSTANTS = 50

NUMBER_OF_TESTS = 5
GENERATIONS = 10
INDIVIDUALS = 50
MAX_HEIGHT = 7
ELITISM = True
TOURNAMENT = False
TOURNAMENT_SIZE = 10
PLOT = False


class GlobalVar:
	def __init__(self):
		self.better_parents = 0
		self.better_children = 0

	def reset(self):
		self.better_parents = 0
		self.better_children = 0


class NodeData:
	def __init__(self, node_type, node_value):
		self.node_type = node_type
		self.node_value = str(node_value)
		self.terminal_value = str(node_value)


global_var = GlobalVar()
random = NotRandom()


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
		average_parent_fitness = (tree_a_obj["fitness"] + tree_b_obj["fitness"]) / 2
		if child_ab_obj["fitness"] > average_parent_fitness:
			global_var.better_children += 1
		else:
			global_var.better_parents += 1

		if child_ba_obj["fitness"] > average_parent_fitness:
			global_var.better_children += 1
		else:
			global_var.better_parents += 1

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
		if child_obj["fitness"] > tree_obj["fitness"]:
			global_var.better_children += 1
			return child_obj
		else:
			global_var.better_parents += 1
			return tree_obj

	return child_obj


def tournament(population: list):
	participants = random.choice(
			a=population,
			size=TOURNAMENT_SIZE,
			replace=False
		)
	participants = sorted(participants, key=lambda k: k['fitness'])
	return participants[0]


def check_for_clones(population):
	index_a = 0
	index_b = 1
	clones = 0
	while index_b < len(population):
		if population[index_a]["fitness"] == population[index_b]["fitness"] and \
			sum(population[index_a]["value"]) == sum(population[index_b]["value"]):
			clones += 1
			index_b += 1
		else:
			index_a += 1
			index_b += 1
	return clones


def start():
	# read data from file
	x_set_train, y_set_train = read_data(FILE_TRAIN)
	x_set_test, y_set_test = read_data(FILE_TEST)

	list_of_results = { "train": [], "test": [] }

	for l in range(NUMBER_OF_TESTS):
		print("iteration", l)

		random.update_seed(l)

		# define set of functions and terminals
		possible_nodes_values = {
			"constant": [random.uniform(-1, 1) for _ in range(CONSTANTS)],
			"terminal": ["X1", "X2"],
			"function": ["+", "-", "*", "/"]
		}

		results = {
			"best_fitness": [],
			"worst_fitness": [],
			"average_fitness": [],
			"clones": [],
			"superior_children": [],
			"superior_parents": []
		}

		generation = []
		# generate starting population
		population = []
		for k in range(INDIVIDUALS):
			population.append(generate_individual(possible_nodes_values))

		# evaluate the fitness of each individual
		for ind in population:
			calculate_fitness(ind, possible_nodes_values["terminal"], x_set_train, y_set_train)

		population = sorted(population, key=lambda k: k['fitness'])

		results["best_fitness"].append(population[0]["fitness"])
		results["worst_fitness"].append(population[-1]["fitness"])
		results["average_fitness"].append(sum([ind["fitness"] for ind in population]) / len(population))
		results["clones"].append(check_for_clones(population))
		results["superior_children"].append(global_var.better_children)
		results["superior_parents"].append(global_var.better_parents)

		global_var.reset()

		for j in range(GENERATIONS):
			# selection
			new_population = []
			# tournament selection
			if TOURNAMENT:
				new_population.append(tournament(population))

			# applying operators
			did_crossover = False
			for i in range(0, INDIVIDUALS):
				if did_crossover:
					did_crossover = False
					continue
				if random.randrange(1000) < CHANCE_CROSSOVER:
					ind_a, ind_b = operator_crossover(
						population[i], population[(i+1) % INDIVIDUALS], possible_nodes_values["terminal"], x_set_train, y_set_train
					)
					new_population.extend((ind_a, ind_b))
					did_crossover = True
				elif random.randrange(1000) < CHANCE_MUTATION:
					new_population.append(operator_mutation(population[i], possible_nodes_values, x_set_train, y_set_train))
				else:
					new_population.append(population[i])

			generation.append(new_population)
			population = copy.deepcopy(new_population)
			generation[j] = sorted(generation[j], key=lambda k: k['fitness'])

			results["best_fitness"].append(generation[j][0]["fitness"])
			results["worst_fitness"].append(generation[j][-1]["fitness"])
			results["average_fitness"].append(sum([ind["fitness"] for ind in generation[j]]) / len(generation[j]))
			results["clones"].append(check_for_clones(generation[j]))
			results["superior_children"].append(global_var.better_children)
			results["superior_parents"].append(global_var.better_parents)

			global_var.reset()

			# print(j, generation[j][0]["fitness"])

			if PLOT:
				best_ind_fitness = [x[0]["fitness"] for x in generation]
				plt.plot(best_ind_fitness)
				plt.xlabel("Number of generations")
				plt.ylabel("NRMSE")
				plt.title("Best Individual")
				plt.show()

				average_fitness = []
				for generation in generation:
					average_fitness.append(sum([ind["fitness"] for ind in generation]) / len(generation))
				plt.plot(average_fitness)
				plt.xlabel("Number of generation")
				plt.ylabel("NRMSE")
				plt.title("Average Fitness of the Population")
				plt.show()

		list_of_results["train"].append(results)

		best_generation = copy.deepcopy(generation[-1])

		# evaluate the fitness of each individual
		for ind in best_generation:
			calculate_fitness(ind, possible_nodes_values["terminal"], x_set_test, y_set_test)

		best_generation = sorted(best_generation, key=lambda k: k['fitness'])

		print(l, best_generation[0]["fitness"])

		results_test = {}
		results_test["best_fitness"] = best_generation[0]["fitness"]
		results_test["worst_fitness"] = best_generation[-1]["fitness"]
		results_test["average_fitness"] = sum([ind["fitness"] for ind in best_generation]) / len(best_generation)
		results_test["clones"] = check_for_clones(best_generation)
		results_test["superior_children"] = global_var.better_children
		results_test["superior_parents"] = global_var.better_parents

		global_var.reset()

		list_of_results["test"].append(results_test)

		print()
	print()
