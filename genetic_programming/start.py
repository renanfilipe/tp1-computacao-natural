from treelib import Node, Tree
import pandas as pd

def start():
	data = pd.read_csv('datasets/synth1/synth1-train.csv', header=None)
	funcoes = ['+', '-', '*', '/']
	terminais = [-1.235928614240245, -1.3641055948703154]

	tree = Tree()
	tree.create_node("Harry", "harry")  # root node
	tree.create_node("Jane", "jane", parent="harry")
	tree.create_node("Bill", "bill", parent="harry")
	tree.create_node("Diane", "diane", parent="jane")
	tree.create_node("Mary", "mary", parent="diane")
	tree.create_node("Mark", "mark", parent="jane")
	tree.show()