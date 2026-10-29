import pandas as pd
import random

class Node:
    #creating and storing info for tree
    def __int__(self, name, mutation):
        self.name = name
        self.mutation = mutation

        self.children = []
    #adding child
    def add_child(self, child_node):
        self.children.append(child_node)
    #printing the tree
    def print_mutation(self, level = 0):
        print(f"\t" * level, self.mutation)
        for child in self.children:
            print_mutation(level + 1)
        
# calculate the F value to pick mutation
# return the mutation with the best F value
def cal_F(df, group):
    

# calculate the phi value to pick mutation
# return mutation with the best phi value
def cal_phi(df, group):
    pass

# calculate all stats for the trees
# return the stats
def cal_stats(df, group):
    pass

# print stats, prints all the stats
def print_stats():
    pass 

#  build a tree based of off F value
# return tree
def F_build_tree(node, group):
    pass

# build a tree based off of phi value
# return tree
def phi_build_tree(node, group):
    pass

def main():
    random.seed(72)
    samples = []
    #loading the cvs into the df
    df = pd.read_csv('mutationu.csv')

    for i in range(len(df)):
        samples.append(i)
    random.shuffle(samples)

    # creating the three gorups
    group_A = samples[:81]
    group_B = samples[81:162]
    group_C = samples[162:]






if __name__ == "__main__":
    main()