import pandas as pd
import numpy as np
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
    #only looking for mutations in the group
    test_group = df.loc[group]
    #getting those who have cancer and the mutations to run through to calculate F
    has_cancer = test_group.iloc[:, 0].astype(str).str.startswith("C").to_numpy()
    not_cancer = ~has_cancer
    mutations = test_group.columns[1:]
    #getting all samples that have a mutation
    has_mutation = test_group[mutations].to_numpy() == 1

    #calculating tp and fp
    tp = (has_mutation & has_cancer[:, None]).sum(axis = 0)
    fp = (has_mutation & not_cancer[:, None]).sum(axis = 0)
    #calculate all F values
    F_value = pd.Series(tp - fp, index = mutations)

    return F_value.idxmax()


# calculate the phi value to pick mutation
# return mutation with the best phi value
def cal_phi(df, group):
    #only looking for mutations in the group
    test_group = df.loc[group]
    #getting those who have cancer and the mutations to run through to calculate phi
    has_cancer = test_group.iloc[:, 0].astype(str).str.startswith("C").to_numpy()
    not_cancer = ~has_cancer
    mutations = test_group.columns[1:]
    
    n_t = len(test_group)

    #for each mutation, which samples go left / right
    go_left = test_group[mutations].to_numpy() == 1
    go_right = ~go_left

    #n(tL), n(tR)
    n_tL = go_left.sum(axis=0)
    n_tR = go_right.sum(axis=0)

    #n(tL,C), n(tL,NC), n(tR,C), n(tR,NC)
    n_tL_C  = (go_left  & has_cancer[:, None]).sum(axis=0)
    n_tL_NC = (go_left  & not_cancer[:, None]).sum(axis=0)
    n_tR_C  = (go_right & has_cancer[:, None]).sum(axis=0)
    n_tR_NC = (go_right & not_cancer[:, None]).sum(axis=0)

    #PL and PR
    PL = n_tL / n_t
    PR = n_tR / n_t

    #P(C|tL), P(NC|tL), P(C|tR), P(NC|tR)  (NaN if a child is empty)
    with np.errstate(divide="ignore", invalid="ignore"):
        P_C_tL  = n_tL_C  / n_tL
        P_NC_tL = n_tL_NC / n_tL
        P_C_tR  = n_tR_C  / n_tR
        P_NC_tR = n_tR_NC / n_tR

    #Q(s|t)
    Q = np.abs(P_C_tL - P_C_tR) + np.abs(P_NC_tL - P_NC_tR)

    #F = phi(s,t) = 2*PL*PR*Q
    F = pd.Series(2 * PL * PR * Q, index=mutations)

    #best mutation = largest F (NaNs are skipped automatically)
    return F.idxmax()

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

    best_mutation = cal_F(df, samples)
    print(best_mutation)
    best_mutation = cal_phi(df, samples)
    print(best_mutation)




if __name__ == "__main__":
    main()