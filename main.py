import pandas as pd
import numpy as np
import math
import random

class Node:
    #creating and storing info for tree
    def __init__(self, name, mutation = None, past_mutations = []):
        self.name = name
        self.mutation = mutation
        self.split = None
        self.leaf_lable = None
        self.children = []
        self.past_mutations = past_mutations
        self.samples = []
        self.samples_name = []

    # adding child
    def add_child(self, child_node):
        self.children.append(child_node)
    # adding samples
    def add_sample(self, sample):
        self.samples.append(sample)
    # adding the name of the sample not just location ie C1 NC3
    def add_sample_name(self, sample):
        self.samples_name.append(sample)
    # adding past mutations
    def add_past_mutation(self, mutations):
        self.past_mutations = mutations.copy()
    # printing the tree
    def print_mutation(self, level = 0):
        if self.children:
            print(f"\t" * level, self.name,  "children split with", self.split[:10])
        else:
            print(f"\t" * level, self.name, self.mutation[:10], "leaf node", self.leaf_lable)
        for child in self.children:
            child.print_mutation(level + 1)
    
    def print_samples(self, level = 0):
        if self.children == []:
            print(f"\t" * level, self.name, self.samples_name[:10], "total samples:", len(self.samples))
        else:
            print(f"\t" * level, self.name)
            for child in self.children:
                child.print_samples(level + 1)      
# calculate the F value to pick mutation
# return the mutation with the best F value
def cal_F(df, group, past_mutations):
    #only looking for mutations in the group
    test_group = df.loc[group]
    #getting those who have cancer and the mutations to run through to calculate F
    has_cancer = test_group.iloc[:, 0].astype(str).str.startswith("C").to_numpy()
    not_cancer = ~has_cancer
    mutations = test_group.columns[1:]
    mutations = [m for m in mutations if m not in past_mutations]
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
def cal_phi(df, group, past_mutations):
    #only looking for mutations in the group
    test_group = df.loc[group]
    #getting those who have cancer and the mutations to run through to calculate phi
    has_cancer = test_group.iloc[:, 0].astype(str).str.startswith("C").to_numpy()
    not_cancer = ~has_cancer
    mutations = test_group.columns[1:]
    mutations = [m for m in mutations if m not in past_mutations]

    
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

# calculate the gain value
# return the best mutation for H(t) value
def cal_H(df, group, past_mutations, mutations = []):
    #only looking for mutations in the group
    test_group = df.loc[group]
    #getting those who have cancer and the mutations to run through to calculate phi
    has_cancer = test_group.iloc[:, 0].astype(str).str.startswith("C").to_numpy()
    not_cancer = ~has_cancer
    # allows this functions to either pick from all mutations or a given amount
    if mutations == []:
        mutations = test_group.columns[1:]
    mutations = [m for m in mutations if m not in past_mutations]

    # n(t) n(t, C) N(t, NC)
    n_t = len(test_group)
    n_tC = has_cancer.sum()
    n_tNC = not_cancer.sum()

    #P(c,t) and P(nc, t) = n(t, NC) / n(t)
    P_c_t = n_tC / n_t
    P_nc_t = n_tNC / n_t

    # H(t)
    H_t = -sum(p * np.log2(p) for p in [P_c_t, P_nc_t] if p > 0)

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

    # P(tl, C) P(tl, NC) P(tR, C) P(tR, NC) = n(tR, NC) / n(tR)
    p_tL_C = np.divide(n_tL_C, n_tL, out=np.zeros_like(n_tL_C, dtype=float), where=n_tL != 0)
    p_tL_NC = np.divide(n_tL_NC, n_tL, out=np.zeros_like(n_tL_NC, dtype=float), where=n_tL != 0)
    p_tR_C = np.divide(n_tR_C, n_tR, out=np.zeros_like(n_tR_C, dtype=float), where=n_tR != 0)
    p_tR_NC = np.divide(n_tR_NC, n_tR, out=np.zeros_like(n_tR_NC, dtype=float), where=n_tR != 0)

    # H(s, t)
    H_s_t = PL * entropy(p_tL_C, p_tL_NC) + PR * entropy(p_tR_C, p_tR_NC)

    # gains
    gains = pd.Series(H_t - H_s_t, index = mutations)
    gains = gains.sort_values(ascending = False)

    # print(gains.head(10))
    return gains.idxmax()

# calcualte the H(t) value
# retun H(t)
def entropy(P_c_t, P_nc_t):
    # H(t)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        return -( 
            np.where(P_c_t > 0, P_c_t * np.log2(P_c_t), 0) +
            np.where(P_nc_t > 0, P_nc_t * np.log2(P_nc_t), 0)
        )

# calculate tp, fp, tn, fn for the tree made by F values
# return tp, fp, tn, fn
def F_stats(df, node):
    #checking if in leaf node
    if not node.children:
        samples = df.loc[node.samples]
        if node.name == "Child_with": # saying everyone has cancer
            tp = samples.iloc[:, 0].astype(str).str.startswith("C").sum()
            fp = samples.iloc[:, 0].astype(str).str.startswith("N").sum()
            return tp, fp, 0, 0
        else: # saying no one has cancer
            tn = samples.iloc[:, 0].astype(str).str.startswith("N").sum()
            fn = samples.iloc[:, 0].astype(str).str.startswith("C").sum()
            return 0, 0, tn, fn
    else:
        tp1, fp1, tn1, fn1 = F_stats(df, node.children[0])
        tp2, fp2, tn2, fn2 = F_stats(df, node.children[1])

    return (
        tp1 + tp2,
        fp1 + fp2,
        tn1 + tn2,
        fn1 + fn2
    )

# calculate tp, fp, tn, fn for the tree made by phi values also works for Gains values
# return tp, fp, tn, fn
def phi_stats_cal(df, node):
    #checking if in leaf node
    if not node.children:
        samples = df.loc[node.samples]
        if node.leaf_lable == "cancer": # saying everyone has cancer
            tp = samples.iloc[:, 0].astype(str).str.startswith("C").sum()
            fp = samples.iloc[:, 0].astype(str).str.startswith("N").sum()
            return tp, fp, 0, 0
        else: # saying no one has cancer
            tn = samples.iloc[:, 0].astype(str).str.startswith("N").sum()
            fn = samples.iloc[:, 0].astype(str).str.startswith("C").sum()
            return 0, 0, tn, fn
    else:
        tp1, fp1, tn1, fn1 = phi_stats_cal(df, node.children[0])
        tp2, fp2, tn2, fn2 = phi_stats_cal(df, node.children[1])

    return (
        tp1 + tp2,
        fp1 + fp2,
        tn1 + tn2,
        fn1 + fn2
    )

# sorting a sample into the correct leaf node to then be clasified
def sort(df, sample, node):
    # checking if a the last node before a leaf node and if it is adding the sample to the node
    if node.children == []:
        node.add_sample(sample)
        node.add_sample_name(df.iloc[sample, 0])
        return
    # the node isn't a leaf node and sending the sample down another layer
    if df.loc[sample, node.split] == 1:
        sort(df, sample, node.children[0])
    else:
        sort(df, sample, node.children[1])

# build a tree based of off F value
# return tree
def F_build_tree(df, node, group, level):
    # checking to make sure no splitting an empty group
    # also returning if reached the wanted hight of the tree
    if level == 0 or len(group) <= 1:
        return node
    
    top_mutation = cal_F(df, group, node.past_mutations)
    #finding the top mutation
    top_mutation = cal_F(df, group, node.past_mutations)
    # setting the nodes mutation and keeping track of path
    node.split = top_mutation
    past_mutations = node.past_mutations.copy()
    past_mutations.append(top_mutation)
    
    node_group = df.loc[group]
    # find the samples that do and don't have the mutation
    has_mutation = node_group[top_mutation] == 1
    # have the indexes instead of a list of bools
    group_with = has_mutation.index[has_mutation]
    group_without = has_mutation.index[~has_mutation]

    # creating the child nodes with the groups of who do and don't have the mutation
    child_with = Node("Child_with", top_mutation)
    child_without = Node("Child_without", top_mutation)
    # keeping track of all mutations used
    child_with.add_past_mutation(past_mutations)
    child_without.add_past_mutation(past_mutations)
    # adding them to their parent node to build the tree
    node.add_child(child_with)
    node.add_child(child_without)
    # building the next layer of the tree
    F_build_tree(df, child_with, group_with, level - 1)
    F_build_tree(df, child_without, group_without, level - 1)

    return node

# build a tree based off of phi value
# return tree
def phi_build_tree(df, node, group, level):
    # checking to make sure not splitting an empty group
    # also returning if reached the wanted hight of the tree
    if level == 0 or len(group) <= 1:
        samples = df.loc[group] # getting all samples in the group
        total_C_samples = samples.iloc[:,0].astype(str).str.startswith("C").sum() # getting the total number of caner sample in the node
        if total_C_samples >= len(group) - total_C_samples: # if there are more cancer samples then label node cancer
            node.leaf_lable = "cancer"
        else:
            node.leaf_lable = "not cancer"
        return node
    
    # finding the top mutation
    top_mutation = cal_phi(df, group, node.past_mutations)
    # setting the nodes mutation and keeping track of path
    node.split = top_mutation
    past_mutations = node.past_mutations.copy()
    past_mutations.append(top_mutation)

    # create a group all only the samples in this group instead of whole data set
    node_group = df.loc[group]
    # find the samples that do and don't have the mutation
    has_mutation = node_group[top_mutation] == 1
    # have the indexes instead of a list of bools
    group_with = has_mutation.index[has_mutation]
    group_without = has_mutation.index[~has_mutation]

    # creating the child nodes with the groups of who do and don't have the mutation
    child_with = Node("Child_with", top_mutation)
    child_without = Node("Child_without", top_mutation)
    # keeping track of all mutations used
    child_with.add_past_mutation(past_mutations)
    child_without.add_past_mutation(past_mutations)
    # adding them to their parent node to build the tree
    node.add_child(child_with)
    node.add_child(child_without)
    # building the next layer of the tree
    phi_build_tree(df, child_with, group_with, level - 1)
    phi_build_tree(df, child_without, group_without, level - 1)

    return node

# build a tree using H as the value
# return tree
def H_build_tree(df, node, group, level):
    # checking to make sure not splitting an empty group
    # also returning if reached the wanted hight of the tree
    if level == 0 or len(group) <= 1:
        samples = df.loc[group] # getting all samples in the group
        total_C_samples = samples.iloc[:,0].astype(str).str.startswith("C").sum() # getting the total number of caner sample in the node
        if total_C_samples >= len(group) - total_C_samples: # if there are more cancer samples then label node cancer
            node.leaf_lable = "cancer"
        else:
            node.leaf_lable = "not cancer"
        return node
    
    # finding the top mutation
    top_mutation = cal_H(df, group, node.past_mutations)
    # setting the nodes mutation and keeping track of path
    node.split = top_mutation
    past_mutations = node.past_mutations.copy()
    past_mutations.append(top_mutation)

    # create a group all only the samples in this group instead of whole data set
    node_group = df.loc[group]
    # find the samples that do and don't have the mutation
    has_mutation = node_group[top_mutation] == 1
    # have the indexes instead of a list of bools
    group_with = has_mutation.index[has_mutation]
    group_without = has_mutation.index[~has_mutation]

    # creating the child nodes with the groups of who do and don't have the mutation
    child_with = Node("Child_with", top_mutation)
    child_without = Node("Child_without", top_mutation)
    # keeping track of all mutations used
    child_with.add_past_mutation(past_mutations)
    child_without.add_past_mutation(past_mutations)
    # adding them to their parent node to build the tree
    node.add_child(child_with)
    node.add_child(child_without)

    # building the next layer of the tree
    H_build_tree(df, child_with, group_with, level - 1)
    H_build_tree(df, child_without, group_without, level - 1)

    return node

# build a tree using H as the spilt but also using random forest classifier
# return tree
def forest_build_tree(df, node, samples_loc, level):
    # checking to make sure not splitting an empty group
    # also returning if reached the wanted hight of the tree
    if level == 0 or len(samples_loc) <= 1:
        samples = df.loc[samples_loc] # getting all samples in the group
        total_C_samples = samples.iloc[:,0].astype(str).str.startswith("C").sum() # getting the total number of caner sample in the node
        if total_C_samples >= len(samples_loc) - total_C_samples: # if there are more cancer samples then label node cancer
            node.leaf_lable = "cancer"
        else:
            node.leaf_lable = "not cancer"
        return node
    # getting the random mutations to split from
    mutations = random_mut(df)
    top_mutation = cal_H(df, samples_loc, node.past_mutations, mutations)
    # keeping track of what mutation was used for split
    node.split = top_mutation
    past_mutations = node.past_mutations.copy()
    past_mutations.append(top_mutation)
    # getting the samples and splitting them into with and without groups
    samples = df.loc[samples_loc]
    has_mutation = samples[top_mutation] == 1
    group_with = has_mutation.index[has_mutation]
    group_without = has_mutation.index[~has_mutation]
    # creating children
    child_with = Node("child_with", top_mutation, past_mutations)
    child_without = Node("child_without", top_mutation, past_mutations)
    # adding the childen to node
    node.add_child(child_with)
    node.add_child(child_without)
    # creating the next level
    forest_build_tree(df, child_with, group_with, level - 1)
    forest_build_tree(df, child_without, group_without, level - 1)

    return node

# create the bootstrap and out of bag data
# return the bootstrap data and the out of bag data
def boot_strap_data(df):
    n = len(df)
    bsd = []
    oob = []

    for i in range(n):
        bsd.append(random.randint(0, n - 1))
    selected = set(bsd)
    oob = [i for i in range(n) if i not in selected and i not in oob]

    return bsd, oob

# create a list of sqrt(n) random mutations
# return random mutations
def random_mut(df):
    # calcualte the sqrt
    num_mut = len(df.columns)
    sqrt_mut = int(math.sqrt(num_mut))
    # get list of all mutation and shuffle
    mutations = list(df.columns)
    random.shuffle(mutations)
    # return the first sqrt(n) mutations
    return mutations[:sqrt_mut]

# allow for safe deviding
def safe_dev(num, den):
    return num / den if den != 0 else 0

# give tp, fp, tn, fn
# return all required stats
def evaluate(tp, fp, tn, fn):
    accuracy = safe_dev((tp + tn), (tp + tn + fp + fn))
    sensitivity = safe_dev(tp, (tp + fn))
    specificity = safe_dev(tn, (tn + fp))
    precision = safe_dev(tp, (tp + fp))
    miss_rate = safe_dev(fn, (tp + fn))
    false_discovery_rate = safe_dev(fp, (tp + fp))
    false_omission_rate = safe_dev(fn, (tn + fn))

    return {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "Miss Rate": miss_rate,
            "False Discovery Rate": false_discovery_rate,
            "False Omission Rate": false_omission_rate
            }

# given samples, average them
# return the averages
def ave(stats1, stats2, stats3):
    tp = (stats1[0] + stats2[0] + stats3[0]) / 3
    fp = (stats1[1] + stats2[1] + stats3[1]) / 3
    tn = (stats1[2] + stats2[2] + stats3[2]) / 3
    fn = (stats1[3] + stats2[3] + stats3[3]) / 3

    return (tp, fp, tn, fn)

# printing all the stats
def print_stats(f_stats, phi_stats, H_stats):
    # calculate stats for f trees
    tp = f_stats[0]
    fp = f_stats[1]
    tn = f_stats[2]
    fn = f_stats[3]

    f_value_stats = evaluate(tp, fp, tn, fn)

    # calculate stats for phi trees
    tp = phi_stats[0]
    fp = phi_stats[1]
    tn = phi_stats[2]
    fn = phi_stats[3]

    phi_value_stats = evaluate(tp, fp, tn, fn)

    # calculate stats for H trees
    tp = H_stats[0]
    fp = H_stats[1]
    tn = H_stats[2]
    fn = H_stats[3]

    H_value_stats = evaluate(tp, fp, tn, fn)

    # printing everything
    print(f"{'Metric':<20} {'F Tree':<10} {'Phi Tree':<10} {'H tree':<10}")
    for stat in f_value_stats:
        print(f"{stat:<20} {f_value_stats[stat]:<10.4f} {phi_value_stats[stat]:<10.4f} {H_value_stats[stat]:<10.4f}")

def before_forest(df):
    for i in range(len(df)):
        samples.append(i)
    random.shuffle(samples)

    # creating the three groups
    group_A = samples[:81]
    group_B = samples[81:162]
    group_C = samples[162:]

    # F trees 
    # creating the roots
    F_group_A = Node("Root", None)
    F_group_B = Node("Root", None)
    F_group_C = Node("Root", None)
    # # building the trees
    F_group_A = F_build_tree(df, F_group_A, group_B + group_C, 2)
    F_group_B = F_build_tree(df, F_group_B, group_A + group_C, 2)
    F_group_C = F_build_tree(df, F_group_C, group_B + group_A, 2)
    # # sorting lists
    for sample in group_A:
        sort(df, sample, F_group_A)
    for sample in group_B:
        sort(df, sample, F_group_B)
    for sample in group_C:
        sort(df, sample, F_group_C)

    # adding all stats
    # averging all stats
    f_stats = ave(F_stats(df, F_group_A), F_stats(df, F_group_B), F_stats(df, F_group_C))
    
    # phi trees 
    # creating the roots
    phi_group_A = Node("Root", None)
    phi_group_B = Node("Root", None)
    phi_group_C = Node("Root", None)
    # # building the trees
    phi_group_A = phi_build_tree(df, phi_group_A, group_B + group_C, 2)
    phi_group_B = phi_build_tree(df, phi_group_B, group_A + group_C, 2)
    phi_group_C = phi_build_tree(df, phi_group_C, group_B + group_A, 2)
    # # sorting lists
    for sample in group_A:
        sort(df, sample, phi_group_A)
    for sample in group_B:
        sort(df, sample, phi_group_B)
    for sample in group_C:
        sort(df, sample, phi_group_C)

    # adding all stats
    phi_stats = ave(phi_stats_cal(df, phi_group_B), phi_stats_cal(df, phi_group_C), phi_stats_cal(df, phi_group_A))

    # gains
    # creating the roots
    H_group_A = Node("Root", None)
    H_group_B = Node("Root", None)
    H_group_C = Node("Root", None)
    
    # # building the trees
    H_group_A = H_build_tree(df, H_group_A, group_B + group_C, 2)
    H_group_B = H_build_tree(df, H_group_B, group_A + group_C, 2)
    H_group_C = H_build_tree(df, H_group_C, group_B + group_A, 2)
    # # sorting lists
    for sample in group_A:
        sort(df, sample, H_group_A)
    for sample in group_B:
        sort(df, sample, H_group_B)
    for sample in group_C:
        sort(df, sample, H_group_C)

    # adding all stats
    H_stats = ave(phi_stats_cal(df, H_group_A), phi_stats_cal(df, H_group_B), phi_stats_cal(df, H_group_C))
   
    print("F tree")
    F_group_A.print_mutation()
    print("phi tree")
    phi_group_A.print_mutation()
    print("H tree")
    H_group_A.print_mutation()

    print_stats(f_stats, phi_stats, H_stats)

# create the forest of size n
# return the forest
def build_forest(df, size, bsd):
    forest = []
    for i in range(size):
        tree = Node("root", None)
        tree = forest_build_tree(df, tree, bsd, 2)
        forest.append(tree)

    return forest

# return the vote for a single tree to be added later
def vote(df, sample, node):
    # checking if a the last node before a leaf node and if it is adding the sample to the node
    if node.children == []:
        if node.leaf_lable == "cancer":
            return 'C'
        else: 
            return 'N'
    # the node isn't a leaf node and sending the sample down another layer
    if df.loc[sample, node.split] == 1:
        return vote(df, sample, node.children[0])
    else:
        return vote(df, sample, node.children[1])

     
# calculate the stats of the forest
# return the stats of the forest
def forest_stats(df, forest, oob):
    # sorrting all samples into the forest
    for tree in forest:
        for sample in oob:
            sort(df, sample, tree)

    tp, tn, fn, fp = 0, 0, 0, 0
    # voting the the samples
    for sample in oob:
        total_cancer_vote = 0
        total_no_cancer_vote = 0
        for tree in forest:
            tree_vote = vote(df, sample, tree)
            if tree_vote == 'C':
                total_cancer_vote += 1
            else:
                total_no_cancer_vote += 1
        
        if total_cancer_vote >= total_no_cancer_vote: # predict cancer
            if df.iloc[sample, 0].startswith("C"): # has cancer
                tp += 1
            else: # doesn't have caner
                fp += 1
        else: # predict no cancer
            if df.iloc[sample, 0].startswith("N"): # doesn't have cancer
                tn += 1
            else: # has cancer
                fn += 1
    stats_forest = evaluate(tp, fp, tn, fn)

    print(f"{'Metric':<20} {'Forest':<10}")
    for stat in stats_forest:
        print(f"{stat:<20} {stats_forest[stat]:<10.4f}")

def tree_stats(forest):

    used_mutation = {}
    # seeing the list of mutations used
    for tree in forest:
        used_mutation[tree.split] = used_mutation.get(tree.split, 0) + 1
        used_mutation[tree.children[0].split] = used_mutation.get(tree.children[0].split, 0) + 1
        used_mutation[tree.children[1].split] = used_mutation.get(tree.children[1].split, 0) + 1

    for mutation, count in used_mutation.items():
        print(f"Mutation: {mutation[:10]} Used {count} of times")

def main():
    f_stats = [0, 0, 0, 0]
    phi_stats = [0, 0, 0, 0]
    H_stats = [0, 0, 0, 0]

    random.seed(72)
    samples = []
    #loading the cvs into the df
    df = pd.read_csv('mutationu.csv')

    for i in range(len(df)):
        samples.append(i)

    bsd, oob = boot_strap_data(df)

    forest = build_forest(df, 10, bsd)

    forest_stats(df, forest, oob)

    print(f"Lenght of oob {len(oob)}")
    print(oob)
    tree_stats(forest)

    # root = Node("root", None)

    # root = forest_build_tree(df, root, samples, 2)
    # for sample in samples:
    #     sort(df, sample, root)
    # root.print_samples()
    # root.print_mutation()


if __name__ == "__main__":
    main()