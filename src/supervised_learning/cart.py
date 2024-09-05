import numpy as np

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        self.value = value
        
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2, info_gain_method="entropy"):
        self.root = None
        
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.info_gain_method = info_gain_method
        
    def build_tree(self, dataset, curr_depth=0):
        
        X, y = dataset[:,:-1], dataset[:,-1]
        n_samples, n_features = np.shape(X)
        
        if n_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            best_split = self.get_best_split(dataset, n_samples, n_features)
            if best_split["info_gain"]>0:
                left_child = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_child = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_child, right_child, best_split["info_gain"])
        
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, n_samples, n_features):
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in range(n_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y, self.info_gain_method)
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if(mode=="gini"):
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        elif(mode=="entropy"):
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        else:
            raise ValueError ("Invalid information gain method, should either be: gini or entropy.")
        return gain
    
    def entropy(self, y):
        labels = np.unique(y)
        entropy = 0
        for label in labels:
            p_label = len(y[y == label]) / len(y)
            entropy += -p_label * np.log2(p_label)
        return entropy
    
    def gini_index(self, y):
        labels = np.unique(y)
        gini = 0
        for label in labels:
            p_label = len(y[y == label]) / len(y)
            gini += p_label**2
        return 1 - gini
        
    def calculate_leaf_value(self, y):
        y = list(y)
        return max(y, key=y.count)
    
    def show_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sLeft:" % (indent), end="")
            self.show_tree(tree.left, indent + indent)
            print("%sRight:" % (indent), end="")
            self.show_tree(tree.right, indent + indent)
    
    def fit(self, X, y):
        X = X.values
        y = y.values.reshape(-1, 1)
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        X = X.values
        predictions = [self._predict(x, self.root) for x in X]
        return predictions
    
    def _predict(self, X, tree):
        if tree.value!=None:
            return tree.value
        feature_val = X[tree.feature_index]
        if feature_val<=tree.threshold:
            return self._predict(X, tree.left)
        else:
            return self._predict(X, tree.right)