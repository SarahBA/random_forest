
import numpy as np
from scipy import stats
import os


# 0= NUMERICAL ;;; 1= CATEGORICAL





def import_datasets(name_data, data):
    if name_data == 'glass.txt':
        dataset = data[:, 1:-1]
        [num_samples, num_features] = dataset.shape
        class_v = data[:,-1]
        cat = 1
        return dataset, num_samples, num_features, class_v, cat

#   # d = dataset
# 	# n = n_samples
# 	# m = n_features
#
# 	# categorical variables should be indexes (0,1,2,3 not "R" etc.) remember mapping
# 	# type should be 0=numeric, 1=categorical
# 	# output type 0=regression, 1=categorical





def readdata(datapath):
    '''
    Data are separated with ','
    Read the data into the a 'matrix' which is ndarray in python
    indexing in this way
    data[1,2], data[:,-1] or data[1,:]

    Sometimes, the first column is the index of the data
    Somtimes, the last column is the class of the data, or the first column
    :param datapath: the path of the data
    :return: 'data' is ndarray
    '''
    data_l = []
    with open(datapath,"r") as f:
        mylist = f.read().splitlines()
        for line in mylist:
            currentline = line.split(",")
            data_l.append(currentline)
    data = np.array(data_l)
    return data

def get_paths(data_name):
    '''
    The directory system:
    The project is in the file named "project"
    It has "src", which save all the source codes, and "data", which has all the training data.

    Get the project path and training data path
    :param data_name: the name of the test data
    :return: training data path
    '''
    src_path = os.getcwd()  # get the directory of src
    project_path = os.path.dirname(src_path)  # get the parent directory of src, that is project
    data_path = os.path.join(project_path, 'data')  # get the directory of data, which should be in the project
    training_data_dir = os.path.join(data_path, data_name)  # now we are in the folder containing the training data
    return training_data_dir

def demo():
    name_data = 'glass.txt'
    datapath = get_paths(name_data)  # give the name of the data file, return the accessible path
    data = readdata(datapath)
    dataset, num_samples, num_features, class_v, cat = import_datasets(name_data, data)

    print dataset, num_samples, num_features, class_v, cat



class random_forest: #{
	def __init__(number_of_trees, F, min_leaf_size = 1, f_num = VR, f_cat = IG): #{
		
	def add_data(#n x m feature data, m variable/feature type, n output, training_split = 0.1)


	#}





#}


class tree: #{ #UNDER CONSTRUCTION
	def __init__(self, data, data_type, y, y_type, n_classes, min_leaf_size = 1,f_num = "VR", f_cat = "IG"):
		self.data = data;
		self.data_type = data_type;
		self.y = y;
		self.y_type = y_type;
		self.n_classes = n_classes;
		self.min_leaf_size = min_leaf_size;
		self.options = {"VR" : self.VR, "IG" : self.IG, "GINI" : self.GINI, "TEST" : self.TEST}
		self.f_num = self.options[f_num];
		self.f_cat = self.options[f_cat];
		
		split_feature, split_number, data_left_idx, data_right_idx = self.find_split(np.array(list(range(len(data)))));
		
		self.root = node(data_type[split_feature]);
		self.root.split_value = split_number;
		self.root.split_feature = split_feature;
		self.grow_tree(self.root, data_left_idx, data_right_idx);
		
		#~ init();
				
		
		
	def init(self,f_num = "VR", f_cat = "IG"):		
			self.f_num = self.options[f_num];
			self.f_cat = self.options[f_cat];
		
		
	def predict(self, data_point):
		return self.root.predict(data_point);

	# TODO: VISUALIZATION??
	def visualize(self, bounds):
		#return a list of pairs of points
		#each pair defines a seperation line
		#bounds are the bounds for plotting, should be same as plt.axis(bounds)
		return 0;
	
		
	def grow_tree(self, root, data_left_idx, data_right_idx): #{
		uq_left = np.unique(self.y[data_left_idx])
		uq_right = np.unique(self.y[data_right_idx])
		left_leaf = False;
		right_leaf = False;
		
		
		if(len(data_left_idx) <= self.min_leaf_size):
			#create leaf node
			node_left = node(self.y_type);
			if(self.y_type == 0):
				node_left.value = np.mean(self.y[data_left_idx]);
			else:
				node_left.value = stats.mode(self.y[data_left_idx])[0][0];
			root.left = node_left;
			left_leaf = True;
			
		else:
			if(len(uq_left) <= 1):
				node_left = node(self.y_type);
				node_left.value = uq_left[0];
				root.left = node_left;
				left_leaf = True;
				
				
			
		if(len(data_right_idx) <= self.min_leaf_size):
			#create leaf node
			node_right = node(self.y_type);
			if(self.y_type == 0):
				node_right.value = np.mean(self.y[data_right_idx]);
			else:
				node_right.value = stats.mode(self.y[data_right_idx])[0][0];
			root.right = node_right;
			right_leaf = True;
			
		else:
			if(len(uq_right) <= 1):
				node_right = node(self.y_type);
				node_right.value = uq_right[0];
				root.right = node_right;
				right_leaf = True;
		
		
		if(not right_leaf): #{
			split_feature, split_number, data_left_idx1, data_right_idx1 = self.find_split(data_right_idx);
			node_right = node(self.n_classes[split_feature]);
			node_right.split_feature = split_feature;
			node_right.split_value = split_number;
			
			root.right = node_right;
			self.grow_tree(root.right,data_left_idx1,data_right_idx1);		
		#}
		
		if(not left_leaf): #{
			split_feature, split_number, data_left_idx1, data_right_idx1 = self.find_split(data_left_idx);
			node_left = node(self.n_classes[split_feature]);
			node_left.split_feature = split_feature;
			node_left.split_value = split_number;
			
			root.left = node_left;
			self.grow_tree(root.left,data_left_idx1,data_right_idx1);		
		#}		
	#}
		
	
	
	def find_split(self, data_idxs): #{
		best_feature = -1;
		best_split_number = -1;
		best_value = -999999;
		best_data_left_idx = -1;
		best_data_right_idx = -1;
		
		
		for feature in range(self.data.shape[1]): #{			
			if(self.data_type[feature] == 0): #{				
				order = np.argsort(self.data[data_idxs,feature]);
				sorted_data_idxs = data_idxs[order];
				uq_values, uq_idx = np.unique(self.data[sorted_data_idxs,feature],return_index = True);
				for split in range(len(uq_idx)-1): #{
					split_number = (uq_values[split] + uq_values[split+1])/2
					data_left_idx = data_idxs[self.data[data_idxs,feature] < split_number]
					data_right_idx = data_idxs[self.data[data_idxs,feature] >= split_number]
					
					value = self.f_num(data_left_idx,data_right_idx);
					
					if(value > best_value):
						best_feature = feature;
						best_split_number = split_number;
						best_value = value;
						best_data_left_idx = data_left_idx;
						best_data_right_idx = data_right_idx;
						

				#}
			#}	
			else: #{
				for split in range(self.n_classes[feature]): #{	
					idx_left = data_idxs[self.data[data_idxs,feature] != split]
					idx_right = data_idxs[self.data[data_idxs,feature] == split]
					
					value = self.f_cat(idx_left,idx_right);
					
					if(value > best_value):
						best_feature = feature;
						best_split_number = split;
						best_value = value;
						best_data_left_idx = idx_left;
						best_data_right_idx = idx_right;
						
					
				#}
				
			#}
		#}
		
		
		return best_feature,best_split_number,best_data_left_idx, best_data_right_idx
	#}
	
	
	
	# input is indexes over data
	# these are member functions, so you can use self.data
	# should return a float
	def IG(self,data_left_idx,data_right_idx):
		return 0
	def GINI(self,data_left_idx,data_right_idx):
		return 0
	def VR(self,data_left_idx,data_right_idx):
		return 0
	def TEST(self,data_left_idx,data_right_idx):
		return -(np.var(self.y[data_left_idx]) + np.var(self.y[data_right_idx]))
	
	
#}
	
	
	
	
	
class node: #{ 
	def __init__(self,is_num):
		self.left = None;
		self.right = None;
		self.is_num = is_num;
		self.split_feature = None;
		self.split_value = None;
		self.value = None;

	def predict(self, data_point): #{
		if(self.is_leaf()): #{
			return self.value;
		else:
			if(self.compare(data_point)):
				return self.right.predict(data_point);
			else:
				return self.left.predict(data_point);
		#}
	#}	
	
	def is_leaf(self):
		return (self.left == None and self.right == None)


	def compare(self, data_point):
		if(is_num):
			return data_point[self.split_feature] > self.split_value;
		else:
			return data_point[self.split_feature] == self.split_value;
		
		
#}













