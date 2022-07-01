import sys
import os
import numpy as np
import tensorflow as tf
import pickle as pickle
from time import time
import torch

def printWithoutNewline(s):
   sys.stdout.write(s)
   sys.stdout.flush() 

# A FcnFamily is a function template with unrealized placeholders (e.g. coefficients)
# A Fcn is a member of a FcnFamily with actual values substituted in for the placeholders

# For input to the functions "evaluate", "grad", "hess", x can be a list of variables, but each variable must be an N x 1 vector
# fcn must be a function that takes two arguments, x and params. x is a list of variables, and params is a dict, with the keys corresponding to names of placeholders and values being the substituted values
class FcnFamily(object):
    
    # params is a dict whose entries are (name, type)
    # hyperparams is a dict and must be the SAME as the parameters passed into the constructor of the child class - it is used for pickling
    # Options can be passed in as extra keyword arguments. Available options: disabled_hess, session, start_session_manually, gpu_id, tensor_prefix
    # Options that are for internal use only: graph_def and tensor_names - these are used when unpickling
    def __init__(self, fcn, num_dims, params, hyperparams, **kwargs):
        self.num_dims = num_dims
        self.fcn_defns = fcn
        self.param_defns = params
        self.hyperparams = hyperparams
        self.options = kwargs
        self.params = {}
        self.is_param_subsampled = {} 
        
    
    def assign_param_vals_(self, param_vals):
        placeholder_vals = {}
        for key in self.params:
            placeholder_vals[self.params[key]] = param_vals[key]
        return placeholder_vals
    
    def evaluate(self, x, param_vals):
        val = self.fcn_defns(x,param_vals)
        return [val.numpy()]
    
    def grad(self, x, param_vals):
        new_x = []
        vals = np.array([])
        for arr in x:
            new_x.append(tf.convert_to_tensor(arr))
        with tf.GradientTape() as g:
            g.watch(new_x)
            y = self.fcn_defns(new_x, param_vals)
        g = g.gradient(y, new_x)
        for i in range(len(g)):
            vals= np.append(vals, g[i].numpy(), axis=0) if vals.size!=0 else g[i].numpy()
        return [vals]
    
    # Returns a list of lists, with vals[i][j] containing the second derivative wrt self.x_[i] and self.x_[j]
    def hess(self, x, param_vals):
        assert ("disable_hess" not in self.options) or (not self.options["disable_hess"]), "Hessian is disabled. "
        
        # placeholder_vals = {self.x_[i]: x[i] for i in range(len(self.x_))}
        # placeholder_vals.update(self.assign_param_vals_(param_vals))
        # with tf.device(self.device_string):
        #     flattened_vals = self.session.run([hess_elem for hess_list in self.hess_ for hess_elem in hess_list], placeholder_vals)
        # vals = []
        # j = 0
        # for i in range(len(self.x_)):
        #     vals.append(flattened_vals[j:j+(i+1)])
        #     vals[-1].extend([None] * (len(self.x_)-i-1))
        #     j += (i+1)
        # # Fill in the upper triangle of the Hessian by taking advantage of the symmetry of the Hessian
        # for i in range(1,len(self.x_)):
        #     for j in range(i):
        #         vals[j][i] = vals[i][j].T
        
        x = [tf.convert_to_tensor(np.reshape(x, (-1, 1)))]
        with tf.GradientTape() as gg:
            gg.watch(x)
            with tf.GradientTape() as g:
                g.watch(x)
                y = self.fcn_defns(x, param_vals)
            dy_dx = g.jacobian(y, x)
        gg = gg.jacobian(dy_dx, x)
        return gg
    
    def get_total_num_dim(self):
        total_num_dim = 0
        for num_dim in self.num_dims:
            total_num_dim += num_dim
        return total_num_dim
    
    def destroy(self):
        pass
    #     if self.session is not None:
    #         self.session.close()
    #         self.session = None
    
    # For pickling
    def __getstate__(self):
        tf.io.write_graph(self.session.graph_def, "/tmp", "tf_graph.pb", False) #proto
        with open("/tmp/tf_graph.pb", "rb") as f:
            graph_def_str = f.read()
        os.remove("/tmp/tf_graph.pb")
        
        tensor_names = dict()
        tensor_names["params"] = {param_name: self.params[param_name].name for param_name in self.params}
        tensor_names["x"] = [cur_x.name for cur_x in self.x_]
        tensor_names["fcn"] = self.fcn_.name
        tensor_names["grad"] = [cur_grad.name for cur_grad in self.grad_]
        
        if ("disable_hess" not in self.options) or (not self.options["disable_hess"]):
            tensor_names["hess"] = [[cur_hess_block.name for cur_hess_block in cur_hess_block_row] for cur_hess_block_row in self.hess_]
        
        return {"hyperparams": self.hyperparams, "options": {option_name: self.options[option_name] for option_name in self.options if option_name not in ["session", "graph_def", "start_session_manually"]}, "graph_def": graph_def_str, "tensor_names": tensor_names}

    # For unpickling
    def __setstate__(self, state):
        kwargs = state["hyperparams"].copy()
        kwargs.update(state["options"])
        kwargs["graph_def"] = state["graph_def"]
        kwargs["tensor_names"] = state["tensor_names"]
        kwargs["start_session_manually"] = True
        self.__init__(**kwargs)
  
class Fcn(object):

    # If disable_subsampling is set to True, will never subsample regardless of what batch_size is set to be, either in the constructor or in Fcn.new_sample()
    def __init__(self, family, param_vals, batch_size = "all", disable_subsampling = False):
        self.family = family
        self.param_vals = param_vals
        self.batch_size = batch_size
        self.disable_subsampling = (disable_subsampling or all(not self.family.is_param_subsampled[key] for key in self.family.params))
        # If batch size is "all" or no params are subsampled, don't require calling Fcn.new_sample() before calling Fcn.evaluate/grad/hess. 
        if self.disable_subsampling or batch_size == "all":
            self.subsampled_param_vals = self.param_vals
        else:
            self.subsampled_param_vals = None
                   
    # If self.disable_subsampling is True, this is a no-op. 
    # If batch_size is set, temporarily overrides self.batch_size
    # By setting batch_size to "all", can temporarily disable subsampling
    def new_sample(self, batch_size = None):
        if not self.disable_subsampling:
            if batch_size is None:
                batch_size = self.batch_size
            if batch_size != "all":
                subsampled_idx = None       # Same sampled indices are used for all params to preserve correspondence between individual entries (i.e. one row of data corresponds to one element of label)
                self.subsampled_param_vals = {}
                for key in self.family.params:
                    if not self.family.is_param_subsampled[key] or batch_size >= self.param_vals[key].shape[0]:
                        self.subsampled_param_vals[key] = self.param_vals[key]
                    else:
                        if subsampled_idx is None:
                            subsampled_idx = np.random.permutation(self.param_vals[key].shape[0])[:batch_size]
                        self.subsampled_param_vals[key] = self.param_vals[key][subsampled_idx]
            else:
                self.subsampled_param_vals = self.param_vals
    
    def evaluate(self, x):
        assert self.subsampled_param_vals is not None, "Fcn.new_sample() must be called first. "
        return self.family.evaluate(x, self.subsampled_param_vals)
    
    def grad(self, x):
        assert self.subsampled_param_vals is not None, "Fcn.new_sample() must be called first. "
        return self.family.grad(x, self.subsampled_param_vals)
    
    def hess(self, x):
        assert self.subsampled_param_vals is not None, "Fcn.new_sample() must be called first. "
        return self.family.hess(x, self.subsampled_param_vals)
    
class QuadFormFcnFamily(FcnFamily):
    
    def __init__(self, num_dim, **kwargs):
        def fcn(x, params):
            return tf.matmul(x[0], tf.matmul(params["A"], x[0]), transpose_a=True)
        
        FcnFamily.__init__(self, fcn, [num_dim], {"A": {"type": tf.float64}}, {"num_dim": num_dim}, **kwargs)
    
class QuadFormFcn(Fcn):
    
    def __init__(self, family, A, *args, **kwargs):
        Fcn.__init__(self, family, {"A": A}, *args, **kwargs)
    
    def evaluate(self, x):
        return Fcn.evaluate(self, [x])
    
    def grad(self, x):
        return Fcn.grad(self, [x])[0]
    
    def hess(self, x):
        return Fcn.hess(self, [x])[0][0]
    
class LogisticRegressionFcnFamily(FcnFamily):
    
    def __init__(self, dim, **kwargs):
        # params["data"] is N x dim, params["labels"] is N x 1
        # params["sigma_sq"] is 1 x 1 and represents the squared of the sigma parameter in the GM estimator
        # The larger sigma is, the larger the non-saturating range
        def fcn(x, params):
            weights = tf.slice(x[0], [0,0], [dim,-1])  # dim x 1 matrix
            bias = tf.slice(x[0], [dim,0], [1,-1])   # 1 x 1 matrix
            
            preds = tf.matmul(params["data"], weights) + bias     # N x 1 matrix, where N is the number of data points
            
            loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(preds, params["labels"]))
            
            # L2 regularization for the fully connected parameters.
            regularizers = tf.nn.l2_loss(weights)
            # Add the regularization term to the loss.
            loss += 5e-4 * regularizers
            
            return loss
        
        FcnFamily.__init__(self, fcn, [dim+1], {"data": {"type": tf.float64, "subsampled": True}, "labels": {"type": tf.float64, "subsampled": True}}, {"dim": dim}, **kwargs)

class LogisticRegressionFcn(Fcn):
    
    def __init__(self, family, data, labels, *args, **kwargs):
        Fcn.__init__(self, family, {"data": data, "labels": labels}, *args, **kwargs)
    
    def evaluate(self, x):
        return Fcn.evaluate(self, [x])
    
    def grad(self, x):
        return Fcn.grad(self, [x])[0]
    
    def hess(self, x):
        return Fcn.hess(self, [x])[0][0]
    
class LogisticRegressionWithoutBiasFcnFamily(FcnFamily):
    
    def __init__(self, dim, **kwargs):
        # params["data"] is N x dim, params["labels"] is N x 1
        def fcn(x, params):
            weights = x[0]
            
            preds = tf.matmul(params["data"], weights)     # N x 1 matrix, where N is the number of data points
            
            loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(preds, params["labels"]))
            
            # L2 regularization for the fully connected parameters.
            regularizers = tf.nn.l2_loss(weights)
            # Add the regularization term to the loss.
            loss += 5e-4 * regularizers
            
            return loss
        
        FcnFamily.__init__(self, fcn, [dim], {"data": {"type": tf.float64, "subsampled": True}, "labels": {"type": tf.float64, "subsampled": True}}, {"dim": dim}, **kwargs)

class LogisticRegressionWithoutBiasFcn(Fcn):
    
    def __init__(self, family, data, labels, *args, **kwargs):
        Fcn.__init__(self, family, {"data": data, "labels": labels}, *args, **kwargs)
    
    def evaluate(self, x):
        return Fcn.evaluate(self, [x])
    
    def grad(self, x):
        return Fcn.grad(self, [x])[0]
    
    def hess(self, x):
        return Fcn.hess(self, [x])[0][0]
    
# Robust linear regresison using Geman-McLure (GM) estimator
class RobustRegressionFcnFamily(FcnFamily):
    
    def __init__(self, dim, **kwargs):
        # params["data"] is N x dim, params["labels"] is N x 1
        # params["sigma_sq"] is 1 x 1 and represents the squared of the sigma parameter in the GM estimator
        # The larger sigma is, the larger the non-saturating range
        def fcn(x, params):
            weights = tf.slice(x[0], [0,0], [dim,-1])  # dim x 1 matrix
            bias = tf.slice(x[0], [dim,0], [1,-1])   # 1 x 1 matrix
            
            preds = tf.matmul(params["data"], weights) + bias     # N x 1 matrix, where N is the number of data points
            err = params["labels"] - preds
            err_sq = tf.square(err)
            loss = tf.reduce_mean(input_tensor=tf.truediv(err_sq, tf.add(err_sq, params["sigma_sq"])))
            
            return loss
        
        FcnFamily.__init__(self, fcn, [dim+1], {"data": {"type": tf.float64, "subsampled": True}, "labels": {"type": tf.float64, "subsampled": True}, "sigma_sq": {"type": tf.float64}}, {"dim": dim}, **kwargs)

class RobustRegressionFcn(Fcn):
    
    def __init__(self, family, data, labels, sigma_sq, *args, **kwargs):
        Fcn.__init__(self, family, {"data": data, "labels": labels, "sigma_sq": sigma_sq}, *args, **kwargs)
    
    def evaluate(self, x):
        return Fcn.evaluate(self, [x])
    
    def grad(self, x):
        return Fcn.grad(self, [x])[0]
    
    def hess(self, x):
        return Fcn.hess(self, [x])[0][0]
    
class NeuralNetFcnFamily(FcnFamily):
    
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim]
        
        dims = [input_dim] + hidden_dim + [output_dim]
        
        def fcn(x, params):
            weights = []
            biases = []
            for i in range(len(dims)-1):
                weights.append(tf.reshape(x[2*i], [dims[i], dims[i+1]]))
                biases.append(tf.reshape(x[2*i+1], [1, dims[i+1]]))
            
            cur_layer = params["data"]
            for i in range(len(dims)-1):
                if i == len(dims)-2:
                    cur_layer = tf.matmul(cur_layer, weights[i]) + biases[i]
                else:
                    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weights[i]) + biases[i])
                
            output = cur_layer
                        
            loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=output, labels=params["labels"]))
            
            # L2 regularization for the fully connected parameters.
            regularizers = tf.nn.l2_loss(weights[0])
            for i in range(1,len(dims)-1):
                regularizers += tf.nn.l2_loss(weights[i])
            # Add the regularization term to the loss.
            loss += params["l2_weight"] * regularizers
            
            return loss
        
        param_sizes = []
        for i in range(len(dims)-1):
            param_sizes.append(dims[i]*dims[i+1])
            param_sizes.append(dims[i+1])
        
        FcnFamily.__init__(self, fcn, param_sizes, {"data": {"type": tf.float64, "subsampled": True}, "labels": {"type": tf.int64, "subsampled": True}, "l2_weight": {"type": tf.float64}}, {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim}, **kwargs)

class NeuralNetFcn(Fcn):
    
    # labels is an N x 1 array, where N is the batch size
    def __init__(self, family, data, labels, l2_weight = 5e-4, *args, **kwargs):
        assert(labels.shape[1] == 1)
        Fcn.__init__(self, family, {"data": data, "labels": labels[:,0], "l2_weight": l2_weight}, *args, **kwargs)
    
    def unpack_x(self, x):
        unpacked_x = []
        prev_dim = 0
        for num_dim in self.family.num_dims:
            unpacked_x.append(x[prev_dim:prev_dim+num_dim,:])
            prev_dim += num_dim
        return unpacked_x
    
    def evaluate(self, x):
        return Fcn.evaluate(self, self.unpack_x(x))
    
    def grad(self, x):
        return np.vstack(Fcn.grad(self, self.unpack_x(x)))
    
    def hess(self, x):
        return np.vstack([np.hstack(block_row) for block_row in Fcn.hess(self, self.unpack_x(x))])
    
def main(*args):
    
    family = QuadFormFcnFamily(2)
    fcn = QuadFormFcn(family, np.array([[2., 1.], [1., 2.]]))
    print(type(fcn.evaluate(np.array([[-1.],[2.]]))))
    print(type(fcn.grad(np.array([[-1.],[2.]]))))
    assert fcn.evaluate(np.array([[-1.],[2.]]))==np.array([[6.]])
    assert np.array_equal(fcn.grad(np.array([[-1.],[2.]])), np.array([[0.], [6.]]))
    assert fcn.grad(np.array([[-1.],[2.]])).size==2
    # print(fcn.hess(np.array([[-1.],[2.]]))) ##Hess isn't used anywhere in the code, don't need as of 6/29
    family.destroy()
    
    np.random.seed(0)
    input_dim = 5
    hidden_dim = [5]
    output_dim = 5
    num_examples = 10
    family = NeuralNetFcnFamily(input_dim,hidden_dim,output_dim)
    data = np.random.randn(num_examples,input_dim)
    labels = np.random.randint(output_dim,size=(num_examples,1))
    fcn = NeuralNetFcn(family, data, labels)
    weights1 = np.random.randn(input_dim*hidden_dim[0],1)
    biases1 = np.random.randn(hidden_dim[0],1)
    weights2 = np.random.randn(hidden_dim[0]*output_dim,1)
    biases2 = np.random.randn(output_dim,1)
    x = np.vstack((weights1,biases1,weights2,biases2))
    print(x.shape)
    print("Dimensionality: %d" % (x.shape[0]))
    print(fcn.evaluate(x))
    print(fcn.grad(x))
    assert fcn.grad(x).size == (input_dim*hidden_dim[0]+hidden_dim[0]+hidden_dim[0]*output_dim+output_dim)
    
    family.destroy()
    
if __name__ == '__main__':
    main(*sys.argv[1:])
