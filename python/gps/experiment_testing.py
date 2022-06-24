from cmath import isnan
import logging
import imp
import os
import os.path
import sys
import argparse
import time
from datetime import datetime
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import gps as gps_globals
from gps.utility.display import Display
from gps.sample.sample_list import SampleList
from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy.lto.gd_policy import GradientDescentPolicy
from gps.algorithm.policy.lto.momentum_policy import MomentumPolicy
from gps.algorithm.policy.lto.cg_policy import ConjugateGradientPolicy
from gps.algorithm.policy.lto.lbfgs_policy import LBFGSPolicy
from gps.agent.lto.agent_lto import AgentLTO
from gps.proto.gps_pb2 import ACTION

FCN_FAMILY_NAME = 'Logistic Regression'
ALGS = 5

def testComparisons(config):
    _hyperparams = config
    network_config=config['algorithm']['policy_opt']['network_params']
    network_config['def_obs']= config['algorithm']['policy_opt']['network_params']['obs_include']
    network_config['deg_action']= config['agent']['sensor_dims'][ACTION]
    tf_policy = TfPolicy.load_policy(config['common']['policy_filename'], 
                                        config['algorithm']['policy_opt']['network_model'], 
                                        network_config=network_config)
    # fcns = config['agent']['fcns']
    
    agent = config['agent']['type'](config['agent'])

    conditions = config['common']['test_conditions']
    range_test_idx = range(conditions)
    T = config['agent']['T']
    test_log = config['common']['test_log']
    
    append_output_text(test_log, 'BEGINNING OF TESTING %s' % datetime.now())
    append_output_text(test_log, '')
    for cond in range(conditions):
        agent.sample(tf_policy, cond)
        
        grad_desc_policy = GradientDescentPolicy(agent, 
                                                config['algorithm']['policy_opt']['lr'],
                                                cond)
        agent.sample(grad_desc_policy, cond)
        
        momentum_policy = MomentumPolicy(agent, 
                                        config['algorithm']['policy_opt']['lr'],
                                        config['algorithm']['policy_opt']['momentum'],
                                        cond)
        agent.sample(momentum_policy, cond)
        
        cg_policy = ConjugateGradientPolicy(agent, 
                                            config['algorithm']['policy_opt']['lr'],
                                            cond)
        agent.sample(cg_policy, cond)
        
        lbfgs_policy = LBFGSPolicy(agent, 
                                    config['algorithm']['policy_opt']['lr'],
                                    config['algorithm']['policy_opt']['mem_len'],
                                    cond)
        agent.sample(lbfgs_policy, cond)
        
        
    traj_sample_lists = [agent.get_samples(cond) for cond in range_test_idx]
    tf_locs = np.zeros((conditions,T))
    gd_locs = np.zeros((conditions,T))
    momentum_locs = np.zeros((conditions,T))
    cg_locs = np.zeros((conditions,T))
    lbfgs_locs = np.zeros((conditions,T))

    
    ##Iterate through each sample function
    for idx in range_test_idx:
        new_fcn_text = 'Test %s objective function #%02d' % (FCN_FAMILY_NAME, idx)
        append_output_text(test_log, new_fcn_text)
        for alg_num in range(ALGS):
            cur_locs = traj_sample_lists[idx].get_samples()[alg_num].get(1)

            if alg_num==0:
                obj_val_text, tf_locs[idx] = gen_obj_val(agent, cur_locs, alg_num, idx)
            elif alg_num==1:
                obj_val_text, gd_locs[idx] = gen_obj_val(agent, cur_locs, alg_num, idx)
            elif alg_num==2:
                obj_val_text, momentum_locs[idx] = gen_obj_val(agent, cur_locs, alg_num, idx)
            elif alg_num==3:
                obj_val_text, locs = gen_obj_val(agent, cur_locs, alg_num, idx)
                
                # if locs[0]<locs[-1]:
                #     locs[-1]=np.NaN
                cg_locs[idx] = locs
            elif alg_num==4:
                obj_val_text, lbfgs_locs[idx] = gen_obj_val(agent, cur_locs, alg_num, idx)
            else:
                obj_val_text = 'Invalid alg_num!'
            append_output_text(test_log, obj_val_text)
        append_output_text(test_log, '')
    append_output_text(test_log, 'END OF TESTING')
    append_output_text(test_log, '-'* 100)
    append_output_text(test_log, '')
    tf_mean_obj_val = np.mean(tf_locs, axis=0)
    gd_mean_obj_val = np.mean(gd_locs, axis=0)
    momentum_mean_obj_val = np.mean(momentum_locs, axis=0)
    cg_mean_obj_val = np.mean(cg_locs[cg_locs[:,1]>=cg_locs[:,-1]],axis=0)
    lbfgs_mean_obj_val = np.mean(lbfgs_locs, axis=0) 
    
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(tf_mean_obj_val, label='Autonomous Optimizer')
    plt.plot(gd_mean_obj_val, label='Gradient Descent')
    plt.plot(momentum_mean_obj_val, label='Momentum')
    plt.plot(cg_mean_obj_val, label='Conjugate Gradient')
    plt.plot(lbfgs_mean_obj_val, label='L-BFGS')
    plt.xlabel('Iteration')
    plt.ylabel('Average Objective Value')
    plt.legend()
    plt.show()

    agent.clear_samples()

def gen_obj_val(agent, cur_locs, alg_num, idx):
    ##Calculate the obj. value determined by L2O at each time step
    if alg_num==0:
        obj_val_text = '%20s: ' % 'L2O'
    elif alg_num==1:
        obj_val_text = '%20s: ' % 'Gradient Descent'
    elif alg_num==2:
        obj_val_text = '%20s: ' % 'Momentum'
    elif alg_num==3:
        obj_val_text = '%20s: ' % 'Conjugate Gradient' 
    elif alg_num==4:
        obj_val_text = '%20s: ' % 'L-BFGS'
    locs = np.zeros(len(cur_locs))
    for jdx in range(len(cur_locs)):
        locs[jdx] = agent.fcns[idx]['fcn_obj'].evaluate(np.array([cur_locs[jdx]]).reshape(cur_locs[jdx].size,1))
        obj_val_text +='%6.6f '% locs[jdx]
    return obj_val_text, locs

def append_output_text(filename, text):
    with open(filename, 'a') as f:
        f.write(text + '\n')

def main():
    parser = argparse.ArgumentParser(description='Test the RL Agent.')
    parser.add_argument('experiment', type=str, help='experiment name')
    args = parser.parse_args()

    exp_name = args.experiment

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '\\'.join(str.split(gps_filepath, '\\')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'
    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" % (exp_name, hyperparams_file))
    
    # May be used by hyperparams.py to load different conditions
    gps_globals.phase = "TRAIN"
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    seed = hyperparams.config.get('random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    
    testComparisons(hyperparams.config)           
                    
if __name__ == "__main__":
    main()