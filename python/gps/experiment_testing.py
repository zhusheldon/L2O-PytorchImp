import logging
import imp
import os
import os.path
import sys
import argparse
import time
import numpy as np
import random
import pickle

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import gps as gps_globals
from gps.utility.display import Display
from gps.sample.sample_list import SampleList
from gps.algorithm.policy.tf_policy import TfPolicy
from gps.agent.lto.agent_lto import AgentLTO
from gps.proto.gps_pb2 import ACTION

def testComparisons(config):
    _hyperparams = config
    network_config=config['algorithm']['policy_opt']['network_params']
    network_config['def_obs']= config['algorithm']['policy_opt']['network_params']['obs_include']
    network_config['deg_action']= config['agent']['sensor_dims'][ACTION]
    policy = TfPolicy.load_policy(config['common']['policy_filename'], 
                                        config['algorithm']['policy_opt']['network_model'], 
                                        network_config=network_config)
    fcns = config['agent']['fcns']
    
    agent = config['agent']['type'](config['agent'])
    conditions = config['common']['test_conditions']
    range_test_idx = range(conditions)
    T = config['agent']['T']
    test_log = config['common']['test_log']
    
    for m in range(conditions):
        agent.sample(policy, m)
        
    traj_sample_lists = [agent.get_samples(cond) for cond in range_test_idx]
    all_locs = np.zeros((conditions,T))
    
    ##Iterate through each sample function
    for idx in range_test_idx:
        cur_locs = traj_sample_lists[idx].get_samples()[0].get(1)
        
        locs = np.zeros(len(cur_locs))
        new_fcn_text = 'Test sample %s objective function #%02d' % ('Logistic Regression', idx)
        append_output_text(test_log, new_fcn_text)
        ##Calculates the obj. value determined by L2O at each time step
        L2O_obj_val_text = ''
        for jdx in range(len(cur_locs)):
            locs[jdx] = agent.fcns[idx]['fcn_obj'].evaluate(np.array([cur_locs[jdx]]).reshape(cur_locs[jdx].size,1))
            L2O_obj_val_text +='%6.2f '% locs[jdx]
        all_locs[idx]= locs
        append_output_text(test_log, L2O_obj_val_text)
    # with open(self._log_filename, 'a') as f:
    #     f.write(text + '\n')

    agent.clear_samples()

def append_output_text(filename, text):
    with open(filename, 'a') as f:
        f.write(text + '\n')
    print(text)

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