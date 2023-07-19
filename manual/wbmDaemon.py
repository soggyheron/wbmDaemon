#!/usr/bin/env python3
'''
wbmDaemon.py
:author: Shan Zuidema, WSAG/ESRC/EOS/UNH 
:copyright: 2021
----------------------------------------------------------------
Defines and runs model simulations, and pools of simulations.
  Intention is to include residual calculation as needed as well.
Developed for use with the UNH Water Balance Model (WBM)
-----------------------------------------------------------------
:version: v0.0.1 (DRAFT DEVELOPMENT, 2nd project application)
:license:  GPLv3 (TARGET, NOT PRESENTLY DEFINED)
'''

import os
import glob
import pyparsing as pp
from collections import OrderedDict as odict
import pandas as pd
import numpy as np
import xarray as xr
from copy import copy
from shutil import copy2,rmtree
import subprocess
from threading import Thread
from queue import Queue
from pathos import multiprocessing as mp

# these are some functions for manipulating init files poorly.

def parse_init_hash(hash_string, target):
    #  I don't think this handles new lines in the Output_vars list...
    parsed = pp.nestedExpr('{', '}').parseString(hash_string)
    parsed = parsed[0].asList()  # if len(parsed) >= 2 else parsed
    for i, entry in enumerate(parsed):
        if entry == target:
            break
    else:
        raise IndexError("parameter {} not found in hash_string".format(target))
    assert parsed[i + 1] == "=>", "Assignment operator not where expected"

    def renest(block):
        def handle(x): return x if type(x) == str else renest(x)
        output = "{ "
        output += " ".join([handle(x) for x in block])
        output += " }"
        return output
    if type(parsed[i + 2]) == list:
        output = renest(parsed[i + 2])
    else:
        output = parsed[i + 2]
    return output


def traverse_nested_hash(hash_string, target):
    while len(target) > 0:
        t = target.pop(0)
        hash_string = parse_init_hash(hash_string, t)
    return hash_string

class WBM_Simulator_Factory(object):
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs
    def __call__(self,name):
        return WBM_Simulator(name,*self.args,**self.kwargs)

class WBM_Simulator(object):
    """
    Sets up a common simulation, parameters to vary, and
      launches a model when parameters are provided.
    """

    def __init__(self, name, runit_template, parameter_odict, run_flags=None, model_class="monthly",
                 observations=None,wd=None, model="./wbm.pl", results_archive=None,
                 proc_variables=None,actually_run=True,build_spool=False):
        assert os.path.exists(
            runit_template), "WBM Run Init file %s does not exist" % runit_template
        assert type(parameter_odict), "Parameter data must be in ordered dict!"
        if wd is not None:
            assert os.path.exists(
                wd), "WBM Simulator specified a working directory that does not exist!"
            self.wd = wd
        else:
            self.wd = os.getcwd()
        assert os.path.exists(
            self.wd + "/WBM.conf"), "WBM Simulator specified in a working directory that does not contain a WBM.conf configuration file!" # not strictly necessary anymore

        if results_archive == None:
            if not os.path.exists(self.wd + "/archive/"):
                os.mkdir(self.wd + "/archive/")
            results_archive = self.wd + "/archive/"
        self.results_archive = results_archive

        self.runit = open(runit_template, 'r').read()
        self.parameters = parameter_odict
        self._define_flags(run_flags)
        self.parameter_keys = self._ensure_params()
        self.name = name
        self.output_dir = traverse_nested_hash(
            self.runit, "MT_Code_Name__Output_dir".split('__'))
        self.model = model
        if proc_variables == None:
            self.proc_variables = self._get_output_variables()
        else:
            self.proc_variables = proc_variables
        if model_class == "sample":
            assert observations is not None, "Model class 'Sample' requires point observation dict"
            if type(observations) == str:
                # assuming this is a path to a csv with the necessary structure
                # index datetime, columns = [variable, site] - nested columns to conform to WBM sample outer_mean
                observations = pd.read_csv(observations,index_col=0,parse_dates=True,header=[0,1])
                # need start date and end-date of simulation to pre-compute the observation record
#                print(observations.index)
                start= traverse_nested_hash(self.runit, "MT_Code_Name__Run_Start".split('__'))
                end= traverse_nested_hash(self.runit, "MT_Code_Name__Run_End".split('__'))
                epoch=pd.date_range(start=start,end=end,freq='d')
                observations=observations.loc[epoch].dropna()
                print(observations)
                print(start,end)
            elif type(observations) == pd.DataFrame:
                # Check for conformation of appropriate format here?
                pass
            else:
                raise RuntimeError( "observation type {} is not accepted in this context.".format(repr(type(observations))))
            self.processor = WBM_sample(observations)
        elif model_class == "monthly":
            if observations is not None:
                print("Need to define a WBM_Observations object, assert that this is one")
                print(" !!! OBSERVATIONS NOT YET IMPLEMENTED !!! ")
                print("     presently just copying key output to archive")
            self.processor = WBM_monthly(self.proc_variables)
        else:
            raise RuntimeError("model_class: {} is not understood".format(model_class))
        self.actually_run = actually_run
        self.build_spool = build_spool
        
    def _get_output_variables(self):
        var_list = traverse_nested_hash(
            self.runit, ["Output_vars"]).strip('"').strip("'").split()
        return var_list

    def _ensure_params(self):
        param_search_tokens = odict()
        for k, v in self.parameters.items():
            param = k.split('__')
            param_search_tokens[k] = traverse_nested_hash(self.runit, param).strip(
                '"').strip("'")  # This should fail if the value isn't already a float
        return param_search_tokens

    def _define_flags(self, flags):
        known_flags = ["spoolDir", "v", "rm"]
        self.flags = {}
        if flags is not None:
            for k, v in flags.items():
                if k in known_flags:
                    self.flags[k] = v
                else:
                    print(
                        "  WARNING! User specified run flag {} => {} not recognized!".format(k, v))

    def __call__(self, X,run_id=None):
        if type(X) in [float,np.float64]:  # If there is only one parameter
            X = np.array([X])
        assert X.shape[0] == len(
                self.parameters), "Parameter vector wrong size for simulator!"
        # now we are making a new simulator for each with a name given by the pid
        this_run=self.prepare_init(X,run_id)
        self.call_model(this_run)
        if False:  # True if this is a Spotpy run ...
            self.process_results()
            results = self.processor.sim
            rmtree(self.output_dir_now)  # clean up output directory.  
            os.remove("{}.init".format(this_run))
            return results

    def check_run_id(self, run_id):
        if os.path.exists(self.results_archive + run_id + "/"):
            raise IOError("wbmDaemon: run_id exists in run archive!")

    def prepare_init(self, X,run_id=None):
        new_runit = copy(self.runit)
        modelToken = traverse_nested_hash(
            new_runit, "ID".split('__')).strip('"').strip("'")
        print(modelToken,flush=True)
        run_name=self.name+run_id
        # shouldn't be more than this ... but keep in  mind
        new_runit = new_runit.replace(modelToken, run_name, 3)
        self.output_dir_now = self.output_dir.strip(
            "'").strip('"').replace(modelToken, run_name, 1)
        # Make sure the processor knows where to look for output
        self.processor.path=self.output_dir_now
        # Are there dependencies between parameters that need to be accounted for?
        #  thinking of the min and max range for infiltration fraction or sdp constant
        for x, (k, token) in zip(X, self.parameter_keys.items()):
            new_runit = new_runit.replace(token, "{:0.15f}".format(x))
        with open("{}.init".format(run_name), "w") as f:
            f.write(new_runit)
        return run_name

    def call_model(self,run_name=None):

        j = np.random.randint(10)
        f = open(self.wd + '/logs/{}.{}.log'.format(run_name, j), 'w')
        f.close()
        f = open(self.wd + '/logs/{}.{}.log'.format(run_name, j), 'a')
        
        if self.build_spool:
            call = [self.model]
            # build test
            call += ["-noRun"]
            call.append("{}/{}.init".format(self.wd,run_name))
            subprocess.call(call,
                            stdout=f, stderr=f, cwd=self.wd)
            # build spool
            call = ["perl"]
            call += [self.output_dir+"build_spool_batch.pl"]
            call += ["-q","-f","8"]
            subprocess.call(call,stdout=f,stderr=f,cwd=self.wd)
            
        # run model
        call = [self.model]
        for k, v in self.flags.items():
            call += ["-" + k, v]
        if run_name is None:
            run_name = self.name
        call.append("{}/{}.init".format(self.wd, run_name))
        while '' in call:
            call.remove('')


            print(call, flush=True)
        if self.actually_run:
            subprocess.call(call,
                            stdout=f, stderr=f, cwd=self.wd)
        else:
            subprocess.call(["grep","phi","{}.init".format(run_name)])

    def process_results(self):
        self.processor()
        
    def archive_results(self, X, run_id):
        # Copy sample data to an archive folder
        os.mkdir(self.results_archive + run_id + "/")
        arch_dir = self.results_archive + run_id + "/"
        print(arch_dir)
        for fname in os.listdir(self.output_dir_now + "sample/"):
            print(arch_dir + fname)
            copy2(self.output_dir_now + "sample/" + fname, arch_dir + fname)
        # Calculate climatologies and save to the archive folder

class WBM_monthly(object):
    """
    Each instance should sample the model domain and grid
    """

    def __init__(self, variables):
        self.variables = variables

    def get_drop_variables(self, path):
        # too lazy to build this as a closure
        try:
            return self.drop_variables
        except AttributeError:
            all_variables = xr.open_dataset(
                path + "/monthly/wbm_2000.nc").data_vars.keys()
            self.drop_variables = list(
                set(all_variables) - set(self.variables))
            return self.drop_variables

    def get_area_scale(self, path):
        try:
            return self.scale_ds
        except AttributeError:
            area = xr.open_dataset(path + "cell_area/full_cell_area.nc")
            area_scale = [False if ("WWTP" in x) or ("Denit" in x) or (
                "Sub" in x) else True for x in self.variables]
            scale_it = dict([(var, {True: area['cell_area'], False:area['cell_area'] * 0.0 + 1.0}[sc])
                             for var, sc in zip(self.variables, area_scale)])
            self.scale_ds = xr.Dataset(data_vars=scale_it)
            return self.scale_ds

    def __call__(self, path, archive_path):
        # Copy any sample results from path to the archive
        if os.path.exists(path + "sample/"):
            for fname in os.listdir(path + "sample/"):
                copy2(path + "sample/" + fname, archive_path + fname)
        # Create monthly climatologies and write to the archive
        with xr.open_mfdataset(path + "/monthly/wbm_*.nc", drop_variables=self.get_drop_variables(path)) as ds:
            inner_mean = (ds * ds.time.dt.days_in_month)
            outer_mean = inner_mean.groupby(
                'time.month').mean(dim='time', keep_attrs=True)
            result = outer_mean * self.get_area_scale(path)
            result = result.transpose('time', 'month', 'lat', 'lon')
            result.to_netcdf(
                archive_path + "wbm_month_clim.nc", format="NETCDF4")

            
class WBM_Pool(object):
    """
    Sets up a pool of WBM Simulators for a common domain,
      to distribute runs of unique parameter sets across scenarios.
    """

    def __init__(self, parameter_df, init_file_templates, names, run_flags=None,
                 observations=None, wd=None, model="./wbm.pl", model_class='monthly',
                 results_archive=None, proc_variables=None,actually_run=True,build_spool=False):

        # Define the parameters, their distributions, and their transforms
        # NEED SOME ABILITY TO KEEP INDIVIDUAL PARAMETERS STATIONARY
        self.param = pd.read_csv(parameter_df, index_col=0)
        self.define_parameter_transforms()
        # parameter dict will have both the stationary and varied variables in this order
        param_dict = odict([(k, []) for k in self.param.Parameter_int.values])

        # Define unique runit_templates for each scenario
        #  FEATURE: include specific paths as parameters, and then just not vary them
        #  through the sensitivity analysis.  That would make it possible to run scenarios with the same template
        assert type(
            init_file_templates) == list, "Expecting list of template file paths"
        assert type(
            names) == list, "Expecting list of names for model instances"
        assert len(init_file_templates) == len(
            names), "Name needed for each init file template"
        for init_file_template in init_file_templates:
            assert os.path.exists(
                init_file_template), "Init file template %s does not exist" % init_file_template
        self.inits = init_file_templates
        self.names = names
        # initialize a simulator for each init file template
        self.simulators = [WBM_Simulator(name, runit_template, param_dict, run_flags=run_flags,
                                         observations=observations, wd=wd, model=model,model_class=model_class,
                                         results_archive=results_archive, proc_variables=proc_variables,
                                         actually_run=actually_run,build_spool=build_spool) for
                           name, runit_template in zip(self.names, self.inits)]

        # initialize run queues for a collection of parameters

    def __call__(self, X, run_id,transform=True):
        # There may be different functionality if this is calling a single parameter set
        # across multiple scenarios, versus starting a queue of individual models for
        # multiple parameter sets.

        # Need to define datasets that need to be deleted?
        if transform:
            self.sample_parameter_set(X, run_id)
        else:
            self.sample_parameter_set_no_transform(X, run_id)

    def define_parameter_transforms(self):
        T = [None] * len(self.param)

        def unity(x): return x

        def static(x): return lambda y: x   # static closure.
        idx = dict((k, i) for i, k in enumerate(self.param.index))
        for i, (name, row) in enumerate(self.param.iterrows()):
            if row['distribution'] == 'log':
                T[i] = np.exp
            elif row['distribution'] == 'uniform':
                T[i] = unity
            elif row['distribution'] == 'static':
                T[i] = static(row['manual'])
        self.individual_transforms = T

    def transform(self, X):
        # NOTE: X will contain only the non-stationary parameters
        newX = []
        i = 0
        for t, (name, row) in zip(self.individual_transforms, self.param.iterrows()):
            if row['distribution'] == 'static':
                newX.append(t(0))  # always returns the static manual value
            else:
                newX.append(t(X[i]))
                i += 1
        idx = dict((k, i) for i, k in enumerate(self.param.index))
        for i, (name, row) in enumerate(self.param.iterrows()):
            if row['function'] == "infiltrFrac__Var_Max * infiltrFrac__Var_Min_R - infiltrFrac__Var_Max":
                newX[i] = newX[idx['infiltrFrac__Var_Max']] * \
                    newX[idx["infiltrFrac__Var_Min_R"]] - \
                    newX[idx["infiltrFrac__Var_Max"]]
            if row['function'] == "RhRt2__Var_Max - RhRt2__Var_Min":
                newX[i] = newX[idx["RhRt2__Var_Max"]] - \
                    newX[idx["RhRt2__Var_Min"]]
        return np.array(newX)

    def sample_parameter_set(self, X, run_id):
        # X is the parameter set viewed by the sampler, P by WBM
        P = self.transform(X)
        self.sample_parameter_set_no_transform(P, run_id)

    def sample_parameter_set_no_transform(self, P, run_id, joinup=True):
        # traverse the pool for this parameter set
        threads = []
        print("WBM_pool: ", self.names, run_id, flush=True)
        print(P,type(P))
        for name, simulator in zip(self.names, self.simulators):
            print(name, flush=True)
            assert name == simulator.name
            threads.append(Thread(target=simulator, args=(P,run_id))) # 
        for thread in threads:
            thread.start()
        if joinup:
            for thread in threads:
                thread.join()

    def sample_manual(self, run_id):
        X = self.param['manual'].values
        X[self.param['distribution'] == 'log'] = np.log(
            X[self.param['distribution'] == 'log'])
        self.sample_parameter_set(X, run_id)

    def sample_manminmax(self,run_id):
        X0 = self.param['manual'].values
        X0[self.param['distribution'] == 'log'] = np.log(
            X0[self.param['distribution'] == 'log'])
#        self.sample_parameter_set_no_transform(np.array(X0), "_mid", joinup=False)
        X1 = self.param['minimum'].values
        X1[self.param['distribution'] == 'log'] = np.log(
            X1[self.param['distribution'] == 'log'])
#        self.sample_parameter_set(X, run_id)
#        self.sample_parameter_set_no_transform(np.array(X1), "_min",joinup=False)
        X2 = self.param['maximum'].values
        X2[self.param['distribution'] == 'log'] = np.log(
            X2[self.param['distribution'] == 'log'])
#        self.sample_parameter_set_no_transform(np.array(X2), "_max",joinup=True)
        if (len(X0) == len(X1) == len(X2) == 1):
            self.sample_parameter_set_no_transform(np.array(X0), "_mid", joinup=True)
            self.sample_parameter_set_no_transform(np.array(X1), "_min",joinup=True)
            self.sample_parameter_set_no_transform(np.array(X2), "_max",joinup=True)
        elif (len(X0) == len(X1) == len(X2) == 2):
            # Oh bother.  Now we need to permutate
            self.sample_parameter_set_no_transform(np.array(X0), "_midmid",joinup=True)
#            self.sample_parameter_set_no_transform(np.array(X1), "_minmin",joinup=False) # not performing min's for parameter 2
            self.sample_parameter_set_no_transform(np.array([X0[0],X2[1]]), "_midmax",joinup=True)
            self.sample_parameter_set_no_transform(np.array([X1[0],X0[1]]), "_minmid",joinup=True)
            self.sample_parameter_set_no_transform(np.array([X1[0],X2[1]]), "_minmax",joinup=True)
            self.sample_parameter_set_no_transform(np.array([X2[0],X0[1]]), "_maxmid",joinup=True)
            self.sample_parameter_set_no_transform(np.array(X2), "_maxmax",joinup=True)
            
def test(param_df, prefix, testID="nexus_unc_"):
    init_file_templates = glob.glob("./{}*.init".format(prefix))
    init_file_templates.sort()
    print(init_file_templates)
    names = [testID + x.split(prefix)[1].split('.init')[0]
             for x in init_file_templates]

    # Need to have rm for sensitivity/calibration/UQ!
    wbm_pool = WBM_Pool(param_df, init_file_templates,
                        names, run_flags={"v": "","rm": ""},build_spool=True) #,"rm":""})
    print(wbm_pool.simulators)
    wbm_pool.sample_manminmax(testID)


if __name__ == "__main__":
    import sys
    param_df = 'parameter_def_v1.csv'
    init_file_prefix = 'MSag_nexus_nw_'
    test(param_df, init_file_prefix)
    param_df = 'parameter_def_v2.csv'
    init_file_prefix = 'MSag_nexus_wl_'
    test(param_df, init_file_prefix)


#    wbm_go=WBM_Simulator("wbm_mc1",runit,param,model="./wbm_blah.pl")
#    wbm_go(np.array([0.06,0.36,3.0,0.49]))
