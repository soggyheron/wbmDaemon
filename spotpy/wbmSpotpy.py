#!/usr/bin/env -S python3 -u
# -u flag attempts to force flushing of log file.
# -S tells env to split command line
import pandas as pd
import numpy as np
import wbmDaemon
import os
import shutil
from collections import OrderedDict
import spotpy
import spotpy.parameter as sp
import spotpy.objectivefunctions as sof
import matplotlib.pyplot as plt


class wbm_spot_setup(object):
    def __init__(self,param_file,template_init,observation_table,parallel='seq'):
        # Read parameters from csv

        # TODO - assuming uniform or logUniform for now.
        self.param=pd.read_csv(param_file,index_col=0)
        self.define_parameter_transforms()
        param_dict = OrderedDict([(k, []) for k in self.param.Parameter_int.values])
        print(self.param)
        logs=self.param['distribution']=='log'
        self.param.loc[logs, "minimum maximum manual".split()] = self.param.loc[logs, "minimum maximum manual".split(
        )].apply(dict(minimum=np.log, maximum=np.log, manual=np.log))

        params=self.param
        params.index.name="name"
        params=params.rename(columns={'minimum':'low','maximum':'high','manual':'optguess'})
        params=params["low high optguess".split()]

        Ps=[]
        # Presently only have hard-coded Uniform parameter distribution
        for name,vals in params.iterrows():
            kwargs=dict(name=name,**vals)
            Ps.append(sp.Uniform(**kwargs))
        # These are the data with Spot parameters.  The parameters function generates the spot parameter objects?
        self.params=Ps
    
        # Should set-up daemons now.
        # one simulator?
        self.model=wbmDaemon.WBM_Simulator_Factory(template_init,
                                           param_dict,
                                           observations=observation_table,
                                           model_class='sample',
                                           actually_run=True,    #TRUE
                                           run_flags={"rm":"","v":"",
                                                      "spoolDir":"/net/nfs/swift/raid2/data/psiren/WBM_spool/"})
        """
        self.model=wbmDaemon.WBM_Spot_Pool(param_file,"PSIREN_ipswich.init",
                                      run_flags={"rm":"","v":"",
                                                "spoolDir":"/net/nfs/swift/raid2/data/psiren/WBM_spool/"},
                                      observations="../proc/ipswich_usgs_data.csv",
                                      model_class='sample',actually_run=False)
        """
                                      
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

        
    def parameters(self):
        return sp.generate(self.params)


    def simulation(self,X):
        t=self.transform(X)
#        print(list(zip(self.params,X,t)))
        result=self.model(str(os.getpid()))(t)
        return result

    def evaluation(self):
        return self.model(os.getpid()).processor.obs

    # SpotPy seems to take the evaluation data as immutable.  I don't know
    #  how to trick it into being mutable.
    
    def objectivefunction(self,simulation,evaluation):
#        print(simulation.shape,evaluation.shape)
#        print(simulation[0:10],evaluation[0:10])
#        plt.plot(evaluation,simulation,"b.")
#        plt.plot([0,60],[0,60],'k:')
#        plt.show()
        
#        like = spotpy.objectivefunctions.nashsutcliffe(evaluation,simulation)
        like = spotpy.objectivefunctions.rmse(evaluation,simulation)
        return like
    

def main(param_fname):

    # move some things over to shared memory
#    os.mkdir("/dev/shm/spotproc/")
#    os.mkdir("/dev/shm/spotproc/output/")
    templateInit="./PSIREN_ipswich.init"
    obsTab="../proc/ipswich_down_usgs_data.csv"
    setup = wbm_spot_setup(param_fname,templateInit,obsTab)
    #print(spotpy.algorithms.sceua.__init__.__doc__)
    sampler = spotpy.algorithms.sceua(setup,dbname="sceua_psiren",dbformat="csv",parallel="mpi")
    sampler.sample(5000,ngs=12,kstop=4,peps=0.05,pcento=0.05)
#    print(spotpy.analyser.load_csv_results("sceua_psiren"))
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        param_fname=sys.argv[1]
    if len(sys.argv) > 2:
        templateInit=sys.argv[2]
    else:
        param_fname="parameter_def_v0.1.csv"
    main(param_fname)
