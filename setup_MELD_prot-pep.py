#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import meld
from meld.remd import ladder, adaptor, leader
import meld.system.montecarlo as mc
from meld import system
from meld import comm, vault
from meld import parse
from meld import remd
from openmm import unit as u
import glob as glob

N_REPLICAS = 30
N_STEPS =2000
BLOCK_SIZE = 100

def gen_state(s, index):
    state = s.get_state_template()
    state.alpha = index / (N_REPLICAS - 1.0)
    return state


def get_dist_restraints(filename, s, scaler, ramp):  
    dists = []
    rest_group = []
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            i = int(cols[0])-1
            name_i = cols[1]
            j = int(cols[2])-1
            name_j = cols[3]
            dist = float(cols[4])

            rest = s.restraints.create_restraint('distance', scaler, ramp,
                                                 r1=0.0*u.nanometer, r2=0.0*u.nanometer, r3=dist*u.nanometer, r4=(dist+0.2)*u.nanometer,
                                                 k=350*u.kilojoule_per_mole/u.nanometer **2,
						 atom1=s.index.atom(i, name_i,),
						 atom2=s.index.atom(j, name_j))	
            rest_group.append(rest)
    return dists

def get_dist_restraints_protein(filename, s, scaler, ramp):  
    dists = []
    rest_group = []
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            i = int(cols[0])-1
            name_i = cols[1]
            j = int(cols[2])-1
            name_j = cols[3]
            dist = float(cols[4])

            rest = s.restraints.create_restraint('distance', scaler, ramp,
                                                  r1=(dist-0.2)*u.nanometer, r2=(dist-0.1)*u.nanometer, r3=(dist+0.1)*u.nanometer, r4=(dist+0.2)*u.nanometer, 
						  k=350*u.kilojoule_per_mole/u.nanometer **2,
						  atom1=s.index.atom(i, name_i,),
						  atom2=s.index.atom(j, name_j))	
            rest_group.append(rest)
    return dists

def make_cartesian_collections(s, scaler, residues, delta,k):
    cart = []
    backbone = ['CA']
    #Residues are 1 based
    #index of atoms are 1 base
    for i in residues:
        # print i
        for b in backbone:
            # print b
            atom_index = s.index.atom(i,b)
            x,y,z = s.template_coordinates[atom_index] #/10
            rest = s.restraints.create_restraint('cartesian',scaler, atom_index=atom_index,
                x=x*u.nanometer, y=y*u.nanometer, z=z*u.nanometer, delta=delta*u.nanometer,force_const=k*u.kilojoule_per_mole/u.nanometer **2)
            cart.append(rest)
    return cart    
    
def setup_system():

    # load the template Note: note minimized PDB
    templates = glob.glob('TEMPLATES/*')
    # build the system
    p = meld.AmberSubSystemFromPdbFile(templates[0]) 
    build_options = meld.AmberOptions(
      forcefield="ff14sbside",
      implicit_solvent_model = 'gbNeck2',
      use_big_timestep = True,
      cutoff = 1.8*u.nanometers,
      remove_com = False,
      #use_amap = False,
      enable_amap = False,
      amap_beta_bias = 1.0,
    )
    builder = meld.AmberSystemBuilder(build_options)
    s = builder.build_system([p]).finalize()
    #set the temperatures
    s.temperature_scaler = system.temperature.GeometricTemperatureScaler(0, 0.3, 300.*u.kelvin, 550.*u.kelvin)

    n_res = s.residue_numbers[-1]   

    ramp = s.restraints.create_scaler('nonlinear_ramp', start_time=1, end_time=200,
                                      start_weight=1e-3, end_weight=1, factor=4.0)
    #
    # Distance scalers
    #
    prot_scaler = s.restraints.create_scaler('constant')
    protein_pep_scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)

    #
    # Distance Restraints
    #
    # prot_pep_contact
    prot_pep_rest = get_dist_restraints('protein_pep_all.dat',s,protein_pep_scaler, ramp)
    s.restraints.add_selectively_active_collection(prot_pep_rest, int(len(prot_pep_rest)*1.00))
    
    #restrain to keep prot fix
    prot_rest = get_dist_restraints_protein('protein_contacts.dat',s,prot_scaler, ramp)
    s.restraints.add_selectively_active_collection(prot_rest, int(len(prot_rest)*0.90))
    
    pro_res= range(1,69) 
    rest = make_cartesian_collections(s,prot_scaler, pro_res,delta=0.25, k=350.)
    s.restraints.add_as_always_active_list(rest)

    # create the options
    options = meld.RunOptions(
        timesteps = 11111,
        minimize_steps = 20000,
    )

    # create a store
    store = vault.DataStore(gen_state(s,0), N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE) 
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=48 * 48)
    policy_1 = adaptor.AdaptationPolicy(2.0, 50, 50)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy_1)

    remd_runner = remd.leader.LeaderReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a)
    store.save_remd_runner(remd_runner)

    # create and store the communicator
    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS, timeout=60000)
    store.save_communicator(c)

    # create and save the initial states
    states = [gen_state(s, i) for i in range(N_REPLICAS)]
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()

    return s.n_atoms
   
setup_system()
