"""
Global constants representing unit conversions that shourd never change, 
physical properties that should rarely change, and vehicle model parameters 
that can be modified by advanced users.  
"""

import os
import numpy as np
import json
from pathlib import Path
import copy
import inspect
from typing import Callable

import fastsimrust as fsr


THIS_DIR = Path(__file__).parent


### Unit conversions that should NEVER change
MPH_PER_MPS = 2.2369
M_PER_MI = 1609.00

class PhysicalProperties(object):
    """Container class for physical constants that could change under certain special 
    circumstances (e.g. high altitude or extreme weather) """

    @classmethod
    def from_dict(cls, pp_dict:dict):
        obj = cls()
        for k, v in pp_dict.items():
            obj.__setattr__(k, v)
        return obj

    def __init__(self):
        # Make this altitude and temperature dependent, and allow it to change with time
        self.air_density_kg_per_m3 = 1.2  # Sea level air density at approximately 20C
        self.a_grav_mps2 = 9.81
        self.kwh_per_gge = 33.7 # kWh per gallon of gasoline
        self.fuel_rho_kg__L = 0.75 # gasoline density in kg/L https://inchem.org/documents/icsc/icsc/eics1400.htm
        self.fuel_afr_stoich = 14.7 # gasoline stoichiometric air-fuel ratio https://en.wikipedia.org/wiki/Air%E2%80%93fuel_ratio

    def get_fuel_lhv_kJ__kg(self):
        # fuel_lhv_kJ__kg = kwh_per_gge / 3.785 [L/gal] / fuel_rho_kg__L [kg/L] * 3_600 [s/hr] = [kJ/kg]
        return self.kwh_per_gge / 3.785 / self.fuel_rho_kg__L * 3_600 

    def set_fuel_lhv_kJ__kg(self, value):
        # kwh_per_gge = fuel_lhv_kJ__kg * fuel_rho_kg__L [kg/L] * 3.785 [L/gal] / 3_600 [s/hr] = [kJ/kg]
        self.kwh_per_gge = value * 3.785 * self.fuel_rho_kg__L / 3_600

    fuel_lhv_kJ__kg = property(get_fuel_lhv_kJ__kg, set_fuel_lhv_kJ__kg)

ref_physical_properties = PhysicalProperties()

# TODO: the below 3 functions have been moved here temporarily to avoid a circular dependency for importing utils (which imports properties)
def isprop(attr) -> bool:
    "Checks if instance attribute is a property."
    return isinstance(attr, property)

def isfunc(attr) -> bool:
    "Checks if instance attribute is method."
    return isinstance(attr, Callable)

def get_attrs(instance):
    """
    Given an instantiated object, returns attributes that are not:
    -- callable  
    -- special (i.e. start with `__`)  
    -- properties  
    """

    keys = []
    props = [name for (name, _) in inspect.getmembers(type(instance), isprop)]
    methods = [name for (name, _) in inspect.getmembers(type(instance), isfunc)]
    for key in instance.__dir__():
        if not(key.startswith("_")) and key not in (props + methods):
            keys.append(key)
    return keys

def copy_physical_properties(p:PhysicalProperties, return_type:str=None, deep:bool=True):
    """
    Returns copy of PhysicalProperties.
    Arguments:
    p: instantianed PhysicalProperties or RustPhysicalProperties 
    return_type: 
        default: infer from type of p
        'dict': dict
        'physical_properties': PhysicalProperties 
        'legacy': LegacyPhysicalProperties -- NOT IMPLEMENTED YET; is it needed?
        'rust': RustPhysicalProperties
    deep: if True, uses deepcopy on everything
    """
    p_dict = {}

    for key in get_attrs(ref_physical_properties):
        val_to_copy = p.__getattribute__(key)
        p_dict[key] = copy.deepcopy(val_to_copy) if deep else val_to_copy

    if return_type is None:
        if type(p) == fsr.RustPhysicalProperties:
            return_type = 'rust'
        elif type(p) == PhysicalProperties:
            return_type = 'physical_properties'
        else:
            raise NotImplementedError(
                "Only implemented for rust and physical_properties")
    
    if return_type == 'dict':
        return p_dict
    elif return_type == 'physical_properties':
        return PhysicalProperties.from_dict(p_dict)
    elif return_type == 'rust':
        return fsr.RustPhysicalProperties(**p_dict)
    else:
        raise ValueError("Invalid return_type.")

    return p


### Vehicle model parameters that should be changed only by advanced users
# Discrete power out percentages for assigning FC efficiencies -- all hardcoded ***
fc_pwr_out_perc = np.array(
    [0, 0.005, 0.015, 0.04, 0.06, 0.10, 0.14, 0.20, 0.40, 0.60, 0.80, 1.00], 
    dtype=np.float64)

# fc arrays and parameters
# Efficiencies at different power out percentages by FC type -- all
fc_eff_map_si = np.array([0.10, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30])
fc_eff_map_atk = np.array([0.10, 0.12, 0.28, 0.35, 0.375, 0.39, 0.40, 0.40, 0.38, 0.37, 0.36, 0.35])
fc_eff_map_diesel = np.array([0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])
fc_eff_map_fuel_cell = np.array([0.10, 0.30, 0.36, 0.45, 0.50, 0.56, 0.58, 0.60, 0.58, 0.57, 0.55, 0.54])
fc_eff_map_hd_diesel = np.array([0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])


# Relatively continuous power out percentages for assigning FC efficiencies
fc_perc_out_array = np.r_[np.arange(0, 3.0, 0.1), np.arange(
    3.0, 7.0, 0.5), np.arange(7.0, 60.0, 1.0), np.arange(60.0, 105.0, 5.0)] / 100  # hardcoded ***

# motor arrays and parameters
mc_pwr_out_perc = np.array(
    [0.00, 0.02, 0.04, 0.06, 0.08,	0.10,	0.20,	0.40,	0.60,	0.80,	1.00])
large_baseline_eff = np.array(
    [0.83, 0.85,	0.87,	0.89,	0.90,	0.91,	0.93,	0.94,	0.94,	0.93,	0.92])
small_baseline_eff = np.array(
    [0.12,	0.16,	 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93,	0.93,	0.92])
modern_max = 0.95
mc_perc_out_array = np.linspace(0, 1, 101)

chg_eff = 0.86 # charger efficiency for PEVs, this should probably not be hard coded long term

# loading long arrays from json file
with open(THIS_DIR / 'resources' / 'longparams.json', 'r') as paramfile:
    param_dict = json.load(paramfile)

# PHEV-specific parameters
rechg_freq_miles = param_dict['rechgFreqMiles']
uf_array = param_dict['ufArray']
