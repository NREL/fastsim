"""Module containing function for building spec list for numba jitclass decorator."""

import inspect
import warnings
import numpy as np
from numba import float64, int32, bool_    # import the types
from numba.types import string

from fastsim import parameters, cycle, vehicle, simdrive, simdrivehot


def build_spec(instance, error='raise', extra=None):
    """
    Given a FASTSim object instance, returns list of tuples with 
    attribute names and numba types.
    
    Arguments:
    ----------
    instance : instance of FASTSim class (e.g. vehicle.Vehicle())
    error : 'raise' -- raise error when invalid key is used
            'warn' -- warn without error when invalid key is used
            'ignore' -- completely ignore errors
    extra : list of tuples that extends spec in format:
        >>> [
        >>>     ('var_name', type), # examples to follow
        >>>     ('var1', numba.float64), # for a scalar
        >>>     ('var2', numba.float64[:]), # for an array
        >>> ]   
        this can optionally be extended to output of build_spec or 
        provided here as a kwarg.
    """

    # types that are native to python/numpy/numba
    spec_tuples = [
        ([float, np.float32, np.float64, np.float], float64, float64[:]),
        ([int, np.int32, np.int64, np.int], int32, int32[:]),
        ([bool, np.bool, np.bool_], bool_, bool_[:]),
        ([str], string, string[:]),
    ]

    # code for handling custom and/or user-defined types 
    attrs = [instance.__getattribute__(key) for key in instance.__dict__.keys()]
    attr_types = [type(attr) for attr in attrs]

    if 'sim_drive' in instance.__dir__():
        instance.sim_drive()

### BEGIN FASTSim Hot only 

    if simdrivehot.AirProperties in attr_types:
        from . import simdrivehotjit
        spec_tuples.append(([simdrivehot.AirProperties], 
            simdrivehotjit.AirPropertiesJit.class_type.instance_type, None))
    if simdrivehot.VehicleThermal in attr_types:
        from . import simdrivehotjit
        spec_tuples.append(([simdrivehot.VehicleThermal], 
            simdrivehotjit.VehicleThermalJit.class_type.instance_type, None))
    if simdrivehot.ConvectionCalcs in attr_types:
        from . import simdrivehotjit
        spec_tuples.append(([simdrivehot.ConvectionCalcs], 
            simdrivehotjit.ConvectionCalcsJit.class_type.instance_type, None))

### END FASTSim Hot only 

    # list of tuples of base types and their corresponding jit types
    if cycle.Cycle in attr_types:
        from . import cyclejit
        spec_tuples.append(
            ([cycle.Cycle], cyclejit.CycleJit.class_type.instance_type, None))
    if vehicle.Vehicle in attr_types:
        from . import vehiclejit
        spec_tuples.append(
            ([vehicle.Vehicle], vehiclejit.VehicleJit.class_type.instance_type, None))
    if parameters.PhysicalProperties in attr_types:
        from . import parametersjit
        spec_tuples.append((
            [parameters.PhysicalProperties], 
                parametersjit.PhysicalPropertiesJit.class_type.instance_type, None))
    if simdrive.SimDriveParamsClassic in attr_types:
        from . import simdrivejit
        spec_tuples.append((
            [simdrive.SimDriveParamsClassic], simdrivejit.SimDriveParams.class_type.instance_type, None))

    spec = []

    def isprop(attr):
        return isinstance(attr, property)

    prop_attrs = [name for (name, _) in inspect.getmembers(vehicle.Vehicle, isprop)]

    for key, val in instance.__dict__.items():
        if key in prop_attrs:
            continue # properties should not be typed for numba jitclass

        t = type(val)
        jit_type = None
        if type(val) == np.ndarray:
            for matched_types, _, assigned_type in spec_tuples:
                if type(val[0]) in matched_types:
                    jit_type = assigned_type
                    break
        else:
            for matched_types, assigned_type, _ in spec_tuples:
                if type(val) in matched_types:
                    jit_type = assigned_type
                    break
        if jit_type is None:
            msg = f"Type `{t}` of `{val}` for `{key}` not known to build_spec."
            if error == 'raise':
                raise Exception(msg)
            elif error == 'warn':
                warnings.warn(msg)
            elif error == 'ignore':
                pass
            else:
                raise Exception(msg) 
        spec.append((key, jit_type))

        if extra:
            spec.extend(extra)

    return spec