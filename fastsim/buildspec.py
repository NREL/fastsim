"""Module containing function for building spec list for numba jitclass decorator."""

import numpy as np
from numba import float64, int32, bool_    # import the types
from numba.types import string

def build_spec(instance, error='raise'):
    """Given a FASTSim object instance, returns list of tuples with 
    attribute names and numba types.
    
    Arguments:
    ----------
    instance : instance of FASTSim class (e.g. vehicle.Vehicle())
    error : 'raise' -- raise error when invalid key is used
            'warn' -- warn without error when invalid key is used
            'ignore' -- completely ignore errors"""

    spec_tuples = [
        ([float, np.float32, np.float64, np.float], float64, float64[:]),
        ([int, np.int32, np.int64, np.int], int32, int32[:]),
        ([bool, np.bool, np.bool_], bool_, bool_[:]),
        ([str], string, string[:]),
    ]

    if 'sim_drive' in instance.__dir__():
        # if this import is done before this if branch is triggered,
        # weird circular import issues may happen
        from fastsim import vehicle, cycle, parameters, simdrive
        # run sim_drive to flesh out all the attributes
        instance.sim_drive()
        # list of tuples containg possible types, assigned type for scalar,
        # and assigned type for array
        spec_tuples.extend([
            # complex types that are attributes of simdrive.SimDrive*
            ([vehicle.Vehicle], vehicle.VehicleJit.class_type.instance_type, None),
            ([cycle.Cycle], cycle.CycleJit.class_type.instance_type, None),
            ([parameters.PhysicalProperties],
                parameters.PhysicalPropertiesJit.class_type.instance_type, None),
            ([simdrive.SimDriveParamsClassic],
                simdrive.SimDriveParams.class_type.instance_type, None),
        ])

    spec = []

    for key, val in instance.__dict__.items():
        t = type(val)
        jit_type = None
        if t == np.ndarray:
            for matched_types, _, assigned_type in spec_tuples:
                if type(val[0]) in matched_types:
                    jit_type = assigned_type
                    break
        else:
            for matched_types, assigned_type, _ in spec_tuples:
                if t in matched_types:
                    jit_type = assigned_type
                    break
        if jit_type is None:
            if error == 'raise':
                raise Exception(
                    str(t) + " does not map to anything in spec_tuples" 
                    + '\nYou may need to modify `spec_tuples` in `build_spec`.')
            elif error == 'warn':
                print("Warning: " + str(t) + " does not map to anything in spec_tuples."
                    + '\nYou may need to modify `spec_tuples` in `build_spec`.')
            elif error == 'ignore':
                pass
            else:
                raise Exception('Invalid value `' + str(error) 
                    + '` provided in build_spec error argument.') 
        spec.append((key, jit_type))

    return spec
