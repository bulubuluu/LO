"""
Example usage of the non-commutative Standard Model amplitude calculations.

This script demonstrates how to:
1. Generate or use pre-defined momenta for a 2->2 scattering process
2. Calculate both Standard Model and non-commutative amplitudes
3. Handle momentum index positions (up/down)

The example shows a simple q q̄ → Z Z process where q is a quark and Z is a Z-boson.
"""

import math 
import numpy as np
from sympy import symbols, LeviCivita, Matrix
import bsm_nc_module as nc



##########################################################################
##########################################################################
##########################################################################


# indicates if momentum-data are pre-generated or not
# takes values "yes" or "no"
from_database = "no"

if from_database == "no":
    # generates random momenta, p1 + p2 = k1 + k2
    # p1, p2 - momenta of incoming quarks
    # k1, k2 - momenta of outgoing Z-bosons	
    [p1, p2, k1, k2] = nc.random_momenta()
    # p1 = np.array([1, 2, 3, 4])	# uncomment for manual entry
    # p2 = np.array([2, 3, 4, 5])	# uncomment for manual entry
    # k1 = np.array([5, 6, 7, 8])	# uncomment for manual entry
    # k2 = np.array([6, 8, -8, 9])	# uncomment for manual entry
# else:
    # imports data from a file
    # to be added later


# nc.sm_amp gives SM |amplitude|^2
# nc.nc_amp gives NC |amplitude|^2
# index_position indicates co(ntra)variance of momentum-data, takes values "up" or "down"  
# p1 - momentum of one of incoming quarks
# k1, k2 - momenta of outgoing Z-bosons		
index_position = "up"
print("SM contribution = ", nc.sm_amp(p1, k1, k2, index_position))
print("NC contribution = ", nc.nc_amp(p1, k1, k2, index_position))

def calculate_amplitudes(p1: nc.FourVector, k1: nc.FourVector, k2: nc.FourVector,
                        index_position: str = "up") -> tuple[float, float]:
    """Calculate both SM and NC amplitudes for given momenta.
    
    Args:
        p1: Four-momentum of incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        index_position: Either "up" or "down" for momentum indices
    
    Returns:
        tuple[float, float]: (SM amplitude, NC amplitude)
    """
    sm_value = nc.sm_amp(p1, k1, k2, index_position)
    nc_value = nc.nc_amp(p1, k1, k2, index_position)
    return sm_value, nc_value

def main():
    # Example 1: Using random momenta
    print("\nExample 1: Random momenta")
    print("-" * 40)
    
    # Generate random momenta that satisfy energy-momentum conservation
    p1, p2, k1, k2 = nc.random_momenta()
    
    # Calculate amplitudes with upper indices
    sm_amp, nc_amp = calculate_amplitudes(p1, k1, k2, "up")
    print(f"SM amplitude (upper indices) = {sm_amp:.6e}")
    print(f"NC amplitude (upper indices) = {nc_amp:.6e}")
    
    # Example 2: Using pre-defined momenta
    print("\nExample 2: Pre-defined momenta")
    print("-" * 40)
    
    # Define momenta manually (example values)
    p1 = np.array([10.0, 0.0, 0.0, 10.0])  # High-energy quark
    k1 = np.array([5.0, 3.0, 0.0, 4.0])    # First Z-boson
    k2 = np.array([5.0, -3.0, 0.0, -4.0])  # Second Z-boson
    
    # Calculate amplitudes with both upper and lower indices
    sm_amp_up, nc_amp_up = calculate_amplitudes(p1, k1, k2, "up")
    sm_amp_down, nc_amp_down = calculate_amplitudes(p1, k1, k2, "down")
    
    print(f"SM amplitude (upper indices)  = {sm_amp_up:.6e}")
    print(f"SM amplitude (lower indices)  = {sm_amp_down:.6e}")
    print(f"NC amplitude (upper indices)  = {nc_amp_up:.6e}")
    print(f"NC amplitude (lower indices)  = {nc_amp_down:.6e}")

if __name__ == "__main__":
    main()



