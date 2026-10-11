
import sys

def calc_ext(denf, iz, absco):

    # *********
    # The absco coefficients are cm2 per molecule
    # denf are number molecules per cm3 for one species
    # To yield extinction [1/km], multiply
    #   1.0e5 * absco (cm2/molecule) o2den (molecule/cm3) 
    # *********

    conv = 1.0e5 * denf[iz]
    ext = (conv*absco).flatten() 
 
    return ext
