import numpy as np
import h5py
import sys

def rdabs_species(filnm, species):
    """
    filnm: the name of the absco file
    species: the name of the gas species
    """

    if species not in ['o2', 'h2o', 'co2', 'ch4']:
        sys.exit('Error: The species %s is not supported!' % species)

    # Open the hdf5 file
    with h5py.File(filnm, 'r') as h5data:
        # Read in the Pressure values
        # p_species has 71 elements
        p_species = h5data['Pressure'][...]

        # Obtain hpa pressure
        hpa_species = p_species/100.0

        # Read in the Temperature values
        # tk_species is 71 x 17 (by hdfview)
        # and that the data is stored tk_species(17, 71)
        # 71 pressure values
        # 17 temperature values (not the same at each pressure level)
        tk_species = h5data['Temperature'][...]

        # Read in the Wavenumber grid
        wcm_species = h5data['Wavenumber'][...]

        # Read in the Broadener VMRs
        broad_species = h5data['Broadener_01_VMR'][...]

    Gas_ID = {'o2' : '07',
              'h2o': '01',
              'co2': '02',
              'ch4': '06'}

    # Specify the units of the data
    units_species = np.chararray(5)
    units_species[0] = 'Pressure (Pascal)'
    units_species[1] = 'Temperature (K)'
    units_species[2] = 'Wavenumber (cm-1)'
    units_species[3] = f'Gas_{Gas_ID[species]}_Absorption (cm2/mol)'
    units_species[4] = 'Broadner_01_VMR   (volume mix ratio)'
    
    return wcm_species, p_species, tk_species, broad_species,\
           hpa_species, units_species

