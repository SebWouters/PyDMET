'''
    PyDMET: a python implementation of density matrix embedding theory
    Copyright (C) 2014 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

import numpy as np
import math
import HubbardDMET

def CalculateLDOS( HubbardU, lowerbound, upperbound, stepsize, eta ):

    LDOS   = []
    Omegas = []

    lattice_size = np.array( [24, 48], dtype=int )
    cluster_size = np.array( [ 2,  2], dtype=int )
    Nelectrons   = np.prod( lattice_size ) # Half-filling
    antiPeriodic = True
    numBathOrbs  = 6 # Two more than the number of impurity orbitals = np.prod( cluster_size )
    orbital_i    = 0 # Take upper left corner of the impurity to kick out an electron (counting within "cluster_size" as in contiguous fortran array)
    
    theDMET = HubbardDMET.HubbardDMET( lattice_size, cluster_size, HubbardU, antiPeriodic )
    
    for omega in np.arange( lowerbound, upperbound, stepsize ):

        EperSite_addition, GF_addition = theDMET.SolveResponse( Nelectrons, orbital_i, omega, eta, numBathOrbs, 'A' )
        EperSite_remocal,  GF_removal  = theDMET.SolveResponse( Nelectrons, orbital_i, omega, eta, numBathOrbs, 'R' )
        SpectralFunction = - 2.0 * ( GF_addition.imag + GF_removal.imag ) / math.pi # Factor of 2 due to summation over spin projection
        Omegas.append( omega )
        LDOS.append( SpectralFunction )
    
    return ( Omegas, LDOS )
    
