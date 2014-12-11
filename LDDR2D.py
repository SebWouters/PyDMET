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

def CalculateLDDR( HubbardU, Omegas, eta ):

    LDDR = []

    lattice_size = np.array( [48, 96], dtype=int )
    cluster_size = np.array( [ 2,  2], dtype=int )
    Nelectrons   = np.prod( lattice_size ) # Half-filling
    antiPeriodic = True
    numBathOrbs  = 6 # Two more than the number of impurity orbitals = np.prod( cluster_size )
    orbital_i    = 0 # Take upper left corner of the impurity to kick out an electron (counting within "cluster_size" as in contiguous fortran array)
    
    theDMET = HubbardDMET.HubbardDMET( lattice_size, cluster_size, HubbardU, antiPeriodic )
    GSenergyPerSite, umatrix = theDMET.SolveGroundState( Nelectrons )
    
    for omega in Omegas:

        EperSite_forward,  GF_forward   = theDMET.SolveResponse( umatrix, Nelectrons, orbital_i, omega, eta, numBathOrbs, 'F' )
        EperSite_backward, GF_backward  = theDMET.SolveResponse( umatrix, Nelectrons, orbital_i, omega, eta, numBathOrbs, 'B' )
        SpectralFunction = - ( GF_forward.imag - GF_backward.imag ) / math.pi
        LDDR.append( SpectralFunction )
        print "LDDR( U =",HubbardU,"; omega =",omega,") =",SpectralFunction
    
    return LDDR
    
