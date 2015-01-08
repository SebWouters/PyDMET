'''
    PyDMET: a python implementation of density matrix embedding theory
    Copyright (C) 2014, 2015 Sebastian Wouters
    
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
import HamInterface
import LinalgWrappers
import DMETorbitals

def CalculateLDOS( Omegas, eta ):

    LDOS = []

    lattice_size = np.array( [48, 96], dtype=int )
    Nelectrons   = np.prod( lattice_size ) # Half-filling
    antiPeriodic = True
    orbital_i    = 0 # Take upper left corner of the impurity to kick out an electron (counting within "cluster_size" as in contiguous fortran array)

    Ham = HamInterface.HamInterface(lattice_size, 0.0, antiPeriodic)
    energiesRHF, solutionRHF = LinalgWrappers.SortedEigSymmetric( Ham.Tmat )
    assert( Nelectrons % 2 == 0 )
    numPairs = Nelectrons / 2
    if ( energiesRHF[ numPairs ] - energiesRHF[ numPairs-1 ] < 1e-8 ):
        print "ERROR: The single particle gap is zero!"
        assert( energiesRHF[ numPairs ] - energiesRHF[ numPairs-1 ] >= 1e-8 )

    for omega in Omegas:

        omegabis = omega + 0.5 * ( energiesRHF[ numPairs-1 ] + energiesRHF[ numPairs ] ) # Add the chemical potential to omega!
        thisLDOS = DMETorbitals.ConstructMeanFieldLDOS( orbital_i, omegabis, eta, energiesRHF, solutionRHF, numPairs )
        LDOS.append( thisLDOS )

    return LDOS

