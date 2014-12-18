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

def CalculateLDOS( HubbardU, Omegas, eta ):

    LDOS = []

    lattice_size = np.array( [ 960 ], dtype=int )
    cluster_size = np.array( [   4 ], dtype=int )
    Nelectrons   = np.prod( lattice_size ) # Half-filling
    antiPeriodic = True
    numBathOrbs  = 6 # Two more than the number of impurity orbitals = np.prod( cluster_size )

    theDMET = HubbardDMET.HubbardDMET( lattice_size, cluster_size, HubbardU, antiPeriodic )
    GSenergyPerSite, umatrix = theDMET.SolveGroundState( Nelectrons )
    
    for omega in Omegas:

        EperSite_addition, GF_addition = theDMET.SolveResponse( umatrix, Nelectrons, omega, eta, numBathOrbs, 'A' )
        EperSite_remocal,  GF_removal  = theDMET.SolveResponse( umatrix, Nelectrons, omega, eta, numBathOrbs, 'R' )
        SpectralFunction = - 2.0 * ( GF_addition.imag + GF_removal.imag ) / math.pi # Factor of 2 due to summation over spin projection
        LDOS.append( SpectralFunction )
        print "LDOS( U =",HubbardU,"; omega =",omega,") =",SpectralFunction

    return LDOS
    
