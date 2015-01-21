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
import math
import sys
sys.path.append('src')
import HubbardDMET
import InsulatingPMguess

def CalculateLDOS( HubbardU, Omegas, eta ):

    LDOS = []

    lattice_size = np.array( [48, 96], dtype=int )
    cluster_size = np.array( [ 2,  2], dtype=int )
    Nelectrons   = np.prod( lattice_size ) # Half-filling
    antiPeriodic = True
    skew2by2cell = False
    numBathOrbs  = 6 # Two more than the number of impurity orbitals = np.prod( cluster_size )
    
    theDMET = HubbardDMET.HubbardDMET( lattice_size, cluster_size, HubbardU, antiPeriodic, skew2by2cell )
    if ( HubbardU > 6.5 ):
        if ( skew2by2cell ):
            umat_guess = InsulatingPMguess.PullSkew2by2( HubbardU )
        else:
            umat_guess = InsulatingPMguess.PullSquare2by2( HubbardU )
    else:
        umat_guess = None
    GSenergyPerSite, umatrix = theDMET.SolveGroundState( Nelectrons, umat_guess )
    
    for omega in Omegas:

        EperSite_add, GF_add, notSC_GF_add = theDMET.SolveResponse( umatrix, Nelectrons, omega, eta, numBathOrbs, 'A' )
        EperSite_rem, GF_rem, notSC_GF_rem = theDMET.SolveResponse( umatrix, Nelectrons, omega, eta, numBathOrbs, 'R' )
        SpectralFunction       = - 2.0 * ( GF_add.imag + GF_rem.imag ) / math.pi # Factor of 2 due to summation over spin projection
        SpectralFunction_notSC = - 2.0 * ( notSC_GF_add.imag + notSC_GF_rem.imag ) / math.pi
        LDOS.append( SpectralFunction )
        print "LDOS( U =",HubbardU,"; omega =",omega,") NOT self-consistent =",SpectralFunction_notSC
        print "LDOS( U =",HubbardU,"; omega =",omega,")     self-consistent =",SpectralFunction
    
    return LDOS
    
