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

def CalculateLDOS( HubbardU, Omegas, eta, doSelfConsistent=True ):

    LDOS = []

    lattice_size = np.array( [48, 96], dtype=int )
    cluster_size = np.array( [ 2,  2], dtype=int )
    Nelectrons   = np.prod( lattice_size ) # Half-filling
    antiPeriodic = True
    numBathOrbs  = np.prod( cluster_size ) + 2
    
    theDMET = HubbardDMET.HubbardDMET( lattice_size, cluster_size, HubbardU, antiPeriodic )
    if ( HubbardU > 6.5 ):
        umat_guess = InsulatingPMguess.PullSquare2by2( HubbardU )
    else:
        umat_guess = None
    GSenergyPerSite, umatrix = theDMET.SolveGroundState( Nelectrons, umat_guess )
    umats_RESP_add = []
    umats_RESP_rem = []
    for imp in range( np.prod( cluster_size ) ):
        umats_RESP_add.append( umatrix )
        umats_RESP_rem.append( umatrix )
    
    for omega in Omegas:
        
        if ( doSelfConsistent==True ):
            EperSite_add, GF_add, umats_RESP_add = theDMET.SolveResponse( umatrix, umats_RESP_add, Nelectrons, omega, eta, numBathOrbs, 'A' )
            EperSite_rem, GF_rem, umats_RESP_rem = theDMET.SolveResponse( umatrix, umats_RESP_rem, Nelectrons, omega, eta, numBathOrbs, 'R' )
        else:
            EperSite_add, GF_add, dump = theDMET.SolveResponse( umatrix, umats_RESP_add, Nelectrons, omega, eta, numBathOrbs, 'A', 1 ) # At most 1 iteration and do not overwrite umats_RESP
            EperSite_rem, GF_rem, dump = theDMET.SolveResponse( umatrix, umats_RESP_rem, Nelectrons, omega, eta, numBathOrbs, 'R', 1 )
        SpectralFunction = - 2.0 * ( GF_add.imag + GF_rem.imag ) / math.pi # Factor of 2 due to summation over spin projection
        LDOS.append( SpectralFunction )
        print "LDOS( U =",HubbardU,"; omega =",omega,") self-consistent =",SpectralFunction
    
    return LDOS
    
