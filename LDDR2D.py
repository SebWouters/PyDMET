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

def CalculateLDDR( HubbardU, Omegas, eta ):

    LDDR = []

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
    
    for omega in Omegas:

        EperSite_FW, GF_FW, notSC_GF_FW = theDMET.SolveResponse( umatrix, Nelectrons, omega, eta, numBathOrbs, 'F' )
        EperSite_BW, GF_BW, notSC_GF_BW = theDMET.SolveResponse( umatrix, Nelectrons, omega, eta, numBathOrbs, 'B' )
        SpectralFunction       = - ( GF_FW.imag - GF_BW.imag ) / math.pi
        SpectralFunction_notSC = - ( notSC_GF_FW.imag - notSC_GF_BW.imag ) / math.pi
        LDDR.append( SpectralFunction )
        print "LDDR( U =",HubbardU,"; omega =",omega,") NOT self-consistent =",SpectralFunction_notSC
        print "LDDR( U =",HubbardU,"; omega =",omega,")     self-consistent =",SpectralFunction
    
    return LDDR
    
