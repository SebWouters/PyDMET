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
import LinalgWrappers
import DMETorbitals
from scipy.optimize import minimize

def UmatFlat2Square( umatflat, lindim ):

    umatsquare = np.zeros( [ lindim, lindim ], dtype=float )
    for row in range(0, lindim):
        for col in range(row, lindim):
            umatsquare[row, col] = umatflat[ row + ( col * ( col + 1 ) ) / 2 ]
            umatsquare[col, row] = umatsquare[row, col]
    return umatsquare
            
def UmatSquare2Flat( umatsquare, lindim ):

    umatflat = np.zeros( [ ( lindim * ( lindim + 1 ) ) / 2 ], dtype=float )
    for row in range(0, lindim):
        for col in range(row, lindim):
            umatflat[ row + ( col * ( col + 1 ) ) / 2 ] = umatsquare[row, col]
    return umatflat

def GS_1RDMdifference( umatsquare, GS_1RDM, HamDMET, numPairs ):

    HamDMET.OverwriteImpurityUmat( umatsquare )
    energiesRHF, solutionRHF = LinalgWrappers.SortedEigSymmetric( HamDMET.Tmat )
    if ( energiesRHF[ numPairs ] - energiesRHF[ numPairs-1 ] < 1e-8 ):
        print "ERROR: The single particle gap is zero!"
        assert( energiesRHF[ numPairs ] - energiesRHF[ numPairs-1 ] >= 1e-8 )
    
    GS_1RDM_Slater = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
    errorGS = GS_1RDM - GS_1RDM_Slater
    assert( abs( np.trace(errorGS) ) < 1e-8 ) # If not OK, the particle number is wrong in one of the two 1-RDMs
    return errorGS

def CostFunction( umatflat, GS_1RDM, HamDMET, numPairs ):

    umatsquare = UmatFlat2Square( umatflat, HamDMET.numImpOrbs )
    error = GS_1RDMdifference( umatsquare, GS_1RDM, HamDMET, numPairs )
    return np.linalg.norm( error )**2
    
def Minimize( umat_guess, GS_1RDM, HamDMET, NelecActiveSpace ):

    umatflat = UmatSquare2Flat( umat_guess, HamDMET.numImpOrbs )

    assert( NelecActiveSpace % 2 == 0 )
    numPairs = NelecActiveSpace / 2
    result = minimize( CostFunction, umatflat, args=(GS_1RDM, HamDMET, numPairs), options={'disp': False} )
    if ( result.success==False ):
        print "   Minimize ::",result.message
    print "   Minimize :: Cost function after convergence =",result.fun
    
    umatsquare = UmatFlat2Square( result.x, HamDMET.numImpOrbs )
    return umatsquare
    
def RESP_1RDM_differences( umatsquare, GS_1RDM, RESP_1RDM, HamDMET, numPairs, orbital_i, omega, eta, toSolve ):

    HamDMET.OverwriteImpurityUmat( umatsquare )
    energiesRHF, solutionRHF = LinalgWrappers.SortedEigSymmetric( HamDMET.Tmat )
    if ( energiesRHF[ numPairs ] - energiesRHF[ numPairs-1 ] < 1e-8 ):
        print "ERROR: The single particle gap is zero!"
        assert( energiesRHF[ numPairs ] - energiesRHF[ numPairs-1 ] >= 1e-8 )

    GS_1RDM_Slater = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
    if (toSolve=='A'):
        RESP_1RDM_Slater = DMETorbitals.Construct1RDM_addition( orbital_i, omega, eta, energiesRHF, solutionRHF, numPairs )
    if (toSolve=='R'):
        RESP_1RDM_Slater = DMETorbitals.Construct1RDM_removal(  orbital_i, omega, eta, energiesRHF, solutionRHF, numPairs )
    if (toSolve=='F'):
        RESP_1RDM_Slater = DMETorbitals.Construct1RDM_forward(  orbital_i, omega, eta, energiesRHF, solutionRHF, numPairs )
    if (toSolve=='B'):
        RESP_1RDM_Slater = DMETorbitals.Construct1RDM_backward( orbital_i, omega, eta, energiesRHF, solutionRHF, numPairs )
    
    errorGS   = GS_1RDM   - GS_1RDM_Slater
    errorRESP = RESP_1RDM - RESP_1RDM_Slater
    assert( abs( np.trace(errorGS  ) ) < 1e-8 ) # If not OK, the particle number is wrong in one of the two 1-RDMs
    assert( abs( np.trace(errorRESP) ) < 1e-8 ) # If not OK, the particle number is wrong in one of the two 1-RDMs
    return ( errorGS, errorRESP )
    
def CostFunctionResponse( umatflat, GS_1RDM, RESP_1RDM, HamDMET, numPairs, orbital_i, omega, eta, toSolve, prefactResponseRDM ):

    umatsquare = UmatFlat2Square( umatflat, HamDMET.numImpOrbs )
    errorGS, errorRESP = RESP_1RDM_differences( umatsquare, GS_1RDM, RESP_1RDM, HamDMET, numPairs, orbital_i, omega, eta, toSolve )
    return ( 1.0 - prefactResponseRDM ) * np.linalg.norm( errorGS )**2 + prefactResponseRDM * np.linalg.norm( errorRESP )**2
    
def MinimizeResponse( umat_guess, GS_1RDM, RESP_1RDM, HamDMET, NelecActiveSpace, orbital_i, omega, eta, toSolve, prefactResponseRDM ):

    umatflat = UmatSquare2Flat( umat_guess, HamDMET.numImpOrbs )

    assert( NelecActiveSpace % 2 == 0 )
    numPairs = NelecActiveSpace / 2
    result = minimize( CostFunctionResponse, umatflat, args=(GS_1RDM, RESP_1RDM, HamDMET, numPairs, orbital_i, omega, eta, toSolve, prefactResponseRDM), options={'disp': False} )
    if ( result.success==False ):
        print "   Minimize ::",result.message
    print "   Minimize :: Cost function after convergence =",result.fun

    umatsquare = UmatFlat2Square( result.x, HamDMET.numImpOrbs )
    return umatsquare
    
    
