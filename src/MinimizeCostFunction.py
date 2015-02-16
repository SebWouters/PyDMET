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
import LinalgWrappers
import DMETorbitals
from scipy.optimize import minimize

def UmatFlat2SquareGS( umatflat, lindim ):

    umatsquare = np.zeros( [ lindim, lindim ], dtype=float )
    for row in range(0, lindim):
        umatsquare[row, row] = umatflat[0] # All diagonal elements are forced to be the same
        for col in range(row+1, lindim):
            umatsquare[row, col] = umatflat[ 1 + row + ( col * ( col - 1 ) ) / 2 ]
            umatsquare[col, row] = umatsquare[row, col]
    return umatsquare
            
def UmatSquare2FlatGS( umatsquare, lindim ):

    umatflat = np.zeros( [ 1 + ( lindim * ( lindim - 1 ) ) / 2 ], dtype=float )
    umatflat[0] = umatsquare[0, 0] # All diagonal elements are forced to be the same
    for row in range(0, lindim):
        for col in range(row+1, lindim):
            umatflat[ 1 + row + ( col * ( col - 1 ) ) / 2 ] = umatsquare[row, col]
    return umatflat
    
'''def UmatFlat2SquareRESP( umatflat, lindim ):

    umatsquare = np.zeros( [ lindim, lindim ], dtype=float )
    for row in range(0, lindim):
        for col in range(row, lindim):
            umatsquare[row, col] = umatflat[ row + ( col * ( col + 1 ) ) / 2 ]
            umatsquare[col, row] = umatsquare[row, col]
    return umatsquare
            
def UmatSquare2FlatRESP( umatsquare, lindim ):

    umatflat = np.zeros( [ ( lindim * ( lindim + 1 ) ) / 2 ], dtype=float )
    for row in range(0, lindim):
        for col in range(row, lindim):
            umatflat[ row + ( col * ( col + 1 ) ) / 2 ] = umatsquare[row, col]
    return umatflat'''

def OneRDMdifferencesGS( umatsquare, GS_1RDMs, HamDMETs, numRDMs, numPairs ):

    rdmDifferences = []
    for cnt in range( numRDMs ):
        HamDMETs[cnt].OverwriteImpurityUmat( umatsquare )
        energiesRHF, solutionRHF = LinalgWrappers.RestrictedHartreeFock( HamDMETs[cnt].Tmat, numPairs )
        GS_1RDM_Slater = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
        errorGS = GS_1RDMs[cnt] - GS_1RDM_Slater
        assert( abs( np.trace(errorGS) ) < 1e-8 ) # If not OK, the particle number is wrong in one of the two 1-RDMs
        rdmDifferences.append( errorGS )
    return rdmDifferences

def CostFunctionGS( umatflat, GS_1RDMs, HamDMETs, numRDMs, numPairs ):

    umatsquare = UmatFlat2SquareGS( umatflat, HamDMETs[0].numImpOrbs )
    rdmDifferences = OneRDMdifferencesGS( umatsquare, GS_1RDMs, HamDMETs, numRDMs, numPairs )
    error = 0
    for cnt in range( numRDMs ):
        error += np.linalg.norm( rdmDifferences[cnt] )**2
    return error
    
def MinimizeGS( umat_guess, GS_1RDMs, HamDMETs, numRDMs, NelecActiveSpace ):

    umatflat = UmatSquare2FlatGS( umat_guess, HamDMETs[0].numImpOrbs )

    assert( NelecActiveSpace % 2 == 0 )
    numPairs = NelecActiveSpace / 2
    result = minimize( CostFunctionGS, umatflat, args=(GS_1RDMs, HamDMETs, numRDMs, numPairs), options={'disp': False} )
    if ( result.success==False ):
        print "   Minimize ::",result.message
    print "   MinimizeGS :: Cost function after convergence =",result.fun
    
    umatsquare = UmatFlat2SquareGS( result.x, HamDMETs[0].numImpOrbs )
    return umatsquare
    
def RESP_1RDM_difference( umatsquare, RESP_1RDM, HamDMET, orb_i, numPairs, omega, eta, toSolve, normalizedRDMs ):

    HamDMET.OverwriteImpurityUmat( umatsquare )
    energiesRHF, solutionRHF = LinalgWrappers.RestrictedHartreeFock( HamDMET.Tmat, numPairs )
    RDM_0 = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
    if (toSolve=='A'):
        RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_addition( orb_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
    if (toSolve=='R'):
        RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_removal(  orb_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
    if (toSolve=='F'):
        RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_forward(  orb_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
    if (toSolve=='B'):
        RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_backward( orb_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
    if ( normalizedRDMs == True ):
        errorRESP = RESP_1RDM - ( S_R * RDM_R + S_I * RDM_I ) / ( S_R + S_I )
    else:
        errorRESP = RESP_1RDM - ( S_R * RDM_R + S_I * RDM_I )
    return errorRESP
    
def CostFunctionResponse( umatflat, RESP_1RDM, HamDMET, orb_i, numPairs, omega, eta, toSolve, normalizedRDMs ):

    umatsquare = UmatFlat2SquareGS( umatflat, HamDMET.numImpOrbs )
    errorRESP = RESP_1RDM_difference( umatsquare, RESP_1RDM, HamDMET, orb_i, numPairs, omega, eta, toSolve, normalizedRDMs )
    totalError = np.linalg.norm( errorRESP )**2
    return totalError
    
def RespNORMAL( umat_guess, RESP_1RDM, HamDMET, orb_i, NelecActiveSpace, omega, eta, toSolve, normalizedRDMs ):

    umatflat = UmatSquare2FlatGS( umat_guess, HamDMET.numImpOrbs )

    assert( NelecActiveSpace % 2 == 0 )
    numPairs = NelecActiveSpace / 2
    #boundaries = []
    #for element in umatflat:
    #    boundaries.append( ( element-(1 + np.random.uniform(-0.1, 0.1)), element+(1 + np.random.uniform(-0.1,0.1)) ) )
    #result = minimize( CostFunctionResponse, umatflat, args=(RESP_1RDM, HamDMET, orb_i, numPairs, omega, eta, toSolve, normalizedRDMs), method='L-BFGS-B', bounds=boundaries, options={'disp': False} )
    result = minimize( CostFunctionResponse, umatflat, args=(RESP_1RDM, HamDMET, orb_i, numPairs, omega, eta, toSolve, normalizedRDMs), options={'disp': False} )
    if ( result.success==False ):
        print "   Minimize ::",result.message
    print "   MinimizeRESP :: Cost function after convergence =",result.fun

    umatsquare = UmatFlat2SquareGS( result.x, HamDMET.numImpOrbs )
    return umatsquare
    
def RESP_1RDM_differences_DDMRG( umatsquare, ED_RDM_A, ED_RDM_R, ED_RDM_I, HamDMET, orb_i, numPairs, omega, eta, toSolve, normalizedRDMs ):

    HamDMET.OverwriteImpurityUmat( umatsquare )
    energiesRHF, solutionRHF = LinalgWrappers.RestrictedHartreeFock( HamDMET.Tmat, numPairs )
    RDM_0 = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
    if (toSolve=='A'):
        RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_addition( orb_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
    if (toSolve=='R'):
        RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_removal(  orb_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
    if (toSolve=='F'):
        RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_forward(  orb_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
    if (toSolve=='B'):
        RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_backward( orb_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
    if ( normalizedRDMs == True ):
        errorA = ED_RDM_A - RDM_A
        errorR = ED_RDM_R - RDM_R
        errorI = ED_RDM_I - RDM_I
    else:
        errorA = ED_RDM_A - S_A * RDM_A
        errorR = ED_RDM_R - S_R * RDM_R
        errorI = ED_RDM_I - S_I * RDM_I
    return ( errorA, errorR, errorI )
    
def CostFunctionResponseDDMRG( umatflat, ED_RDM_A, ED_RDM_R, ED_RDM_I, HamDMET, orb_i, numPairs, omega, eta, toSolve, normalizedRDMs, errorType ):

    umatsquare = UmatFlat2SquareGS( umatflat, HamDMET.numImpOrbs )
    errorA, errorR, errorI = RESP_1RDM_differences_DDMRG( umatsquare, ED_RDM_A, ED_RDM_R, ED_RDM_I, HamDMET, orb_i, numPairs, omega, eta, toSolve, normalizedRDMs )
    if ( errorType == 1 ):
        totalError = np.linalg.norm( errorA )**2 + np.linalg.norm( errorR )**2 + np.linalg.norm( errorI )**2
    else:
        totalError = np.linalg.norm( errorA + errorR + errorI )**2
    return totalError
    
def RespDDMRG( umat_guess, ED_RDM_A, ED_RDM_R, ED_RDM_I, HamDMET, orb_i, NelecActiveSpace, omega, eta, toSolve, normalizedRDMs, errorType ):

    umatflat = UmatSquare2FlatGS( umat_guess, HamDMET.numImpOrbs )

    assert( NelecActiveSpace % 2 == 0 )
    numPairs = NelecActiveSpace / 2
    #boundaries = []
    #for element in umatflat:
    #    boundaries.append( ( element-(1 + np.random.uniform(-0.1, 0.1)), element+(1 + np.random.uniform(-0.1,0.1)) ) )
    #result = minimize( CostFunctionResponseDDMRG, umatflat, args=(ED_RDM_A, ED_RDM_R, ED_RDM_I, HamDMET, orb_i, numPairs, omega, eta, toSolve, normalizedRDMs, errorType), method='L-BFGS-B', bounds=boundaries, options={'disp': False} )
    result = minimize( CostFunctionResponseDDMRG, umatflat, args=(ED_RDM_A, ED_RDM_R, ED_RDM_I, HamDMET, orb_i, numPairs, omega, eta, toSolve, normalizedRDMs, errorType), options={'disp': False} )
    if ( result.success==False ):
        print "   Minimize ::",result.message
    print "   MinimizeRESP :: Cost function after convergence =",result.fun

    umatsquare = UmatFlat2SquareGS( result.x, HamDMET.numImpOrbs )
    return umatsquare
    
    
