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

def UmatFlat2Square( umatflat, lindim ):

    umatsquare = np.zeros( [ lindim, lindim ], dtype=float )
    for row in range(0, lindim):
        umatsquare[row, row] = umatflat[0] # All diagonal elements are forced to be the same
        for col in range(row+1, lindim):
            umatsquare[row, col] = umatflat[ 1 + row + ( col * ( col - 1 ) ) / 2 ]
            umatsquare[col, row] = umatsquare[row, col]
    return umatsquare
            
def UmatSquare2Flat( umatsquare, lindim ):

    umatflat = np.zeros( [ 1 + ( lindim * ( lindim - 1 ) ) / 2 ], dtype=float )
    umatflat[0] = umatsquare[0, 0] # All diagonal elements are forced to be the same
    for row in range(0, lindim):
        for col in range(row+1, lindim):
            umatflat[ 1 + row + ( col * ( col - 1 ) ) / 2 ] = umatsquare[row, col]
    return umatflat

def GS_1RDMdifference( umatsquare, GS_1RDM, HamDMET, numPairs ):

    HamDMET.OverwriteImpurityUmat( umatsquare )
    energiesRHF, solutionRHF = LinalgWrappers.RestrictedHartreeFock( HamDMET.Tmat, numPairs )
    
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
    
def RESP_1RDM_differences( umatsquare, GS_1RDMs, RESP_1RDMs, HamDMETs, numPairs, omega, eta, toSolve, normalizedRDMs ):

    errorsGS   = []
    errorsRESP = []
    for orbital_i in range(0, HamDMETs[0].numImpOrbs):
        HamDMETs[ orbital_i ].OverwriteImpurityUmat( umatsquare )
        energiesRHF, solutionRHF = LinalgWrappers.RestrictedHartreeFock( HamDMETs[ orbital_i ].Tmat, numPairs )
        GS_1RDM_Slater = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
        errorsGS.append( GS_1RDMs[ orbital_i ] - GS_1RDM_Slater )
        if (toSolve=='A'):
            RESP_1RDM_Slater, overlap = DMETorbitals.Construct1RDM_addition( orbital_i, omega, eta, energiesRHF, solutionRHF, GS_1RDM_Slater, numPairs )
        if (toSolve=='R'):
            RESP_1RDM_Slater, overlap = DMETorbitals.Construct1RDM_removal(  orbital_i, omega, eta, energiesRHF, solutionRHF, GS_1RDM_Slater, numPairs )
        if (toSolve=='F'):
            RESP_1RDM_Slater, overlap = DMETorbitals.Construct1RDM_forward(  orbital_i, omega, eta, energiesRHF, solutionRHF, GS_1RDM_Slater, numPairs )
        if (toSolve=='B'):
            RESP_1RDM_Slater, overlap = DMETorbitals.Construct1RDM_backward( orbital_i, omega, eta, energiesRHF, solutionRHF, GS_1RDM_Slater, numPairs )
        if ( normalizedRDMs == True ):
            errorsRESP.append( RESP_1RDMs[ orbital_i ] - RESP_1RDM_Slater )
        else:
            errorsRESP.append( RESP_1RDMs[ orbital_i ] - overlap * RESP_1RDM_Slater )
    return ( errorsGS, errorsRESP )
    
def CostFunctionResponse( umatflat, GS_1RDMs, RESP_1RDMs, HamDMETs, numPairs, omega, eta, toSolve, normalizedRDMs, errorType ):

    umatsquare = UmatFlat2Square( umatflat, HamDMETs[0].numImpOrbs )
    errorsGS, errorsRESP = RESP_1RDM_differences( umatsquare, GS_1RDMs, RESP_1RDMs, HamDMETs, numPairs, omega, eta, toSolve, normalizedRDMs )
    totalError = 0.0
    for orbital_i in range(0, HamDMETs[0].numImpOrbs):
        if ( errorType == 1 ):
            totalError += np.linalg.norm( errorsGS[ orbital_i ] )**2 + np.linalg.norm( errorsRESP[ orbital_i ] )**2
        else:
            totalError += np.linalg.norm( errorsGS[ orbital_i ] + errorsRESP[ orbital_i ] )**2
    return totalError
    
def MinimizeResponse( umat_guess, GS_1RDMs, RESP_1RDMs, HamDMETs, NelecActiveSpace, omega, eta, toSolve, maxdelta, normalizedRDMs, errorType ):

    umatflat = UmatSquare2Flat( umat_guess, HamDMETs[0].numImpOrbs )

    assert( NelecActiveSpace % 2 == 0 )
    numPairs = NelecActiveSpace / 2
    boundaries = []
    for element in umatflat:
        boundaries.append( ( element-maxdelta*(1 + np.random.uniform(-0.1, 0.1)), element+maxdelta*(1 + np.random.uniform(-0.1,0.1)) ) )
    result = minimize( CostFunctionResponse, umatflat, args=(GS_1RDMs, RESP_1RDMs, HamDMETs, numPairs, omega, eta, toSolve, normalizedRDMs, errorType), method='L-BFGS-B', bounds=boundaries, options={'disp': False} )
    if ( result.success==False ):
        print "   Minimize ::",result.message
    print "   Minimize :: Cost function after convergence =",result.fun

    umatsquare = UmatFlat2Square( result.x, HamDMETs[0].numImpOrbs )
    return umatsquare
    
def RESP_1RDM_differences_DDMRG( umatsquare, ED_RDM_0, ED_RDM_A, ED_RDM_R, ED_RDM_I, HamDMETs, numPairs, omega, eta, toSolve, normalizedRDMs ):

    errors0 = []
    errorsA = []
    errorsR = []
    errorsI = []
    for orbital_i in range(0, HamDMETs[0].numImpOrbs):
        HamDMETs[ orbital_i ].OverwriteImpurityUmat( umatsquare )
        energiesRHF, solutionRHF = LinalgWrappers.RestrictedHartreeFock( HamDMETs[ orbital_i ].Tmat, numPairs )
        RDM_0 = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
        if (toSolve=='A'):
            RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_addition_bis( orbital_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
        if (toSolve=='R'):
            RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_removal_bis(  orbital_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
        if (toSolve=='F'):
            RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_forward_bis(  orbital_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
        if (toSolve=='B'):
            RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_backward_bis( orbital_i, omega, eta, energiesRHF, solutionRHF, RDM_0, numPairs )
        errors0.append( ED_RDM_0[ orbital_i ] - RDM_0 )
        if ( normalizedRDMs == True ):
            errorsA.append( ED_RDM_A[ orbital_i ] - RDM_A )
            errorsR.append( ED_RDM_R[ orbital_i ] - RDM_R )
            errorsI.append( ED_RDM_I[ orbital_i ] - RDM_I )
        else:
            errorsA.append( ED_RDM_A[ orbital_i ] - S_A * RDM_A )
            errorsR.append( ED_RDM_R[ orbital_i ] - S_R * RDM_R )
            errorsI.append( ED_RDM_I[ orbital_i ] - S_I * RDM_I )
    return ( errors0, errorsA, errorsR, errorsI )
    
def CostFunctionResponse_DDMRG( umatflat, ED_RDM_0, ED_RDM_A, ED_RDM_R, ED_RDM_I, HamDMETs, numPairs, omega, eta, toSolve, normalizedRDMs, errorType ):

    umatsquare = UmatFlat2Square( umatflat, HamDMETs[0].numImpOrbs )
    errors0, errorsA, errorsR, errorsI = RESP_1RDM_differences_DDMRG( umatsquare, ED_RDM_0, ED_RDM_A, ED_RDM_R, ED_RDM_I, HamDMETs, numPairs, omega, eta, toSolve, normalizedRDMs )
    totalError = 0.0
    for orbital_i in range(0, HamDMETs[0].numImpOrbs):
        if ( errorType == 1 ):
            totalError += np.linalg.norm( errors0[ orbital_i ] )**2
            totalError += np.linalg.norm( errorsA[ orbital_i ] )**2
            totalError += np.linalg.norm( errorsR[ orbital_i ] )**2
            totalError += np.linalg.norm( errorsI[ orbital_i ] )**2
        else:
            totalError += np.linalg.norm( errors0[ orbital_i ] + errorsA[ orbital_i ] + errorsR[ orbital_i ] + errorsI[ orbital_i ] )**2
    return totalError
    
def MinimizeResponse_DDMRG( umat_guess, ED_RDM_0, ED_RDM_A, ED_RDM_R, ED_RDM_I, HamDMETs, NelecActiveSpace, omega, eta, toSolve, maxdelta, normalizedRDMs, errorType ):

    umatflat = UmatSquare2Flat( umat_guess, HamDMETs[0].numImpOrbs )

    assert( NelecActiveSpace % 2 == 0 )
    numPairs = NelecActiveSpace / 2
    boundaries = []
    for element in umatflat:
        boundaries.append( ( element-maxdelta*(1 + np.random.uniform(-0.1, 0.1)), element+maxdelta*(1 + np.random.uniform(-0.1,0.1)) ) )
    result = minimize( CostFunctionResponse_DDMRG, umatflat, args=(ED_RDM_0, ED_RDM_A, ED_RDM_R, ED_RDM_I, HamDMETs, numPairs, omega, eta, toSolve, normalizedRDMs, errorType), method='L-BFGS-B', bounds=boundaries, options={'disp': False} )
    if ( result.success==False ):
        print "   Minimize ::",result.message
    print "   Minimize :: Cost function after convergence =",result.fun

    umatsquare = UmatFlat2Square( result.x, HamDMETs[0].numImpOrbs )
    return umatsquare
    
    
