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
import math as m
    
def Construct1RDM_groundstate( solutionRHF, numPairs ):

    DoublyOccOrbs = solutionRHF[:, 0:numPairs]
    GS_OneRDM = 2 * np.dot( DoublyOccOrbs , DoublyOccOrbs.T )
    return GS_OneRDM
    
def Construct1RDM_addition( orbital_i, omega, eta, eigsRHF, solutionRHF, groundstate1RDM, numPairs ):

    C_i_virt = np.matrix( solutionRHF[ orbital_i , numPairs: ] )
    if (C_i_virt.shape[0] > 1):
        C_i_virt = C_i_virt.T # Now certainly row-like matrix (shape = 1 x len(C_i_virt))
    
    Overlap_A = np.dot( C_i_virt , C_i_virt.T )[0,0]
    Vector_A = np.matrix( np.dot( solutionRHF[ : , numPairs: ] , C_i_virt.T ) )
    if (Vector_A.shape[0] > 1):
        Vector_A = Vector_A.T
    RDM_add_A = groundstate1RDM + np.dot( Vector_A.T , Vector_A ) / Overlap_A
    
    center_R = omega - eigsRHF[ numPairs: ] # (omega - eps_virt)
    center_R = np.multiply( center_R , 1.0 / ( np.multiply( center_R , center_R ) + eta*eta ) ) # (omega - eps_virt) / [ (omega - eps_virt)^2 + eta^2 ]
    C_i_virt_R = np.matrix( np.multiply( C_i_virt, center_R ) )
    if (C_i_virt_R.shape[0] > 1):
        C_i_virt_R = C_i_virt_R.T
    Overlap_R = np.dot( C_i_virt_R, C_i_virt_R.T )[0,0]
    Vector_R = np.matrix( np.dot( solutionRHF[ : , numPairs: ] , C_i_virt_R.T ) )
    if (Vector_R.shape[0] > 1):
        Vector_R = Vector_R.T
    RDM_add_R = groundstate1RDM + np.dot( Vector_R.T , Vector_R ) / Overlap_R
    
    center_I = omega - eigsRHF[ numPairs: ] # (omega - eps_virt)
    center_I = eta / ( np.multiply( center_I , center_I ) + eta*eta ) # eta / [ (omega - eps_virt)^2 + eta^2 ]
    C_i_virt_I = np.matrix( np.multiply( C_i_virt, center_I ) )
    if (C_i_virt_I.shape[0] > 1):
        C_i_virt_I = C_i_virt_I.T
    Overlap_I = np.dot( C_i_virt_I, C_i_virt_I.T )[0,0]
    Vector_I = np.matrix( np.dot( solutionRHF[ : , numPairs: ] , C_i_virt_I.T ) )
    if (Vector_I.shape[0] > 1):
        Vector_I = Vector_I.T
    RDM_add_I = groundstate1RDM + np.dot( Vector_I.T , Vector_I ) / Overlap_I
    
    return ( RDM_add_A, RDM_add_R, RDM_add_I, Overlap_A, Overlap_R, Overlap_I )
    
def Construct1RDM_removal( orbital_i, omega, eta, eigsRHF, solutionRHF, groundstate1RDM, numPairs ):

    C_i_occ = np.matrix( solutionRHF[ orbital_i , :numPairs ] )
    if (C_i_occ.shape[0] > 1):
        C_i_occ = C_i_occ.T # Now certainly row-like matrix (shape = 1 x len(C_i_occ))
    
    Overlap_A = np.dot( C_i_occ , C_i_occ.T )[0,0]
    Vector_A = np.matrix( np.dot( solutionRHF[ : , :numPairs ] , C_i_occ.T ) )
    if (Vector_A.shape[0] > 1):
        Vector_A = Vector_A.T
    RDM_rem_A = groundstate1RDM - np.dot( Vector_A.T , Vector_A ) / Overlap_A # Minus sign!!
    
    center_R = omega - eigsRHF[ :numPairs ] # (omega - eps_occ)
    center_R = np.multiply( center_R , 1.0 / ( np.multiply( center_R , center_R ) + eta*eta ) ) # (omega - eps_occ) / [ (omega - eps_occ)^2 + eta^2 ]
    C_i_occ_R = np.matrix( np.multiply( C_i_occ, center_R ) )
    if (C_i_occ_R.shape[0] > 1):
        C_i_occ_R = C_i_occ_R.T
    Overlap_R = np.dot( C_i_occ_R, C_i_occ_R.T )[0,0]
    Vector_R = np.matrix( np.dot( solutionRHF[ : , :numPairs ] , C_i_occ_R.T ) )
    if (Vector_R.shape[0] > 1):
        Vector_R = Vector_R.T
    RDM_rem_R = groundstate1RDM - np.dot( Vector_R.T , Vector_R ) / Overlap_R # Minus sign!!
    
    center_I = omega - eigsRHF[ :numPairs ] # (omega - eps_occ)
    center_I = eta / ( np.multiply( center_I , center_I ) + eta*eta ) # eta / [ (omega - eps_occ)^2 + eta^2 ]
    C_i_occ_I = np.matrix( np.multiply( C_i_occ, center_I ) )
    if (C_i_occ_I.shape[0] > 1):
        C_i_occ_I = C_i_occ_I.T
    Overlap_I = np.dot( C_i_occ_I, C_i_occ_I.T )[0,0]
    Vector_I = np.matrix( np.dot( solutionRHF[ : , :numPairs ] , C_i_occ_I.T ) )
    if (Vector_I.shape[0] > 1):
        Vector_I = Vector_I.T
    RDM_rem_I = groundstate1RDM - np.dot( Vector_I.T , Vector_I ) / Overlap_I # Minus sign!!
    
    return ( RDM_rem_A, RDM_rem_R, RDM_rem_I, Overlap_A, Overlap_R, Overlap_I )
    
def ConstructMeanFieldLDOS( orbital_i, omega, eta, eigsRHF, solutionRHF, numPairs ):

    # GF = 2 * \sum_{alpha all} C_{i,alpha} C^+_{alpha,i} / ( omega - epsilon_alpha + I*eta)
    GreenFunction = 2.0 * np.sum( np.multiply( 1.0 / ( omega - eigsRHF[ : ] + 1j*eta ) , np.square( solutionRHF[ orbital_i, : ] ) ) )
    LDOS = - GreenFunction.imag / m.pi
    return LDOS
    
def Construct1RDM_forward( orbital_i, omega, eta, eigsRHF, solutionRHF, groundstate1RDM, numPairs ):
    
    C_i_virt = np.matrix( solutionRHF[ orbital_i , numPairs: ] )
    if (C_i_virt.shape[0] > 1):
        C_i_virt = C_i_virt.T # Now certainly row-like matrix (shape = 1 x len(C_i_virt))
    
    C_i_occ = np.matrix( solutionRHF[ orbital_i , :numPairs ] )
    if (C_i_occ.shape[0] > 1):
        C_i_occ = C_i_occ.T
    
    II_virt = np.dot( C_i_virt , C_i_virt.T )[0,0]
    TwoII_occ = groundstate1RDM[ orbital_i, orbital_i ]
    Overlap_A = TwoII_occ * II_virt
    Vector_A = np.matrix( np.dot( solutionRHF[ : , numPairs: ] , C_i_virt.T ) )
    if (Vector_A.shape[0] > 1):
        Vector_A = Vector_A.T
    Vector_A2 = np.matrix( groundstate1RDM[ : , orbital_i ] )
    if (Vector_A2.shape[0] > 1):
        Vector_A2 = Vector_A2.T
    RDM_fw_A = groundstate1RDM + ( np.dot( Vector_A.T , Vector_A ) * TwoII_occ - 0.5 * II_virt * np.dot( Vector_A2.T , Vector_A2 ) ) / Overlap_A
    
    center_R = np.zeros([ C_i_virt.shape[1] , C_i_occ.shape[1] ], dtype=float)
    for virt in range( C_i_virt.shape[1] ):
        for occ in range( C_i_occ.shape[1] ):
            base = omega - ( eigsRHF[ numPairs + virt ] - eigsRHF[ occ ] )
            central = base / ( base * base + eta * eta )
            center_R[ virt, occ ] = solutionRHF[ orbital_i, numPairs + virt ] * central * solutionRHF[ orbital_i, occ ]
    Overlap_R = 2 * np.sum( np.multiply( center_R, center_R ) )
    RDM_fw_R = groundstate1RDM \
             + 2 * ( np.dot( np.dot( solutionRHF[ :, numPairs: ], np.dot( center_R , center_R.T ) ) , ( solutionRHF[ :, numPairs: ] ).T ) \
                   - np.dot( np.dot( solutionRHF[ :, :numPairs ], np.dot( center_R.T , center_R ) ) , ( solutionRHF[ :, :numPairs ] ).T ) ) / Overlap_R
    
    center_I = np.zeros([ C_i_virt.shape[1] , C_i_occ.shape[1] ], dtype=float)
    for virt in range( C_i_virt.shape[1] ):
        for occ in range( C_i_occ.shape[1] ):
            base = omega - ( eigsRHF[ numPairs + virt ] - eigsRHF[ occ ] )
            central = eta / ( base * base + eta * eta )
            center_I[ virt, occ ] = solutionRHF[ orbital_i, numPairs + virt ] * central * solutionRHF[ orbital_i, occ ]
    Overlap_I = 2 * np.sum( np.multiply( center_I, center_I ) )
    RDM_fw_I = groundstate1RDM \
             + 2 * ( np.dot( np.dot( solutionRHF[ :, numPairs: ], np.dot( center_I , center_I.T ) ) , ( solutionRHF[ :, numPairs: ] ).T ) \
                   - np.dot( np.dot( solutionRHF[ :, :numPairs ], np.dot( center_I.T , center_I ) ) , ( solutionRHF[ :, :numPairs ] ).T ) ) / Overlap_I
    
    return ( RDM_fw_A, RDM_fw_R, RDM_fw_I, Overlap_A, Overlap_R, Overlap_I )
    
def Construct1RDM_backward( orbital_i, omega, eta, eigsRHF, solutionRHF, groundstate1RDM, numPairs ):

    return Construct1RDM_forward( orbital_i, -omega, -eta, eigsRHF, solutionRHF, groundstate1RDM, numPairs )
    
def ConstructBathOrbitals( impurityOrbs, OneRDM, numBathOrbs ):

    embeddingOrbs = 1 - impurityOrbs
    embeddingOrbs = np.matrix( embeddingOrbs )
    if (embeddingOrbs.shape[0] > 1):
        embeddingOrbs = embeddingOrbs.T # Now certainly row-like matrix (shape = 1 x len(vector))
    isEmbedding = np.dot( embeddingOrbs.T , embeddingOrbs ) == 1
    numEmbedOrbs = np.sum( embeddingOrbs )
    embedding1RDM = np.reshape( OneRDM[ isEmbedding ], ( numEmbedOrbs , numEmbedOrbs ) )

    numImpOrbs = np.sum( impurityOrbs )
    numTotalOrbs = len( impurityOrbs )
        
    eigenvals, eigenvecs = np.linalg.eigh( embedding1RDM )
    idx = np.maximum( -eigenvals, eigenvals - 2.0 ).argsort() # Occupation numbers closest to 1 come first
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]
    pureEnvironEigVals = -eigenvals[numBathOrbs:]
    pureEnvironEigVecs = eigenvecs[:,numBathOrbs:]
    idx = pureEnvironEigVals.argsort()
    eigenvecs[:,numBathOrbs:] = pureEnvironEigVecs[:,idx]
    pureEnvironEigVals = -pureEnvironEigVals
    DiscardOccupation = np.sum( np.fabs( np.minimum( pureEnvironEigVals, 2.0 - pureEnvironEigVals ) ) )
    NelecEnvironment  = np.sum( pureEnvironEigVals ) # Number of electrons which are not on impurity or bath orbitals (float!)
    
    for counter in range(0, numImpOrbs):
        eigenvecs = np.insert(eigenvecs, counter, 0.0, axis=1) #Stack columns with zeros in the beginning
    counter = 0
    for counter2 in range(0, numTotalOrbs):
        if ( impurityOrbs[counter2] ):
            eigenvecs = np.insert(eigenvecs, counter2, 0.0, axis=0) #Stack rows with zeros on locations of the impurity orbitals
            eigenvecs[counter2, counter] = 1.0
            counter += 1
    assert( counter == numImpOrbs )
    
    # Orthonormality is guaranteed due to (1) stacking with zeros and (2) orthonormality eigenvecs for symmetric matrix
    assert( np.linalg.norm( np.dot(eigenvecs.T, eigenvecs) - np.identity(numTotalOrbs) ) < 1e-12 )

    # eigenvecs[ : , 0:numImpOrbs ]                      = impurity orbitals
    # eigenvecs[ : , numImpOrbs:numImpOrbs+numBathOrbs ] = bath orbitals
    # eigenvecs[ : , numImpOrbs+numBathOrbs: ]           = pure environment orbitals in decreasing order of occupation number
    return ( eigenvecs, NelecEnvironment, DiscardOccupation )

