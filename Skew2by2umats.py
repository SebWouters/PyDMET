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

def getU6toU14(HubbardU):

    Uvalues   = np.arange(6.0,14.5,1.0)
    umatrices = []

    umat_14 = np.array([[  6.99984384e+00 , -4.61919762e+00 , -2.67357343e-04 , 1.34211945e+00],[ -4.61919762e+00 ,  6.99984384e+00 , -6.45005348e-01 ,  1.14279907e-04],[ -2.67357343e-04 , -6.45005348e-01 ,  6.99984384e+00 , -4.72032569e+00],[  1.34211945e+00 ,  1.14279907e-04 , -4.72032569e+00 ,  6.99984384e+00]])
    umat_13 = np.array([[  6.49982634e+00 , -4.09157081e+00  , 4.24886075e-04 ,  1.20414851e+00], [ -4.09157081e+00 ,  6.49982634e+00 , -5.38391200e-01 ,  2.11957062e-04], [  4.24886075e-04 , -5.38391200e-01 ,  6.49982634e+00 , -4.21699346e+00], [  1.20414851e+00 ,  2.11957062e-04 , -4.21699346e+00 ,  6.49982634e+00]])
    umat_12 = np.array([[  6.00116098e+00 , -3.53872700e+00 , -1.07920334e-04 ,  1.05523762e+00],  [ -3.53872700e+00 ,  6.00116098e+00 , -4.37718152e-01 , -1.13920426e-04],  [ -1.07920334e-04 , -4.37718152e-01 ,  6.00116098e+00 , -3.69729220e+00],  [  1.05523762e+00 , -1.13920426e-04 , -3.69729220e+00 ,  6.00116098e+00]])
    umat_11 = np.array([[  5.50012542e+00 , -2.95714776e+00 ,  9.92420010e-05 ,  8.98462292e-01], [ -2.95714776e+00 ,  5.50012542e+00 , -3.40534737e-01 , -5.00991768e-05], [  9.92420010e-05 , -3.40534737e-01 ,  5.50012542e+00 , -3.16494258e+00], [  8.98462292e-01 , -5.00991768e-05 , -3.16494258e+00 ,  5.50012542e+00]])
    umat_10 = np.array([[  4.99993638e+00 , -2.32459319e+00 ,  3.52535810e-05 ,  7.28640488e-01],  [ -2.32459319e+00 ,  4.99993638e+00 , -2.47549298e-01 , -4.69359572e-05],  [  3.52535810e-05 , -2.47549298e-01 ,  4.99993638e+00 , -2.61035757e+00],  [  7.28640488e-01 , -4.69359572e-05 , -2.61035757e+00 ,  4.99993638e+00]])
    umat_9 = np.array([[  4.50003865e+00 , -1.54898321e+00 , -6.54274785e-06 ,  5.17674696e-01], [ -1.54898321e+00 ,  4.50003865e+00 , -1.56451599e-01, -3.08497889e-06], [ -6.54274785e-06 , -1.56451599e-01 ,  4.50003865e+00 , -1.98031801e+00],  [  5.17674696e-01 , -3.08497889e-06 , -1.98031801e+00 ,  4.50003865e+00]])
    umat_8 = np.array([[  4.00000123e+00 , -2.16933306e-01 , -1.12507993e-05 ,  9.96969206e-02], [ -2.16933306e-01 ,  4.00000123e+00 , -1.37128943e-01 , -1.44486592e-05], [ -1.12507993e-05 , -1.37128943e-01 ,  4.00000123e+00 , -6.01707731e-01], [  9.96969206e-02 , -1.44486592e-05 , -6.01707731e-01 ,  4.00000123e+00]])
    umat_7 = np.array([[  3.49999741e+00 , -1.89488241e-01 , -4.39727640e-06 ,  8.48896579e-02], [ -1.89488241e-01,   3.49999741e+00,  -7.52697212e-02,  -6.92307302e-06], [ -4.39727640e-06,  -7.52697212e-02,   3.49999741e+00,  -3.86368973e-01], [  8.48896579e-02,  -6.92307302e-06,  -3.86368973e-01,   3.49999741e+00]])
    umat_6 = np.array([[  3.00001180e+00,  -9.32588107e-02,  -5.47250314e-06,   3.83227830e-02], [ -9.32588107e-02,   3.00001180e+00,  -4.97550715e-02,  -5.37901182e-06], [ -5.47250314e-06,  -4.97550715e-02,   3.00001180e+00,  -1.43645240e-01], [  3.83227830e-02,  -5.37901182e-06,  -1.43645240e-01,   3.00001180e+00]])
    
    umatrices.append(umat_6)
    umatrices.append(umat_7)
    umatrices.append(umat_8)
    umatrices.append(umat_9)
    umatrices.append(umat_10)
    umatrices.append(umat_11)
    umatrices.append(umat_12)
    umatrices.append(umat_13)
    umatrices.append(umat_14)
    
    index = 0
    for count in range(1, len(Uvalues)):
        if abs( HubbardU - Uvalues[index] ) > abs( HubbardU - Uvalues[count] ):
            index = count
            
    return umatrices[index]
    
    
    
