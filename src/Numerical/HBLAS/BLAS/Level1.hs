{-# LANGUAGE BangPatterns , RankNTypes, GADTs, DataKinds #-}

{- | The 'Numerical.HBLAS.BLAS.Level1' module provides a fully general
yet type safe Level1 BLAS API.

When in doubt about the semantics of an operation,
consult your system's BLAS api documentation, or just read the documentation
for
<https://software.intel.com/sites/products/documentation/hpc/mkl/mklman/index.htm the Intel MKL BLAS distribution>

A few basic notes about how to invoke BLAS routines.

Many BLAS operations take one or more arguments of type 'Transpose'.
'Tranpose' has the following different constructors, which tell BLAS
routines what transformation to implicitly apply to an input matrix @mat@ with dimension @n x m@.

*  'NoTranspose' leaves the matrix @mat@ as is.

* 'Transpose' treats the @mat@ as being implicitly transposed, with dimension
    @m x n@. Entry @mat(i,j)@ being treated as actually being the entry
    @mat(j,i)@. For Real matrices this is also the matrix adjoint operation.
    ie @Tranpose(mat)(i,j)=mat(j,i)@

*  'ConjNoTranspose' will implicitly conjugate @mat@, which is a no op for Real ('Float' or 'Double') matrices, but for
'Complex Float' and 'Complex Double' matrices, a given matrix entry @mat(i,j)==x':+'y@
will be treated as actually being  @conjugate(mat)(i,j)=y':+'x@.

* 'ConjTranpose' will implicitly transpose and conjugate the input matrix.
ConjugateTranpose acts as matrix adjoint for both real and complex matrices.



The *gemm operations  work as follows (using 'sgemm' as an example):

* @'sgemm trLeft trRight alpha beta left right result'@, where @trLeft@ and @trRight@
are values of type 'Transpose' that respectively act on the matrices @left@ and @right@.

* the generalized matrix computation thusly formed can be viewed as being
@result = alpha * trLeft(left) * trRight(right) + beta * result@


the *gemv operations are akin to the *gemm operations, but with @right@ and @result@
being vectors rather than matrices.


the *trsv operations solve for @x@ in the equation @A x = y@ given @A@ and @y@.
The 'MatUpLo' argument determines if the matrix should be treated as upper or
lower triangular and 'MatDiag' determines if the triangular solver should treat
the diagonal of the matrix as being all 1's or not.  A general pattern of invocation
would be @'strsv' matuplo  tranposeMatA  matdiag  matrixA  xVector@.
A key detail to note is that the input vector is ALSO the result vector,
ie 'strsv' and friends updates the vector place.

-}

module Numerical.HBLAS.BLAS.Level1(
        sasum
        ,dasum
        ,scasum
        ,dzasum

        ,saxpy
        ,daxpy
        ,caxpy
        ,zaxpy

        ,scopy
        ,dcopy
        ,ccopy
        ,zcopy

        ,sdot
        ,ddot
        ,sdsdot
        ,dsdot

        --,cdotu
        --,cdotc
        --,zdotu
        --,zdotc

        ,snrm2
        ,dnrm2
        ,scnrm2
        ,dznrm2

        ,srot
        ,drot

        {-
         -,srotg
         -,drotg
         -}

        {-
         -,srotm
         -,drotm
         -}

        {-
         -,srotmg
         -,drotmg
         -}

        ,sscal
        ,dscal
        ,cscal
        ,zscal
        ,csscal
        ,zdscal

        ,sswap
        ,dswap
        ,cswap
        ,zswap

        ,isamax
        ,idamax
        ,icamax
        ,izamax

            ) where


import Numerical.HBLAS.UtilsFFI
import Numerical.HBLAS.BLAS.FFI.Level1
import Numerical.HBLAS.BLAS.Internal.Level1
import Control.Monad.Primitive
import Data.Complex


sasum :: PrimMonad m => AsumFun Float (PrimState m) m Float
sasum = asumAbstraction "sasum" cblas_sasum_safe cblas_sasum_unsafe

dasum :: PrimMonad m => AsumFun Double (PrimState m) m Double
dasum = asumAbstraction "dasum" cblas_dasum_safe cblas_dasum_unsafe

scasum :: PrimMonad m => AsumFun (Complex Float) (PrimState m) m Float
scasum = asumAbstraction "scasum" cblas_scasum_safe cblas_scasum_unsafe

dzasum :: PrimMonad m => AsumFun (Complex Double) (PrimState m) m Double
dzasum = asumAbstraction "dzasum" cblas_dzasum_safe cblas_dzasum_unsafe

saxpy :: PrimMonad m => AxpyFun Float (PrimState m) m
saxpy = axpyAbstraction "saxpy" cblas_saxpy_safe cblas_saxpy_unsafe (\x f -> f x)

daxpy :: PrimMonad m => AxpyFun Double (PrimState m) m
daxpy = axpyAbstraction "daxpy" cblas_daxpy_safe cblas_daxpy_unsafe (\x f -> f x)

caxpy :: PrimMonad m => AxpyFun (Complex Float) (PrimState m) m
caxpy = axpyAbstraction "caxpy" cblas_caxpy_safe cblas_caxpy_unsafe withRStorable_

zaxpy :: PrimMonad m => AxpyFun (Complex Double) (PrimState m) m
zaxpy = axpyAbstraction "zaxpy" cblas_zaxpy_safe cblas_zaxpy_unsafe withRStorable_

scopy :: PrimMonad m => CopyFun Float (PrimState m) m
scopy = copyAbstraction "scopy" cblas_scopy_safe cblas_scopy_unsafe

dcopy :: PrimMonad m => CopyFun Double (PrimState m) m
dcopy = copyAbstraction "dcopy" cblas_dcopy_safe cblas_dcopy_unsafe

ccopy :: PrimMonad m => CopyFun (Complex Float) (PrimState m) m
ccopy = copyAbstraction "ccopy" cblas_ccopy_safe cblas_ccopy_unsafe

zcopy :: PrimMonad m => CopyFun (Complex Double) (PrimState m) m
zcopy = copyAbstraction "zcopy" cblas_zcopy_safe cblas_zcopy_unsafe

sdot :: PrimMonad m => NoScalarDotFun Float (PrimState m) m Float
sdot = noScalarDotAbstraction "sdot" cblas_sdot_safe cblas_sdot_unsafe

ddot :: PrimMonad m => NoScalarDotFun Double (PrimState m) m Double
ddot = noScalarDotAbstraction "ddot" cblas_ddot_safe cblas_ddot_unsafe

sdsdot :: PrimMonad m => ScalarDotFun Float (PrimState m) m Float
sdsdot = scalarDotAbstraction "sdsdot" cblas_sdsdot_safe cblas_sdsdot_unsafe withRStorable withRStorable

dsdot :: PrimMonad m => NoScalarDotFun Float (PrimState m) m Double
dsdot = noScalarDotAbstraction "dsdot" cblas_dsdot_safe cblas_dsdot_unsafe

{-
 -cdotu :: PrimMonad m => ComplexDotFun (Complex Float) (PrimState m) m
 -cdotu = complexDotAbstraction "cdotu" cblas_cdotu_safe cblas_cdotu_unsafe
 -
 -cdotc :: PrimMonad m => ComplexDotFun (Complex Float) (PrimState m) m
 -cdotc = complexDotAbstraction "cdotc" cblas_cdotc_safe cblas_cdotc_unsafe
 -
 -zdotu :: PrimMonad m => ComplexDotFun (Complex Double) (PrimState m) m
 -zdotu = complexDotAbstraction "zdotu" cblas_zdotu_safe cblas_zdotu_unsafe
 -
 -zdotc :: PrimMonad m => ComplexDotFun (Complex Double) (PrimState m) m
 -zdotc = complexDotAbstraction "zdotc" cblas_zdotc_safe cblas_zdotc_unsafe
 -}

snrm2 :: PrimMonad m => Nrm2Fun Float (PrimState m) m Float
snrm2 = norm2Abstraction "snrm2" cblas_snrm2_safe cblas_snrm2_unsafe

dnrm2 :: PrimMonad m => Nrm2Fun Double (PrimState m) m Double
dnrm2 = norm2Abstraction "dnrm2" cblas_dnrm2_safe cblas_dnrm2_unsafe

scnrm2 :: PrimMonad m => Nrm2Fun (Complex Float) (PrimState m) m Float
scnrm2 = norm2Abstraction "scnrm2" cblas_scnrm2_safe cblas_scnrm2_unsafe

dznrm2 :: PrimMonad m => Nrm2Fun (Complex Double) (PrimState m) m Double
dznrm2 = norm2Abstraction "dznrm2" cblas_dznrm2_safe cblas_dznrm2_unsafe

srot :: PrimMonad m => RotFun Float (PrimState m) m
srot = rotAbstraction "srot" cblas_srot_safe cblas_srot_unsafe

drot :: PrimMonad m => RotFun Double (PrimState m) m
drot = rotAbstraction "drot" cblas_drot_safe cblas_drot_unsafe

{-
 -srotg :: PrimMonad m => RotgFun Float (PrimState m) m
 -srotg = rotgAbstraction "srotg" cblas_srotg_safe cblas_srotg_unsafe
 -
 -drotg :: PrimMonad m => RotgFun Double (PrimState m) m
 -drotg = rotgAbstraction "drotg" cblas_drotg_safe cblas_drotg_unsafe
 -}

{-
 -srotm :: PrimMonad m => RotmFun Float (PrimState m) m
 -srotm = rotmAbstraction "srotm" cblas_srotm_safe cblas_srotm_unsafe
 -
 -drotm :: PrimMonad m => RotmFun Double (PrimState m) m
 -drotm = rotmAbstraction "drotm" cblas_drotm_safe cblas_drotm_unsafe
 -}

{-
 -srotmg :: PrimMonad m => RotmgFun Float (PrimState m) m
 -srotmg = rotmgAbstraction "srotmg" cblas_srotmg_safe cblas_srotmg_unsafe
 -
 -drotmg :: PrimMonad m => RotmgFun Double (PrimState m) m
 -drotmg = rotmgAbstraction "drotmg" cblas_drotmg_safe cblas_drotmg_unsafe
 -}

sscal :: PrimMonad m => ScalFun Float Float (PrimState m) m
sscal = scalAbstraction "sscal" cblas_sscal_safe cblas_sscal_unsafe (\x f -> f x )

dscal :: PrimMonad m => ScalFun Double Double (PrimState m) m
dscal = scalAbstraction "dscal" cblas_dscal_safe cblas_dscal_unsafe (\x f -> f x )

cscal :: PrimMonad m => ScalFun (Complex Float) (Complex Float) (PrimState m) m
cscal = scalAbstraction "cscal" cblas_cscal_safe cblas_cscal_unsafe withRStorable_

zscal :: PrimMonad m => ScalFun (Complex Double) (Complex Double) (PrimState m) m
zscal = scalAbstraction "zscal" cblas_zscal_safe cblas_zscal_unsafe withRStorable_

csscal :: PrimMonad m => ScalFun Float (Complex Float) (PrimState m) m
csscal = scalAbstraction "csscal" cblas_csscal_safe cblas_csscal_unsafe (\x f -> f x )

zdscal :: PrimMonad m => ScalFun Double (Complex Double) (PrimState m) m
zdscal = scalAbstraction "zdscal" cblas_zdscal_safe cblas_zdscal_unsafe (\x f -> f x )

sswap :: PrimMonad m => SwapFun Float (PrimState m) m
sswap = swapAbstraction "sswap" cblas_sswap_safe cblas_sswap_unsafe

dswap :: PrimMonad m => SwapFun Double (PrimState m) m
dswap = swapAbstraction "dswap" cblas_dswap_safe cblas_dswap_unsafe

cswap :: PrimMonad m => SwapFun (Complex Float) (PrimState m) m
cswap = swapAbstraction "cswap" cblas_cswap_safe cblas_cswap_unsafe

zswap :: PrimMonad m => SwapFun (Complex Double) (PrimState m) m
zswap = swapAbstraction "zswap" cblas_zswap_safe cblas_zswap_unsafe

isamax :: PrimMonad m => IamaxFun Float (PrimState m) m
isamax = iamaxAbstraction "isamax" cblas_isamax_safe cblas_isamax_unsafe

idamax :: PrimMonad m => IamaxFun Double (PrimState m) m
idamax = iamaxAbstraction "idamax" cblas_idamax_safe cblas_idamax_unsafe

icamax :: PrimMonad m => IamaxFun (Complex Float) (PrimState m) m
icamax = iamaxAbstraction "icamax" cblas_icamax_safe cblas_icamax_unsafe

izamax :: PrimMonad m => IamaxFun (Complex Double)(PrimState m) m
izamax = iamaxAbstraction "izamax" cblas_izamax_safe cblas_izamax_unsafe

{-
isamin :: PrimMonad m => IaminFun Float (PrimState m) m
isamin = iaminAbstraction "isamin" cblas_isamin_safe cblas_isamin_unsafe

idamin :: PrimMonad m => IaminFun Double (PrimState m) m
idamin = iaminAbstraction "idamin" cblas_idamin_safe cblas_idamin_unsafe

icamin :: PrimMonad m => IaminFun (Complex Float) (PrimState m) m
icamin = iaminAbstraction "icamin" cblas_icamin_safe cblas_icamin_unsafe

izamin :: PrimMonad m => IaminFun (Complex Double)(PrimState m) m
izamin = iaminAbstraction "izamin" cblas_izamin_safe cblas_izamin_unsafe
-}
