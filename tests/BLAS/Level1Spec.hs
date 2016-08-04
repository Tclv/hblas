-- {-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds, GADTs, TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}

module BLAS.Level1Spec (main, spec) where

--import Test.HUnit
--import Numerical.Array.Shape as S
import Prelude as P

import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as SMV

import Data.Complex

import Numerical.HBLAS.MatrixTypes as Matrix
import Numerical.HBLAS.BLAS.Level1 as BLAS

import Test.Hspec

main :: IO ()
main = hspec $ do spec

spec :: Spec
spec = do
  asumSpec
  axpySpec
  copySpec
  dotSpec
  nrm2Spec
  rotSpec
  scalSpec
  swapSpec
  amaxSpec

asumSpec :: Spec
asumSpec =
  context "?ASUM: Sum of magnitudes" $ do
    describe "SASUM: Single Precision" $ do
      it "Sum of [1.0,..,6.0] is equal to 21.0" $ do
        vecTest1SASUM
      it "Sum of [1.0,..,12.0] with a stride of 2 is equal to 36.0" $ do
        vecTest2SASUM
    describe "DASUM" $ do
      it "Length 6, Stride 1" $ do
        vecTest1DASUM
      it "Length 12, Stride 2" $ do
        vecTest2DASUM

vecTest1SASUM :: IO ()
vecTest1SASUM = do
  vec <- Matrix.generateMutableDenseVector 6 (\idx -> [1 .. 6] !! idx)
  res <- BLAS.sasum 6 vec 
  res `shouldBe` (21.0 :: Float)

vecTest2SASUM :: IO ()
vecTest2SASUM = do
  vec <- Matrix.generateMutableDenseVectorWithStride 12 2 (\idx -> [1 .. 12] !! idx)
  res <- BLAS.sasum 6 vec
  res `shouldBe` (36.0 :: Float)

vecTest1DASUM :: IO ()
vecTest1DASUM = do
  vec <- Matrix.generateMutableDenseVector 6 (\idx -> [1 .. 6] !! idx)
  res <- BLAS.dasum 6 vec 
  res `shouldBe` (21.0 :: Double)

vecTest2DASUM :: IO ()
vecTest2DASUM = do
  vec <- Matrix.generateMutableDenseVectorWithStride 12 2 (\idx -> [1 .. 12] !! idx)
  res <- BLAS.dasum 6 vec
  res `shouldBe` (36.0 :: Double)

axpySpec :: Spec
axpySpec =
  context "?AXPY a * x + y, (a: scalar; x, y: vector)" $ do
    describe "SAXPY" $ do
      it "(Length 6, Stride 1) (Length 6, Stride 1)" $ do
        vecTest1SAXPY
      it "a = 2.0, x = [1.0,..,18.0] stride 3, y = [1.0,..,12.0] stride 2 axpy = [3, 2, 11, 4, 19, 6, 27, 8, 35, 10, 43, 12] " $ do
        vecTest2SAXPY

vecTest1SAXPY :: IO ()
vecTest1SAXPY = do
  input <- Matrix.generateMutableDenseVector 6 (\idx -> [1 .. 6] !! idx)
  output <- Matrix.generateMutableDenseVector 6 (\idx -> [2, 3, 4, 3, 5, 6] !! idx)
  BLAS.saxpy 6 (-1.0) input output
  resList <- Matrix.mutableVectorToList $ _bufferMutDenseVector output
  resList `shouldBe` [1, 1, 1, -1, 0, 0]


vecTest2SAXPY :: IO ()
vecTest2SAXPY = do
  input <- Matrix.generateMutableDenseVectorWithStride 18 3 (\idx -> [1 .. 18] !! idx)
  output <- Matrix.generateMutableDenseVectorWithStride 12 2 (\idx -> [1 .. 12] !! idx)
  BLAS.saxpy 6 2.0 input output
  resList <- Matrix.mutableVectorToList $ _bufferMutDenseVector output
  resList `shouldBe` [3, 2, 11, 4, 19, 6, 27, 8, 35, 10, 43, 12]


copySpec :: Spec
copySpec =
  context "?COPY x = y (x, y: vector)" $ do
    describe "DCOPY" $ do
      it "x = [1..6], y = [0, 0,..0] (dim 6); x = y, y = [1..6]" $ do
        vecTest1DCOPY
      it "x = [1..6] stride 2, y = [0.0,.., 0.0] (dim 9), stride 3; x = y, y = [1,0,0,3,0,0,5,0,0] " $ do
        vecTest2DCOPY

vecTest1DCOPY :: IO ()
vecTest1DCOPY = do
  input <- Matrix.generateMutableDenseVector 6 (\idx -> [1 .. 6] !! idx)
  output <- Matrix.generateMutableDenseVector 6 (const 0.0)
  BLAS.dcopy 6 input output
  resList <- Matrix.mutableVectorToList $ _bufferMutDenseVector output
  resList `shouldBe` [1, 2, 3, 4, 5, 6]

vecTest2DCOPY :: IO ()
vecTest2DCOPY = do
  input <- Matrix.generateMutableDenseVectorWithStride 6 2 (\idx -> [1 .. 6] !! idx)
  output <- Matrix.generateMutableDenseVectorWithStride 9 3 (const 0.0)
  BLAS.dcopy 3 input output
  resList <- Matrix.mutableVectorToList $ _bufferMutDenseVector output
  resList `shouldBe` [1, 0, 0, 3, 0, 0, 5, 0, 0]


dotSpec :: Spec
dotSpec =
  context "?DOT = x * y, (x, y : vector)" $ do
    describe "SDOT" $ do
      it "x = [1..6], stride 2, y = [1 .. 12], stride 4, x * y = 61" $ do
        vecTest1SDOT
    describe "DDOT" $ do
      it "x = [1..12], stride 2, y = [1..6], stride 1, x * y = 161" $ do
        vecTest1DDOT
    describe "SDSDOT x * y + z (z : scalar)" $ do
      it "x = [1..6], stride 2, y = [1..12], stride 4, x * y + 2 = 63" $ do
        vecTest1SDSDOT
    describe "DSDOT" $ do
      it "x = [1..12], stride 2, y = [1..6], stride 1, x * y = 161" $ do
        vecTest1DSDOT

vecTest1SDOT :: IO ()
vecTest1SDOT = do
  left <- Matrix.generateMutableDenseVectorWithStride 6 2 (\idx -> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] !! idx)
  right <- Matrix.generateMutableDenseVectorWithStride 12 4 (\idx -> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] !! idx)
  res <- sdot 3 left right
  res `shouldBe` ((1 + 15 + 45) :: Float)

vecTest1DDOT :: IO ()
vecTest1DDOT = do
  left <- Matrix.generateMutableDenseVectorWithStride 12 2 (((+) 1) . fromRational . toRational)
  right <- Matrix.generateMutableDenseVectorWithStride 6 1 (\idx -> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] !! idx)
  res <- ddot 6 left right
  res `shouldBe` ((1 + 6 + 15 + 28 + 45 + 66) :: Double)

vecTest1SDSDOT :: IO ()
vecTest1SDSDOT = do
  left <- Matrix.generateMutableDenseVectorWithStride 6 2 (\idx -> ([1 .. 6] :: [Float]) !! idx) -- Float
  right <- Matrix.generateMutableDenseVectorWithStride 12 4 (\idx -> ([1 .. 12] :: [Float]) !! idx) -- Float
  res <- sdsdot 3 2.0 left right
  res `shouldBe` ((2 :: Float) + 1 + 15 + 45)

vecTest1DSDOT :: IO ()
vecTest1DSDOT = do
  left <- Matrix.generateMutableDenseVectorWithStride 12 2 (\idx -> ([1 .. 12] :: [Float]) !! idx)
  right <- Matrix.generateMutableDenseVectorWithStride 6 1 (\idx -> ([1 .. 6] :: [Float]) !! idx)
  res <- dsdot 6 left right
  res `shouldBe` ((1 + 6 + 15 + 28 + 45 + 66) :: Double)

{-
 -vecTest1CDOTU :: IO ()
 -vecTest1CDOTU = do
 -  left <- Matrix.generateMutableDenseVectorWithStride 6 2 (\idx -> [1:+1, 1:+(-1), 1:+1, 1:+(-1), 1:+1, 1:+(-1)] !! idx)
 -  right <- Matrix.generateMutableDenseVectorWithStride 9 3 (\idx -> [1:+(-2), 1:+1, 1:+(-1), 1:+1, 1:+(-1), 1:+1, 1:+(-1), 1:+1, 1:+(-1)] !! idx)
 -  res <- Matrix.generateMutableValue (1:+1)
 -  cdotu 3 left right res
 -  resValue <- Matrix.mutableValueToValue res
 -  resValue `shouldBe` 5:+1
 -
 -vecTest1CDOTC :: IO ()
 -vecTest1CDOTC = do
 -  left <- Matrix.generateMutableDenseVectorWithStride 6 2 (\idx -> [2:+3, 1:+(-1), 1:+1, 1:+(-1), 1:+1, 1:+(-1)] !! idx)
 -  right <- Matrix.generateMutableDenseVectorWithStride 9 3 (\idx -> [1:+(-2), 1:+1, 1:+(-1), 1:+1, 1:+(-1), 1:+1, 1:+(-1), 1:+1, 1:+(-1)] !! idx)
 -  res <- Matrix.generateMutableValue (1:+1)
 -  cdotc 3 left right res
 -  resValue <- Matrix.mutableValueToValue res
 -  resValue `shouldBe` (-2):+(-9)
 -}


nrm2Spec = do
  context "?NRM2, ||x||, euclidian norm/l2 norm" $ do
    describe "SNRM2" $ do
      it "snrm2([1,-2,3,-4,5,-6]) should be close to 9.5394" $ do
        vecTest1SNRM2
    describe "DZNRM2" $ do
      it "dznrm2([1 + i, 1 + 2i, 2 - 3i, 2 - 2i, -3 + i, -4 + 2i, -3 + i, -3, -4 + 2i, -4 + i], stride 2) should be close to 6.7082" $ do
        vecTest1DZNRM2

vecTest1SNRM2 :: IO ()
vecTest1SNRM2 = do
  input <- Matrix.generateMutableDenseVector 6 (\idx -> [1.0, -2.0, 3.0, -4.0, 5.0, -6.0] !! idx)
  res <- snrm2 6 input
  True `shouldBe` 1e-6 > (abs $ res - (sqrt $ sum $ fmap (\x->x^2) [1, 2, 3, 4, 5, 6]))

vecTest1DZNRM2 :: IO ()
vecTest1DZNRM2 = do
  input <- Matrix.generateMutableDenseVectorWithStride 8 2 (\idx -> [1:+1, 1:+2, 2:+(-3), 2:+(-2), (-3):+1, (-3):+0, (-4):+2, (-4):+1] !! idx)
  res <- dznrm2 4 input
  True `shouldBe` 1e-12 > (abs $ res - (sqrt $ sum $ fmap (\x->x^2) [1, 1, 2, 3, 3, 1, 4, 2]))

rotSpec = do
  context "?ROT x,y :vector, s,c: scalar, x = c * x + s * y; y = c * y - s * x" $ do
    describe "SROT" $ do
      it "srot([1,..,6],[6,..,1]) = ([11,2,5,4,-1,6],[-8,5,-10,3,-12,1])" $ do
        vecTest1SROT
    describe "DROT" $ do
      it "drot([1,..,4], [8,..,1] stride 2) = ([-16,-12,-8,-4],[2,7,4,5,6,3,8,1])" $ do
        vecTest1DROT

vecTest1SROT :: IO ()
vecTest1SROT = do
  left <- Matrix.generateMutableDenseVectorWithStride 6 2 (\idx -> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] !! idx)
  right <- Matrix.generateMutableDenseVectorWithStride 6 2 (\idx -> [6.0, 5.0, 4.0, 3.0, 2.0, 1.0] !! idx)
  srot 3 left right (-1) 2
  resLeft <- Matrix.mutableVectorToList $ _bufferMutDenseVector left
  resRight <- Matrix.mutableVectorToList $ _bufferMutDenseVector right
  resLeft `shouldBe` [11.0, 2.0, 5.0, 4.0, -1.0, 6.0]
  resRight `shouldBe` [-8.0, 5.0, -10.0, 3.0, -12.0, 1.0]

vecTest1DROT :: IO ()
vecTest1DROT = do
  left <- Matrix.generateMutableDenseVectorWithStride 4 1 (\idx -> [1, 2, 3, 4] !! idx)
  right <- Matrix.generateMutableDenseVectorWithStride 8 2 (\idx -> [8, 7, 6, 5, 4, 3, 2, 1] !! idx)
  drot 4 left right 0 (-2)
  resLeft <- Matrix.mutableVectorToList $ _bufferMutDenseVector left
  resRight <- Matrix.mutableVectorToList $ _bufferMutDenseVector right
  resLeft `shouldBe` [-16, -12, -8, -4]
  resRight `shouldBe` [2, 7, 4, 5, 6, 3, 8, 1]

{-
 -vecTest1SROTG :: IO ()
 -vecTest1SROTG = do
 -  a <- Matrix.generateMutableValue 3
 -  b <- Matrix.generateMutableValue 4
 -  c <- Matrix.generateMutableValue 0
 -  s <- Matrix.generateMutableValue 0
 -  srotg a b c s
 -  av <- Matrix.mutableValueToValue a
 -  bv <- Matrix.mutableValueToValue b
 -  cv <- Matrix.mutableValueToValue c
 -  sv <- Matrix.mutableValueToValue s
 -  av `shouldBe` 5
 -  True `shouldBe` 1e-6 > (abs $ bv - 1/0.6)
 -  cv `shouldBe` 0.6
 -  sv `shouldBe` 0.8
 -
 -vecTest1DROTG :: IO ()
 -vecTest1DROTG = do
 -  a <- Matrix.generateMutableValue 5.8
 -  b <- Matrix.generateMutableValue 3.4
 -  c <- Matrix.generateMutableValue 0
 -  s <- Matrix.generateMutableValue 0
 -  drotg a b c s
 -  av <- Matrix.mutableValueToValue a
 -  bv <- Matrix.mutableValueToValue b
 -  cv <- Matrix.mutableValueToValue c
 -  sv <- Matrix.mutableValueToValue s
 -  True `shouldBe` 1e-12 > (abs $ av - sqrt(3.4^2 + 5.8^2))
 -  True `shouldBe` 1e-12 > (abs $ bv - 3.4 / sqrt(3.4^2 + 5.8^2))
 -  True `shouldBe` 1e-12 > (abs $ cv - 5.8 / sqrt(3.4^2 + 5.8^2))
 -  True `shouldBe` 1e-12 > (abs $ sv - 3.4 / sqrt(3.4^2 + 5.8^2))
 -
 -vecTest1DROTM :: IO ()
 -vecTest1DROTM = do
 -  x <- Matrix.generateMutableDenseVectorWithStride 4 1 (\idx -> [1, 2, 3, 4] !! idx)
 -  y <- Matrix.generateMutableDenseVectorWithStride 8 2 (\idx -> [8, 7, 6, 5, 4, 3, 2, 1] !! idx)
 -  param <- Matrix.generateMutableDenseVector 5 (\idx -> [-1, 0, -1, 1, 0] !! idx)
 -  drotm 4 x y param
 -  resX <- Matrix.mutableVectorToList $ _bufferMutDenseVector x
 -  resY <- Matrix.mutableVectorToList $ _bufferMutDenseVector y
 -  resX `shouldBe` [8, 6, 4, 2]
 -  resY `shouldBe` [-1, 7, -2, 5, -3, 3, -4, 1]
 -
 -vecTest1SROTM :: IO ()
 -vecTest1SROTM = do
 -  x <- Matrix.generateMutableDenseVectorWithStride 6 2 (\idx -> [1, 2, 3, 4, 5, 6] !! idx)
 -  y <- Matrix.generateMutableDenseVectorWithStride 9 3 (\idx -> [9, 8, 7, 6, 5, 4, 3, 2, 1] !! idx)
 -  param <- Matrix.generateMutableDenseVector 5 (\idx -> [1, 1, 2, -2, 1] !! idx)
 -  srotm 3 x y param
 -  resX <- Matrix.mutableVectorToList $ _bufferMutDenseVector x
 -  resY <- Matrix.mutableVectorToList $ _bufferMutDenseVector y
 -  resX `shouldBe` [10, 2, 9, 4, 8, 6]
 -  resY `shouldBe` [8, 8, 7, 3, 5, 4, -2, 2, 1]
 -
 -vecTest1SROTMG :: IO ()
 -vecTest1SROTMG = do
 -  d1 <- Matrix.generateMutableValue 3
 -  d2 <- Matrix.generateMutableValue 6
 -  x <- Matrix.generateMutableValue 1
 -  let y = 1
 -  param <- Matrix.generateMutableDenseVector 5 (\idx -> [-1, 1, 1, -1, 1] !! idx)
 -  srotmg d1 d2 x y param
 -  paramR <- Matrix.mutableVectorToList $ _bufferMutDenseVector param
 -  updatedD1 <- Matrix.mutableValueToValue d1
 -  updatedD2 <- Matrix.mutableValueToValue d2
 -  updatedX <- Matrix.mutableValueToValue x
 -  paramR `shouldBe` [1, 0, 0.5, 0, 1]
 -  updatedD1 `shouldBe` 4
 -  updatedD2 `shouldBe` 2
 -  updatedX `shouldBe` 1.5
 -}


scalSpec :: Spec
scalSpec = do
  context "?SCAL, c * x, c : scalar, x : vector" $ do
    describe "SSCAL" $ do
      it "c = (-2), x = [1..8], stride 2. c * x = [-2, 2, -6, 4, -10, 6, -14, 8]" $ do
        vecTest1SSCAL
    describe "CSCAL" $ do
      it "c = 2-2i, x = [1+i,1+2i,2-3i,2-2i,-3+i,-3,-4+2i,-4+i] stride 4, c * x = [4, 1+2i, 2-3i, 2-2i,-4+8i,-3,-4+2i,-4+i]" $ do
        vecTest1CSCAL
    describe "CSSCAL" $ do
      it "c = -2, x = [1+i,1+2i,2-3i,2-2i,-3+i,-3,-4+2i,-4+i], c * x = [-2-2i,-2-4i,-4+6i,-4+4i,6-2i,6,8-4i,8-2i]" $ do
        vecTest1CSSCAL

vecTest1SSCAL :: IO ()
vecTest1SSCAL = do
  x <- Matrix.generateMutableDenseVectorWithStride 8 2 (\idx -> [1, 2, 3, 4, 5, 6, 7, 8] !! idx)
  sscal 4 (-2) x
  xRes <- Matrix.mutableVectorToList $ _bufferMutDenseVector x
  xRes `shouldBe` [-2, 2, -6, 4, -10, 6, -14, 8]


vecTest1CSCAL :: IO ()
vecTest1CSCAL = do
  x <- Matrix.generateMutableDenseVectorWithStride 8 4 (\idx -> [1:+1, 1:+2, 2:+(-3), 2:+(-2), (-3):+1, (-3):+0, (-4):+2, (-4):+1] !! idx)
  cscal 2 (2:+(-2)) x -- size 2?
  xRes <- Matrix.mutableVectorToList $ _bufferMutDenseVector x
  xRes `shouldBe` [4:+0, 1:+2, 2:+(-3), 2:+(-2), (-4):+8, (-3):+0, (-4):+2, (-4):+1]

vecTest1CSSCAL :: IO ()
vecTest1CSSCAL = do
  x <- Matrix.generateMutableDenseVector 8 (\idx -> [1:+1, 1:+2, 2:+(-3), 2:+(-2), (-3):+1, (-3):+0, (-4):+2, (-4):+1] !! idx)
  csscal 8 (-2) x
  xRes <- Matrix.mutableVectorToList $ _bufferMutDenseVector x
  xRes `shouldBe` [(-2):+(-2), (-2):+(-4), (-4):+6, (-4):+4, 6:+(-2), 6:+0, 8:+(-4), 8:+(-2)]

swapSpec :: Spec
swapSpec = do
  context "?SWAP, x,y:vector, x = y; y = x" $ do
    describe "SSWAP" $ do
      it "x = [1..8] stride 2, y = [-1,..,-4] stride 1. swap(x,y) = ([-1,2,-2,4,-3,6,-4,8], [1,3,5,7])" $ do
        vecTest1SSWAP
    describe "CSWAP" $ do
      it "x, strided 3, and y, strided 2" $ do
        vecTest1CSWAP
      
vecTest1SSWAP :: IO ()
vecTest1SSWAP = do
  x <- Matrix.generateMutableDenseVectorWithStride 8 2 (\idx -> [1, 2, 3, 4, 5, 6, 7, 8] !! idx)
  y <- Matrix.generateMutableDenseVectorWithStride 4 1 (\idx -> [-1, -2, -3, -4] !! idx)
  sswap 4 x y
  xRes <- Matrix.mutableVectorToList $ _bufferMutDenseVector x
  yRes <- Matrix.mutableVectorToList $ _bufferMutDenseVector y
  xRes `shouldBe` [-1, 2, -2, 4, -3, 6, -4, 8]
  yRes `shouldBe` [1, 3, 5, 7]

vecTest1CSWAP :: IO ()
vecTest1CSWAP = do
  x <- Matrix.generateMutableDenseVectorWithStride 9 3 (\idx -> [1:+1, 1:+2, 2:+(-3), 2:+(-2), (-3):+1, (-3):+0, (-4):+2, (-4):+1, 0:+9] !! idx)
  y <- Matrix.generateMutableDenseVectorWithStride 6 2 (\idx -> [1:+2, 1:+3, 3:+(-3), 2:+2, 3:+1, 3:+3] !! idx)
  cswap 3 x y
  xRes <- Matrix.mutableVectorToList $ _bufferMutDenseVector x
  yRes <- Matrix.mutableVectorToList $ _bufferMutDenseVector y
  xRes `shouldBe` [1:+2, 1:+2, 2:+(-3), 3:+(-3), (-3):+1, (-3):+0, 3:+1, (-4):+1, 0:+9]
  yRes `shouldBe` [1:+1, 1:+3, 2:+(-2), 2:+2, (-4):+2, 3:+3]

amaxSpec :: Spec
amaxSpec = do
  context "I?AMAX, max of index" $ do
    describe "ISAMAX" $ do
      it "isamax([1..8], strided 2) = 3" $ do
        vecTest1ISAMAX
    describe "ICAMAX" $ do
      it "icamax([1+i, 1+2i, 2-3i, 2-2i, -3+i, -3, -4+2i, -4+i, 9i]) = 8" $ do
        vecTest1ICAMAX

vecTest1ISAMAX :: IO ()
vecTest1ISAMAX = do
  x <- Matrix.generateMutableDenseVectorWithStride 8 2 (\idx -> [1, 2, 3, 4, 5, 6, 7, 8] !! idx)
  idx <- isamax 4 x
  idx `shouldBe` 3

vecTest1ICAMAX :: IO ()
vecTest1ICAMAX = do
  x <- Matrix.generateMutableDenseVector 9 (\idx -> [1:+1, 1:+2, 2:+(-3), 2:+(-2), (-3):+1, (-3):+0, (-4):+2, (-4):+1, 0:+9] !! idx)
  idx <- icamax 9 x
  idx `shouldBe` 8

{-
vecTest1ISAMIN :: IO ()
vecTest1ISAMIN = do
  x <- Matrix.generateMutableDenseVector 8 (\idx -> [1, 2, 3, 4, -5, 6, 7, 8] !! idx)
  idx <- isamin 4 x 2
  idx `shouldBe` 2

vecTest1ICAMIN :: IO ()
vecTest1ICAMIN = do
  x <- Matrix.generateMutableDenseVector 9 (\idx -> [1:+2, 1:+2, (-2):+(-3), 2:+(-2), (-3):+1, (-2):+0, (-4):+2, (-4):+1, 0:+9] !! idx)
  idx <- icamin 9 x 1
  idx `shouldBe` 5
-}
