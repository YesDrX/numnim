import math
import random
import ndarray
import sequtils
import ../common
import strformat
import ndarray_masks

proc get_pair_of_normal(): array[2,float]=
  var u,v,z1,z2: float

  u = rand(1.0)
  v = rand(1.0)
  z1 = sqrt(-2*ln(u)) * cos(2*PI*v)
  z2 = sqrt(-2*ln(v)) * cos(2*PI*u)
  result = [z1,z2]

proc uniform*(shape: seq[int], min:float = 0.0, max: float = 1.0, seed:int64 = -1): NdArray[float]=
  assert(min<max,"min is larger than max when generating uniform random numbers.")
  if seed == -1: randomize()

  var length = prodOfSeq(shape)
  var data_seq = newSeq[float](length)
  for idx in 0..length-1: data_seq[idx] = rand(max-min)+min

  result = data_seq.toNdArray(shape)

proc uniform*(shape: openArray[int], min:float = 0.0, max:float = 1.0, seed:int64 = -1): NdArray[float]=
  result = uniform(shape.toSeq, min, max, seed)

proc normal*(shape: seq[int], mu:float = 0.0, sigma:float = 1.0, seed:int64 = -1): NdArray[float]=
  #Box-Muller Method
  #link: https://en.wikipedia.org/wiki/Box-Muller_transform
  assert(sigma>0,"sigma is negative when generating normal random numbers.")
  if seed == -1: randomize()
  var length = prodOfSeq(shape) 
  var data_seq = newSeq[float](length)

  var
    z: array[2,float]

  for idx in 0 ..< ((length-1) div 2) :
    z = get_pair_of_normal()
    data_seq[2*idx] = z[0] * sigma + mu
    data_seq[2*idx+1] = z[1] * sigma + mu
  
  z = get_pair_of_normal()
  if (length-1) mod 2 == 0:
    data_seq[length-1] = z[0] * sigma + mu
  else:
    data_seq[length-2] = z[0] * sigma + mu
    data_seq[length-1] = z[1] * sigma + mu

  result = data_seq.toNdArray(shape)

proc normal*(shape: openArray[int], mu:float = 0.0, sigma:float = 1.0, seed:int64 = -1): NdArray[float]=
  result = normal(shape.toSeq, mu, sigma, seed)

proc bernoulli*(shape: seq[int], p: float): NdArray[float]=
  assert(p>=0 and p<=1, fmt"probability parameter p={p} is not valid for Bernoulli distribution.")
  var
    middle1 = uniform(shape)
    middle2 = (middle1 >= p)
  result = middle2.toSeq.astype(float).toNdArray(shape)

proc gen_one_possion_point(lambda: float): float=
  var
    c = 0.767 - 3.36/lambda
    beta = PI/sqrt(3.0*lambda)
    alpha = beta*lambda
    k = ln(c) - lambda - ln(beta)
    u,v,x,y,lhs,rhs,n: float
  while true:
    u = rand(1.0)
    x = (alpha-ln((1.0-u)/u))/beta
    n = floor(x + 0.5)
    if n<0 :
      continue
    v = rand(1.0)
    y = alpha - beta * x
    lhs = y + ln(v/(1.0+exp(y))^2)
    rhs = k + n*ln(lambda) - ln(n.int.fac.float)

    if lhs <= rhs :
      return n

when isMainModule:
  # echo uniform([10,10])
  # echo normal([10,10])
  # echo bernoulli(@[10,10],0.95)
  echo gen_one_possion_point(4.0)
