# encoding: utf-8
# cython: profile=False
# cython: linetrace=False

from __future__ import division
from numpy import inf
from numpy.random import uniform
from cpython cimport array

def rebuild_livepoint(names, bounds):
  lp=LivePoint(names,bounds)
  return lp

cdef class LivePoint:
    def __cinit__(LivePoint self, list names, list bounds, d=None):
        self.logL = -inf
        self.logP = -inf
        self.names = names
        self.bounds=bounds
        self.dimension = len(names)
        cdef unsigned int i
        if d is not None:
          self.values = array.array('d',d)
        else:
          self.values = array.array('d',[0]*self.dimension)

    def __reduce__(self):
        return (rebuild_livepoint, (self.names,self.bounds),self.__getstate__()) 
    
    def __getstate__(self):
      return (self.logL,self.logP,self.values)
    def __setstate__(self,state):
      self.logL=state[0]
      self.logP=state[1]
      self.values=array.array('d',state[2])
    
    def initialise(LivePoint self):
        for i,n in enumerate(self.names):
            self[n] = uniform(self.bounds[i][0],self.bounds[i][1])

    def inbounds(LivePoint self):
      return all(self.bounds[i][0] < self.values[i] < self.bounds[i][1] for i in range(self.dimension))

    def __str__(LivePoint self):
        return str({n:self[n] for n in self.names})

    def __cmp__(LivePoint self,LivePoint other):
        for i in range(self.dimension):
            if not self.names[i] in other.names or self[self.names[i]]!=other[self.names[i]]:
                return 1
        return 0

    def __add__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        result=LivePoint(self.names,self.bounds)
        for n in self.names:
            result[n]=self[n]+other[n]
        return result

    def __iadd__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        for n in self.names:
            self[n]=self[n]+other[n]
        return self
    
    def __sub__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        result = LivePoint(self.names,self.bounds)
        for n in self.names:
            result[n]=self[n]-other[n]
        return result

    def __isub__(LivePoint self,LivePoint other):
        assert self.dimension == other.dimension
        for n in self.names:
            self[n]=self[n]-other[n]
        return self

    def __mul__(LivePoint self,float other):
        result=LivePoint(self.names,self.bounds)
        for n in self.names:
            result[n]=other*self[n]
        return result

    def __imul__(LivePoint self,float other):
        for n in self.names:
            self[n]=other*self[n]
        return self

    def __truediv__(LivePoint self,float other):
        result = LivePoint(self.names,self.bounds)
        for n in self.names:
            result[n]=self[n]/other
        return result

    def __itruediv__(LivePoint self,float other):
        for n in self.names:
            self[n]=self[n]/other
        return self

    def __len__(LivePoint self):
        return self.dimension
    
    def __getitem__(LivePoint self, str name):
        cdef unsigned int i
        for i in range(self.dimension):
            if self.names[i] == name:
                return self.values[i]
        raise KeyError

    def __setitem__(LivePoint self, str name, double value):
        cdef unsigned int i
        for i in range(self.dimension):
            if self.names[i] == name:
                self.values[i] = value
                return
        raise KeyError

    cpdef copy(LivePoint self):
      result = LivePoint(self.names,self.bounds)
      result.__setstate__(self.__getstate__())
      return result
                

