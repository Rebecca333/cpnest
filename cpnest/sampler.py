from __future__ import division
import sys
import os
import numpy as np
from math import log
from collections import deque
from random import random,randrange
from operator import attrgetter
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from scipy.stats import multivariate_normal

from . import parameter
from . import proposal

class Sampler(object):
    """
    Sampler class.
    Initialisation arguments:
    
    usermodel:
    user defined model to sample
    
    maxmcmc:
    maximum number of mcmc steps to be used in the sampler
    
    verbose:
    display debug information on screen
    default: False
    
    poolsize:
    number of objects for the affine invariant sampling
    default: 1000
    """
    def __init__(self,usermodel,maxmcmc,verbose=False,poolsize=1000):
        self.user = usermodel
        self.maxmcmc = maxmcmc
        self.Nmcmc = 64
        self.Nmcmc_exact = float(self.Nmcmc)
        self.proposals = proposal.DefaultProposalCycle()
        self.poolsize = poolsize
        self.evolution_points = deque(maxlen=self.poolsize + 1) # +1 for the point to evolve
        self.verbose=verbose
        self.acceptance=0.0
        self.initialised=False
    
    def reset(self):
        """
        Initialise the sampler
        """
        for n in range(self.poolsize):
          while True:
            if self.verbose > 2: sys.stderr.write("process {0!s} --> generating pool of {1:d} points for evolution --> {2:.0f} % complete\r".format(os.getpid(), self.poolsize, 100.0*float(n+1)/float(self.poolsize)))
            p = self.user.new_point()
            p.logP = self.user.log_prior(p)
            if np.isfinite(p.logP): break
          p.logL=self.user.log_likelihood(p)
          self.evolution_points.append(p)
        if self.verbose > 2: sys.stderr.write("\n")
        self.proposals.set_ensemble(self.evolution_points)
        for _ in range(len(self.evolution_points)):
          s = self.evolution_points.popleft()
          s = self.metropolis_hastings(s,-np.inf)
          self.evolution_points.append(s)

        self.proposals.set_ensemble(self.evolution_points)
        self.initialised=True

    def estimate_nmcmc(self, safety=5, tau=None):
        """
        Estimate autocorrelation length of chain using acceptance fraction
        ACL = (2/acc) - 1
        multiplied by a safety margin of 5
        Uses moving average with decay time tau iterations (default: self.poolsize)
        Taken from W. Farr's github.com/farr/Ensemble.jl
        """
        if tau is None: tau = self.poolsize

        if self.acceptance == 0.0:
            self.Nmcmc_exact = (1.0 + 1.0/tau)*self.Nmcmc_exact
        else:
            self.Nmcmc_exact = (1.0 - 1.0/tau)*self.Nmcmc_exact + (safety/tau)*(2.0/self.acceptance - 1.0)
        
        self.Nmcmc_exact = float(min(self.Nmcmc_exact,self.maxmcmc))
        self.Nmcmc = max(safety,int(self.Nmcmc_exact))
        return self.Nmcmc

    def produce_sample(self, queue, logLmin, seed, ip, port, authkey):
        """
        main loop that generates samples and puts them in the queue for the nested sampler object
        """
        if not self.initialised:
            self.reset()

        # Prevent process from zombification if consumer thread exits
        queue.cancel_join_thread()
        self.seed = seed
        np.random.seed(seed=self.seed)
        self.counter=0
        
        while(1):
            
            # Pick a the first point from the ensemble to start with
            # Pop it out the stack to prevent cloning

            param = self.evolution_points.popleft()
            
            if logLmin.value==np.inf:
                break
            
            outParam = self.metropolis_hastings(param,logLmin.value)
           
            # Put sample back in the stack
            self.evolution_points.append(outParam.copy())
            # If we bailed out then flag point as unusable
            if self.acceptance==0.0:
                outParam.logL=-np.inf
            # Push the sample onto the queue
            queue.put((self.acceptance,self.jumps,outParam))
            # Update the ensemble every now and again
            if (self.counter%(self.poolsize/10))==0 or self.acceptance == 0.0:
                self.proposals.set_ensemble(self.evolution_points)
            self.counter += 1

        sys.stderr.write("Sampler process {0!s}, exiting\n".format(os.getpid()))
        return 0

    def metropolis_hastings(self,inParam,logLmin):
        """
        metropolis-hastings loop to generate the new live point taking nmcmc steps
        """
        self.jumps = 0
        accepted = 0
        oldparam = inParam.copy()
        logp_old = self.user.log_prior(oldparam)
        
        while self.jumps < self.Nmcmc:
            
            newparam = self.proposals.get_sample(oldparam.copy())
            newparam.logP = self.user.log_prior(newparam)
            
            if newparam.logP-logp_old + self.proposals.log_J > log(random()):
                newparam.logL = self.user.log_likelihood(newparam)
                
                if newparam.logL > logLmin:
                    oldparam = newparam
                    logp_old = newparam.logP
                    accepted+=1
                        
            self.jumps+=1
            
            if self.jumps > self.maxmcmc: break

        self.acceptance = float(accepted)/float(self.jumps)
        self.estimate_nmcmc()
        return oldparam

class HMCSampler(object):
    """
    Hamiltonian Monte Carlo Sampler class.
    Initialisation arguments:
    
    usermodel:
    user defined model to sample
    
    maxmcmc:
    maximum number of mcmc steps to be used in the sampler
    
    verbose:
    display debug information on screen
    default: False
    
    poolsize:
    number of objects for the gradients estimation
    default: 1000
    """
    def __init__(self,usermodel, maxmcmc, verbose=False, poolsize=1000):
        self.user           = usermodel
        self.maxmcmc        = maxmcmc
        self.Nmcmc          = maxmcmc
        self.Nmcmc_exact    = float(maxmcmc)
        self.proposals      = proposal.DefaultProposalCycle()
        self.poolsize       = poolsize
        self.positions      = deque(maxlen=self.poolsize + 1) # +1 for the point to evolve
        self.verbose        = verbose
        self.acceptance     = 0.0
        self.initialised    = False
        self.gradients      = {}
        self.step_size      = 0.03
        self.steps          = 10
        self.momenta_distribution = None
    
    def reset(self):
        """
        Initialise the sampler
        """
        for n in range(self.poolsize):
            while True:
                if self.verbose > 2: sys.stderr.write("process {0!s} --> generating pool of {1:d} points for evolution --> {2:.0f} % complete\r".format(os.getpid(), self.poolsize, 100.0*float(n+1)/float(self.poolsize)))
                p = self.user.new_point()
                p.logP = self.user.log_prior(p)
                if np.isfinite(p.logP): break
            p.logL=self.user.log_likelihood(p)
            self.positions.append(p)
        if self.verbose > 2: sys.stderr.write("\n")

        self.proposals.set_ensemble(self.positions)
        
        # seed the chains with a standard MCMC
        for j in range(len(self.positions)):
            if self.verbose > 2: sys.stderr.write("process {0!s} --> initial MCMC evolution for {1:d} points --> {2:.0f} % complete\r".format(os.getpid(), self.poolsize, 100.0*float(j+1)/float(self.poolsize)))
            s = self.positions.popleft()
            s = self.metropolis_hastings(s,-np.inf)
            self.positions.append(s)
        if self.verbose > 2: sys.stderr.write("\n")

        self.proposals.set_ensemble(self.positions)
        self.momenta_distribution = multivariate_normal(cov=self.proposals.mass_matrix)

        # estimate the initial gradients
        if self.verbose > 2: sys.stderr.write("Computing initial gradients ...")
        logProbs = np.array([-p.logP for p in self.positions])
        self.estimate_gradient(logProbs, 'logprior')
        logProbs = np.array([-p.logL for p in self.positions])
        self.estimate_gradient(logProbs, 'loglikelihood')
        
        if self.verbose > 2: sys.stderr.write("done\n")
        self.initialised=True

    def estimate_nmcmc(self, safety=1, tau=None):
        """
        Estimate autocorrelation length of chain using acceptance fraction
        ACL = (2/acc) - 1
        multiplied by a safety margin of 5
        Uses moving average with decay time tau iterations (default: self.poolsize)
        Taken from W. Farr's github.com/farr/Ensemble.jl
        """
        if tau is None: tau = self.poolsize

        if self.acceptance == 0.0:
            self.Nmcmc_exact = (1.0 + 1.0/tau)*self.Nmcmc_exact
        else:
            self.Nmcmc_exact = (1.0 - 1.0/tau)*self.Nmcmc_exact + (safety/tau)*(2.0/self.acceptance - 1.0)
        
        self.Nmcmc_exact = float(min(self.Nmcmc_exact,self.maxmcmc))
        self.Nmcmc       = max(safety,int(self.Nmcmc_exact))
        return self.Nmcmc

    def produce_sample(self, queue, logLmin, seed, ip, port, authkey):
        """
        main loop that generates samples and puts them in the queue for the nested sampler object
        """
        if not self.initialised:
            self.reset()
        # Prevent process from zombification if consumer thread exits
        queue.cancel_join_thread()
        self.seed = seed
        np.random.seed(seed=self.seed)
        self.counter=0
        
        while(1):
            
            # Pick a random point from the ensemble to start with

            position = self.positions[np.random.randint(self.poolsize)]
            self.positions.remove(position)
            
            if logLmin.value==np.inf:
                break
            
            # evolve it according to hamilton equations
            newposition = self.hamiltonian_sampling(position,logLmin.value)
           
            # Put sample back in the stack
            self.positions.append(newposition.copy())
            
            # If we bailed out then flag point as unusable
            if self.acceptance==0.0:
                newposition.logL=-np.inf
            # Push the sample onto the queue
            queue.put((self.acceptance,self.jumps,newposition))
            
            # Update the ensemble every now and again
            if (self.counter%(self.poolsize/10))==0:
                self.proposals.set_ensemble(self.positions)
                # update the gradients
                logProbs = np.array([-p.logP for p in self.positions])
                self.estimate_gradient(logProbs, 'logprior')
                logProbs = np.array([-p.logL for p in self.positions])
                self.estimate_gradient(logProbs, 'loglikelihood')
                self.momenta_distribution = multivariate_normal(cov=self.proposals.mass_matrix)

            self.counter += 1

        sys.stderr.write("Sampler process {0!s}, exiting\n".format(os.getpid()))
        return 0
    
    def estimate_gradient(self, logProbs, type):
        
        self.gradients[type] = []

        # loop over the parameters, spline interpolate and estimate the gradient
#        import matplotlib.pyplot as plt
        for j,key in enumerate(self.positions[0].names):
            x = np.array([self.positions[i][key] for i in range(len(self.positions))])
            idx = np.argsort(x)
            x = x[idx]
            logProbs = logProbs[idx]
            self.gradients[type].append(InterpolatedUnivariateSpline(x,logProbs,ext=0,check_finite=True).derivative())

    def gradient(self, inParam, gradients_list):
        return np.array([g(inParam[n]) for g,n in zip(gradients_list,self.positions[0].names)])

    def kinetic_energy(self, momentum):
        """Kinetic energy of the current velocity (assuming a standard Gaussian)
            (x dot x) / 2
        Parameters
        ----------
        velocity : tf.Variable
            Vector of current velocity
        Returns
        -------
        kinetic_energy : float
        """
        p = momentum.asnparray()
        p = p.view(dtype=np.float64)[:-2]

        return 0.5 * np.dot(p,np.dot(self.proposals.inverse_mass_matrix,p))

    def potential_energy(self, position):
        """
        potential energy of the current position
        """
        position.logP = self.user.log_prior(position)
        return -position.logP

    def hamiltonian(self, position, momentum):
        """Computes the Hamiltonian of the current position, velocity pair
        H = U(x) + K(v)
        U is the potential energy and is = -log_prior(x)
        Parameters
        ----------
        position : tf.Variable
            Position or state vector x (sample from the target distribution)
        velocity : tf.Variable
            Auxiliary velocity variable
        energy_function
            Function from state to position to 'energy'
             = -log_prior
        Returns
        -------
        hamitonian : float
        """
        return self.potential_energy(position) + self.kinetic_energy(momentum) #+position.logL

    def constrained_leapfrog_step(self, position, momentum, logLmin):
        """
        https://arxiv.org/pdf/1005.0157.pdf
        https://arxiv.org/pdf/1206.1901.pdf
        """
        # Updating the momentum a half-step
        g = self.gradient(position, self.gradients['logprior'])

        for j,k in enumerate(self.positions[0].names):
            momentum[k] += - 0.5 * self.step_size * g[j]

        for i in xrange(self.steps):
            
            # do a step
            for j,k in enumerate(self.positions[0].names):
                position[k] += self.step_size * momentum[k]
            
            # Update gradient
            g = self.gradient(position, self.gradients['logprior'])

            # compute the constraint
            position.logL = self.user.log_likelihood(position)

            # check on the constraint
            if position.logL > logLmin:
                # take a full momentum step
                for j,k in enumerate(self.positions[0].names):
                    momentum[k] += - self.step_size * g[j]
            else:
                # compute the normal to the constraint
                gL = self.gradient(position, self.gradients['loglikelihood'])
                n = gL/np.abs(np.sum(gL))
                # bounce on the constraint
                for j,k in enumerate(self.positions[0].names):
                    momentum[k] += - 2 * (momentum[k]*n[j]) * n[j]

        # Update the position
        for j,k in enumerate(self.positions[0].names):
            position[k] += self.step_size * momentum[k]

        # Do a final update of the momentum for a half step
        g = self.gradient(position, self.gradients['logprior'])
        for j,k in enumerate(self.positions[0].names):
            momentum[k] += - 0.5 * self.step_size * g[j]

        return position, momentum
        
    def hamiltonian_sampling(self, initial_position, logLmin):
        """
        hamiltonian sampling loop to generate the new live point taking nmcmc steps
        """
        self.jumps  = 0
        accepted    = 0
        oldparam    = initial_position.copy()

        # generate the initial momentum from its canonical distribution
        v                = np.atleast_1d(self.momenta_distribution.rvs())
        initial_momentum = self.user.new_point()
        
        for j,k in enumerate(self.positions[0].names):
            initial_momentum[k] = v[j]

        oldmomentum     = initial_momentum.copy()
        starting_energy = self.hamiltonian(oldparam, oldmomentum)
        
        while self.jumps < self.Nmcmc:
            
            newparam, newmomentum = self.constrained_leapfrog_step(oldparam.copy(), oldmomentum.copy(), logLmin)
            newparam.logP         = self.user.log_prior(newparam)
            current_energy        = self.hamiltonian(newparam, newmomentum)

            logp_accept = min(0.0, starting_energy - current_energy)
    
            if logp_accept > np.log(random()):

                oldparam        = newparam
                oldmomentum     = newmomentum
                starting_energy = current_energy
                accepted       += 1
            
            self.jumps+=1
            if self.jumps > self.maxmcmc: break

        self.acceptance = float(accepted)/float(self.jumps)
        self.estimate_nmcmc()
        self.autotune()
        return oldparam

    def metropolis_hastings(self, inParam, logLmin):
        """
        metropolis-hastings loop to generate the new live point taking nmcmc steps
        """
        self.jumps  = 0
        accepted    = 0
        oldparam    = inParam.copy()
        logp_old    = self.user.log_prior(oldparam)
        
        while self.jumps < self.Nmcmc:
            
            newparam        = self.proposals.get_sample(oldparam.copy())
            newparam.logP   = self.user.log_prior(newparam)
            
            if newparam.logP-logp_old + self.proposals.log_J > log(random()):
                newparam.logL = self.user.log_likelihood(newparam)
                
                if newparam.logL > logLmin:
                    oldparam = newparam
                    logp_old = newparam.logP
                    accepted+=1
            
            self.jumps+=1
            
            if self.jumps > self.maxmcmc: break

        self.acceptance = float(accepted)/float(self.jumps)
        self.estimate_nmcmc()
        
        return oldparam
            
    def autotune(self, target = 0.654):
        if self.acceptance < target: self.step_size -= 0.005
        if self.acceptance > target: self.step_size += 0.005    
        if self.step_size < 0.0: self.step_size = 0.00001
