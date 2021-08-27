from numpy.lib.utils import source
from mpi4py import MPI
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()
MASTER = 0
FROM_MASTER = 1
FROM_WORKER = 2


class Galaxy:
    
    G = 6.7*1e-11
    M = 2e29
    r = 1e11
    
    def __init__(self, n, timesteps):
        """
        Base Galaxy class
        
        Arg:
            n: Number of n-bodies
            timesteps: Timesteps of the simulation
        """
        self.n = n
        self.timesteps = timesteps

    def _setup_parameters(self):
        PM = np.zeros((self.n, 3, self.timesteps))
        
        if rank == MASTER:
            masses = Galaxy.M*np.random.rand(self.n)
            M1 = Galaxy.M*1e4
            pos = Galaxy.r*np.random.rand(self.n, 3)
            pos[:, 0] = np.random.exponential(Galaxy.r, self.n)
            pos[:, 1] = np.random.exponential(Galaxy.r, self.n)
            pos[:, 1] = np.random.exponential(Galaxy.r, self.n)
            
            escape_vel = np.sqrt((Galaxy.G*(np.sum(masses)+M1))/(Galaxy.r*np.sqrt(3)))
            print(f"Escape Velocity: {escape_vel}")
            vel = escape_vel*np.random.rand(self.n, 3)
            
            for i in range(0, self.n):
                for j in range(0, self.n):
                    pos[i, j] *= self._rando()
                    vel[i, j] *= self._rando()
            
            pos[:, 2] /= 20
            vel[:, 2] /= np.sqrt(20)
            
            # generates circular motion velocities for planets
            for i in range(0, self.n):
                theta = np.arctan(pos[i, 0]/pos[i, 1])
                R = np.sqrt(pos[i, 0]**2 + pos[i, 1]**2)
                mass = 0
                for j in range(0, self.n):
                    rp = np.sqrt(pos[j, 0]**2 + pos[j, 1]**2)
                    if rp < R:
                        mass += masses[j]
                vp = np.sqrt(Galaxy.G*(mass + M1)/(np.sqrt(pos[i, 0]**2 + pos[i, 1]**2)))
                if pos[i, 1] > 0:
                    vel[i, 0], vel[i, 1] = vp*np.cos(theta), -vp*np.sin(theta)
                else:
                    vel[i, 0], vel[i, 1] = -vp*np.cos(theta), vp*np.sin(theta)
            
            # position and velocity of the Sun
            pos[0, 0], pos[0, 1], pos[0, 2], vel[0, 0], vel[0, 1], vel[0, 2], masses[0] = 0,0,0,0,0,0,M1
        
        return 
    
    def _time_evolve(self, pos, vel, masses, PM, n, h, timesteps):
        if rank == MASTER:
            for i in tqdm(range(0, timesteps)):
                pos, vel = 
        
    def _rando(self):
        return 1 if np.random.rand() < 0.5 else 1
    
    def _run_master(self, pos, vel, masses, n, h):
        # pos, vel = 
        pass
    
    def _MPI_A(self, h, pos, vel, masses, n):
        NCA, NCB = 3, 3
        
        P, V = np.zeros((N, NCA)), np.zeros((N, NCB))
        
        if size < 2:
            print("Need at least two MPI tasks. Quitting...")
            comm.Abort()
        
        num_workers = size - 1
        ave_row = self.n//num_workers
        extra = N%num_workers
        offset = 0
        
        for dest in range(1, num_workers+1):
            rows = ave_row
            if dest <= extra:
                rows += 1
            comm.send(offset, dest=dest, tag = FROM_MASTER)
            comm.send(rows, dest=dest, tag=FROM_MASTER)
            comm.Send(pos, dest=dest, tag=FROM_MASTER)
            comm.Send(vel, dest=dest, tag=FROM_MASTER)
            comm.Send(masses, dest=dest, tag=FROM_MASTER)
            offset += rows
            
        for i in range(1, num_workers+1):
            source = i
            offset = comm.recv(source=source, tag=FROM_MASTER)
            rows = comm.recv(source=source, tag=FROM_WORKER)
            comm.Recv([P[offset:, :], rows*NCB, MPI.DOUBLE], source=source, tag=FROM_WORKER)
            comm.Recv([V[offset:, :], rows*NCB, MPI.DOUBLE], source=source, tag=FROM_WORKER)
        
        return P, V
    
    def _run_worker(self, pos, vel, masses, n, h):
        NCA = 3
        P = np.zeros((self.n, NCA))
        V = np.zeros((self.n, NCA))
        
        masses = np.zeros(self.n)
        pos = np.zeros((self.n, 3))
        vel = np.zeros((self.n, 3))
        
        offset = comm.recv(source=MASTER, tag=FROM_MASTER)
        rows = comm.recv(source=MASTER, tag=FROM_MASTER)
        comm.Recv([pos, self.n*NCA, MPI.DOUBLE], source=MASTER)
        
        
if __name__ == "__main__":
    pass