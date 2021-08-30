from mpi4py import MPI
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import time
import sys
import numba


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()
MASTER = 0
FROM_MASTER = 1
FROM_WORKER = 2
G: int = 6.7e-11
np.random.seed(69420)

@numba.njit()
def runge_kutta(pos, vel, masses, n, N, h, r):
    """Fourth order Runge Kutta equations"""
    
    k1v_a, k2v_a = np.zeros((N, 3)), np.zeros((N, 3))
    k3v_a, k4v_a = np.zeros((N, 3)), np.zeros((N, 3))
    pos_n, vel_n = np.zeros(3), np.zeros(3)
    
    # Gravity softening factor
    S = r*0.58*N**-0.26
    
    k1x, k1y, k1z = vel[n, 0], vel[n, 1], vel[n, 2]        
    k1v_a = accelerate(k1v_a, n, pos, masses, N, h, 0, 0, 0, S)
    k1vx, k1vy, k1vz = np.sum(k1v_a[:, 0]), np.sum(k1v_a[:, 1]), np.sum(k1v_a[:, 2])
    
    k2x, k2y, k2z = vel[n, 0] + 0.5*h*k1vx, vel[n, 1] + 0.5*h*k1vy, vel[n, 2] + 0.5*h*k1vz
    k2v_a = accelerate(k2v_a, n, pos, masses, N, h/2, k1vx, k1vy, k1vz, S)
    k2vx, k2vy, k2vz = np.sum(k2v_a[:, 0]), np.sum(k2v_a[:, 1]), np.sum(k2v_a[:, 2])

    k3x, k3y, k3z = vel[n, 0] + 0.5*h*k2vx, vel[n, 1] + 0.5*h*k2vy, vel[n, 2] + 0.5*h*k2vz
    k3v_a = accelerate(k3v_a, n, pos, masses, N, h/2, k2vx, k2vy, k2vz, S)
    k3vx, k3vy, k3vz = np.sum(k3v_a[:, 0]), np.sum(k3v_a[:, 1]), np.sum(k3v_a[:, 2])

    k4x, k4y, k4z = vel[n, 0] + h*k3vx, vel[n, 1] + h*k3vy, vel[n, 2] + h*k3vz
    k4v_a = accelerate(k4v_a, n, pos, masses, N, h, k3vx, k3vy, k3vz, S)
    k4vx, k4vy, k4vz = np.sum(k4v_a[:, 0]), np.sum(k4v_a[:, 1]), np.sum(k4v_a[:, 2])
    
    vel_n[0] = (h/6)*(k1vx + 2*k2vx + 2*k3vx + k4vx) + vel[n, 0]
    vel_n[1] = (h/6)*(k1vy + 2*k2vy + 2*k3vy + k4vy) + vel[n, 1]
    vel_n[2] = (h/6)*(k1vz + 2*k2vz + 2*k3vz + k4vz) + vel[n, 2]
    
    pos_n[0] = (h/6)*(k1x + 2*k2x + 2*k3x + k4x) + pos[n, 0]
    pos_n[1] = (h/6)*(k1y + 2*k2y + 2*k3y + k4y) + pos[n, 1]
    pos_n[2] = (h/6)*(k1z + 2*k2z + 2*k3z + k4z) + pos[n, 2]

    return pos_n, vel_n

@numba.njit()
def accelerate(k, n, pos, masses, N, h, fx, fy, fz, S):
    x, y, z = pos[n, 0], pos[n, 1], pos[n, 2]
    
    for j in range(0, N):
        if j == n:
            k[j, :] = 0
        else:
            k[j, :] = f(x, y, z, pos[j, 0]+h*fx, pos[j, 1]+h*fy, pos[j, 2]+h*fz, masses[j], k[j, :], S)

    return k

@numba.njit()
def f(x1, y1, z1, x2, y2, z2, mass, K, S):

    # R = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 + S )**(1.5)
    # ks1 = 0
    # ks2 = 0
    # s = 0
    # if R == 0:
    #         K = np.zeros(3)
    # else:
    #     for s in range(0,3):
    #         if s == 0:
    #             ks1 = x1
    #             ks2 = x2
    #         if s == 1:
    #             ks1 = y1
    #             ks2 = y2
    #         if s == 2:
    #             ks1 = z1
    #             ks2 = z2
        
    #         K[s] =  -6.7e-11*mass*(ks1- ks2)/R
    
    delta = np.array([(x1 - x2), (y1 - y2), (z1 - z2)])
    R = (delta@delta + S)**1.5
    K = np.zeros(3) if R == 0 else -G*mass*delta/R
    
    return K


class Galaxy:
    
    M: int = 2e29
    r: int = 1e11
    h: int = 2e3
    
    def __init__(self, n: int, timesteps: int):
        """
        Base Galaxy class
        
        Arg:
            n: Number of n-bodies
            timesteps: Timesteps of the simulation
        """
        self.n = n
        self.timesteps = timesteps

    def simulate(self):
        PM = np.zeros((self.n, 3, self.timesteps))
        
        if rank == MASTER:
            # print(f"{Galaxy.M=}, {Galaxy.r=}, {self.n=}")
            masses = Galaxy.M*np.random.rand(self.n)
            M1 = Galaxy.M*1e4
            # print(f"{M1}")
            pos = Galaxy.r*np.random.rand(self.n, 3)
            # print(pos)
            pos[:, 0] = np.random.exponential(Galaxy.r, self.n)
            pos[:, 1] = np.random.exponential(Galaxy.r, self.n)
            pos[:, 2] = np.random.exponential(Galaxy.r, self.n)

            escape_vel = np.sqrt((G*(np.sum(masses)+M1))/(Galaxy.r*np.sqrt(3)))
            print(f"Escape Velocity: {escape_vel}")
            vel = escape_vel*np.random.rand(self.n, 3)
            
            for i in range(0, self.n):
                for j in range(0, 3):
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
                vp = np.sqrt(G*(mass + M1)/(np.sqrt(pos[i, 0]**2 + pos[i, 1]**2)))
                if pos[i, 1] > 0:
                    vel[i, 0], vel[i, 1] = vp*np.cos(theta), -vp*np.sin(theta)
                else:
                    vel[i, 0], vel[i, 1] = -vp*np.cos(theta), vp*np.sin(theta)

            # print(pos)
            # position and velocity of the Sun
            pos[0, 0], pos[0, 1], pos[0, 2] = 0, 0, 0
            vel[0, 0], vel[0, 1], vel[0, 2], masses[0] = 0, 0, 0, M1
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')

            # plt.show()
            # plt.savefig("asd.png")
            # fig = plt.figure()
            # plt.scatter(pos[:, 0], pos[:, 1])
            # plt.show()
            # plt.savefig("asd2d.png")        
        else:
            pos, vel, masses = None, None, None
        
        # print(pos)
        return self._time_evolve(pos, vel, masses, PM, self.n, Galaxy.h)
    
    def _time_evolve(self, pos, vel, masses, PM, N, h):
        if rank == MASTER:
            for k in range(0, self.timesteps):
                
                NCA, NCB = 3, 3
        
                P, V = np.zeros((N, NCA)), np.zeros((N, NCA))
                
                if size < 2:
                    print("Need >= 2 MPI tasks. Quitting...")
                    comm.Abort()
                
                num_workers = size - 1
                ave_row = N//num_workers
                extra = N%num_workers
                offset = 0
                print(f"MASTER: {num_workers=}")
                print(f"MASTER: {ave_row=}")
                print(f"MASTER: {extra=}")
                for dest in range(1, num_workers+1):
                    rows = ave_row
                    if dest <= extra:
                        rows += 1
                    # print(f"{dest=}")
                    comm.send(offset, dest=dest, tag=FROM_MASTER)
                    comm.send(rows, dest=dest, tag=FROM_MASTER)
                    
                    # Delegate arrays for workers compute
                    comm.Send(pos, dest=dest, tag=FROM_MASTER)
                    comm.Send(vel, dest=dest, tag=FROM_MASTER)
                    comm.Send(masses, dest=dest, tag=FROM_MASTER)
                    offset += rows
                    # print(offset)
                    
                for i in range(1, num_workers+1):
                    # print(f"{i=}")
                    offset = comm.recv(source=i, tag=FROM_WORKER)
                    rows = comm.recv(source=i, tag=FROM_WORKER)
                    # print(f"{offset=}")
                    # print(f"{rows=}")
                    # Receive arrays from workers once they finished
                    comm.Recv([P[offset:, :], rows*NCB, MPI.DOUBLE], source=i, tag=FROM_WORKER)
                    comm.Recv([V[offset:, :], rows*NCB, MPI.DOUBLE], source=i, tag=FROM_WORKER)
                    # print(f"MASTER: {P[offset:, :]=}")
                
                # print(P, V)
                # return P, V
                
                pos, vel = P, V
                # print(pos)
                # print(pos[:, 0])
                PM[:, 0, k], PM[:, 1, k], PM[:, 2, k] = pos[:, 0], pos[:, 1], pos[:, 2]
               
            return PM
        
        if rank > MASTER:
            for i in range(0, self.timesteps):
                # self._run_worker(pos, vel, masses, N, h)
                NCA = 3
                P = np.zeros((N, NCA))
                V = np.zeros((N, NCA))
                
                masses = np.zeros(N)
                pos = np.zeros((N, 3))
                vel = np.zeros((N, 3))
                
                offset = comm.recv(source=MASTER, tag=FROM_MASTER)
                rows = comm.recv(source=MASTER, tag=FROM_MASTER)
                print(f"worker {rank} : \n {offset=}, {rows=}")
                # Receive arrays from master
                comm.Recv([pos, N*NCA, MPI.DOUBLE], source=MASTER, tag=FROM_MASTER)
                comm.Recv([vel, N*NCA, MPI.DOUBLE], source=MASTER, tag=FROM_MASTER)
                comm.Recv([masses, N, MPI.DOUBLE], source=MASTER, tag=FROM_MASTER)
                print(f"\nworker {rank} : \n {pos=}, \n {vel=}")
                for n in range(offset, offset+rows):
                    P[n, :], V[n, :] = runge_kutta(pos, vel, masses, n, N, h, Galaxy.r)
                    print(f"\nworker {rank} : \n P[{n}, :] = {P[n, :]}")
                comm.send(offset, dest=MASTER, tag=FROM_WORKER)
                comm.send(rows, dest=MASTER, tag=FROM_WORKER)
                comm.Send(P[offset:(offset+rows), :], dest=MASTER, tag=FROM_WORKER)
                comm.Send(V[offset:(offset+rows), :], dest=MASTER, tag=FROM_WORKER)

    
    # def _run_master(self, pos, vel, masses, N, h):
        
    #     NCA, NCB = 3, 3
        
    #     P, V = np.zeros((N, NCA)), np.zeros((N, NCA))
        
    #     if size < 2:
    #         print("Need >= 2 MPI tasks. Quitting...")
    #         comm.Abort()
        
    #     num_workers = size - 1
    #     ave_row = N//num_workers
    #     extra = N%num_workers
    #     offset = 0
    #     print(f"MASTER: {num_workers=}")
    #     print(f"MASTER: {ave_row=}")
    #     print(f"MASTER: {extra=}")
    #     for dest in range(1, num_workers+1):
    #         rows = ave_row
    #         if dest <= extra:
    #             rows += 1
    #         # print(f"{dest=}")
    #         comm.send(offset, dest=dest, tag=FROM_MASTER)
    #         comm.send(rows, dest=dest, tag=FROM_MASTER)
            
    #         # Delegate arrays for workers compute
    #         comm.Send(pos, dest=dest, tag=FROM_MASTER)
    #         comm.Send(vel, dest=dest, tag=FROM_MASTER)
    #         comm.Send(masses, dest=dest, tag=FROM_MASTER)
    #         offset += rows
    #         # print(offset)
            
    #     for i in range(1, num_workers+1):
    #         # print(f"{i=}")
    #         offset = comm.recv(source=i, tag=FROM_WORKER)
    #         rows = comm.recv(source=i, tag=FROM_WORKER)
    #         # print(f"{offset=}")
    #         # print(f"{rows=}")
    #         # Receive arrays from workers once they finished
    #         comm.Recv([P[offset:, :], rows*NCB, MPI.DOUBLE], source=i, tag=FROM_WORKER)
    #         comm.Recv([V[offset:, :], rows*NCB, MPI.DOUBLE], source=i, tag=FROM_WORKER)
    #         # print(f"MASTER: {P[offset:, :]=}")
        
    #     # print(P, V)
    #     return P, V


    # def _run_worker(self, pos, vel, masses, N, h):
    #     NCA = 3
    #     P = np.zeros((N, NCA))
    #     V = np.zeros((N, NCA))
        
    #     masses = np.zeros(N)
    #     pos = np.zeros((N, 3))
    #     vel = np.zeros((N, 3))
        
    #     offset = comm.recv(source=MASTER, tag=FROM_MASTER)
    #     rows = comm.recv(source=MASTER, tag=FROM_MASTER)
    #     print(f"worker {rank} : \n {offset=}, {rows=}")
    #     # Receive arrays from master
    #     comm.Recv([pos, N*NCA, MPI.DOUBLE], source=MASTER, tag=FROM_MASTER)
    #     comm.Recv([vel, N*NCA, MPI.DOUBLE], source=MASTER, tag=FROM_MASTER)
    #     comm.Recv([masses, N, MPI.DOUBLE], source=MASTER, tag=FROM_MASTER)
    #     print(f"\nworker {rank} : \n {pos=}, \n {vel=}")
    #     for n in range(offset, offset+rows):
    #         P[n, :], V[n, :] = runge_kutta(pos, vel, masses, n, N, h, Galaxy.r)
    #         print(f"\nworker {rank} : \n P[{n}, :] = {P[n, :]}")
    #     comm.send(offset, dest=MASTER, tag=FROM_WORKER)
    #     comm.send(rows, dest=MASTER, tag=FROM_WORKER)
    #     comm.Send(P[offset:(offset+rows), :], dest=MASTER, tag=FROM_WORKER)
    #     comm.Send(V[offset:(offset+rows), :], dest=MASTER, tag=FROM_WORKER)
    
    def _rando(self):
        return 1 if np.random.rand() < 0.5 else -1            

    
if __name__ == "__main__":
    N = int(sys.argv[1])
    timesteps = 2
    galaxy = Galaxy(N, timesteps)
    data = np.asarray(galaxy.simulate())

    if rank == MASTER:
        t1 = time.time()
        print("Making video...")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        K = 1
        B = np.zeros((N,3,int(timesteps/K)))
        
        for i in range (0,int(timesteps/K)):
            B[:,:,i] = data[:,:,K*i]

        def update_lines(num, dataLines, lines):
            for line, data in zip(lines, dataLines):
                # NOTE: there is no .set_data() for 3 dim data...
                line.set_data(data[0:2, (num-1):num])
                line.set_3d_properties(data[2, (num-1):num])
            return lines
    
        # Attaching 3D axis to the figure
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        
        data = [B[i] for i in np.arange(N)]
        
        # NOTE: Can't pass empty arrays into 3d version of plot()
        lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], marker = 'o', markersize = 0.5, color = 'w', alpha = 1)[0] for dat in data]
        
        # Setting the axes properties
        lim = 2*Galaxy.r
        
        ax.set_xlim3d([-lim, lim])
        ax.set_ylim3d([-lim, lim])
        ax.set_zlim3d([-lim, lim])
        ax.set_facecolor('xkcd:black')
        ax.set_axis_off()
        # Creating the Animation object
        line_ani = animation.FuncAnimation(fig, update_lines, int (timesteps/K), fargs=(data, lines), interval=1, blit=True)
        line_ani.save('gravity_sim.mp4', writer=writer) # add this in if you want a vid
        plt.show()
        print(f"Video took {time.time() - t1}s to make.")