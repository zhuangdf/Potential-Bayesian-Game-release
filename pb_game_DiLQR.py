import numpy as np
import time
from Utilities.pb_DiLQR_Utilities import *
from multiprocessing import Process, Array, Barrier
import os
from scenarios import *

class DistributedOptimization:
    def __init__(self,n,m,T,init_states,init_inputs,ref_trajs,N,graph,prob,scene="M"):
        
        self.scene = scene
        self.prob = prob
        self.agent_list = []
        self.n, self.m, self.T, self.n_m = n, m, T, n+m
        self.graph = graph
        self.N = N
        self.d_circle = np.array([3.425,-0.46])
        self.pair_num = int(N*(N-1)/2) * (len(self.d_circle)**2)
        self.gamma = 0.5
        self.alpha = 1.0
        # self.d_ref = 6.0
        self.ref_trajs = ref_trajs
        self.upper_bound = np.array([0.6,3.8])
        self.lower_bound = np.array([-0.6,-7.6])

        if self.scene == "C":
            self.d_ref = 6.5
            self.C_matrix = np.diag([1.,1.,0.,0.,10.,0.1])
            self.CT_matrix = np.diag([10.,10.,0.,0.,10.,0.1])
        if self.scene == "M":
            self.d_ref = 4.5
            self.C_matrix = np.diag([0.,1.,0.,2.,10.,0.1])
            self.CT_matrix = np.diag([0.,1.,0.,2.,10.,0.1])
        
        
        for i in range(N):
            self.agent_list.append(Agent(n,m,T,init_states[i],init_inputs[i],ref_trajs[i],
                                         N,i,self.lower_bound,self.upper_bound,self.d_circle,self.graph,\
                                             self.prob,self.C_matrix,self.CT_matrix))

        self.beta = 1.4
        self.trajectory_list = []
        self.show_convergence = True
        self.y_var_list=[]

    def solve(self, max_iter=100, max_admm_iter=25, criterion1=100.0,criterion2=1.0):

        with_collision = True
        for ite in range(max_iter):
            t1 = time.time()
            agent_trajectory_all = np.zeros((self.N,self.T,self.n_m))
            for i in range(self.N):
                agent_trajectory_all[i] = self.agent_list[i].get_traj()

                self.agent_list[i].sigma = 1. if ite < 3 else 0.03
                self.agent_list[i].rho = 1./self.N if ite < 3 else 0.03/self.N
            
            self.cost_last = self.get_cost(agent_trajectory_all,with_collision,self.beta)
            t2 = time.time()
            for i in range(self.N):
                curr_agent = self.agent_list[i]
                curr_agent.update_all_traj(agent_trajectory_all)
                curr_agent.approximation(self.d_ref,with_collision,self.beta)
                curr_agent.reset()
            t3 = time.time()
            admm_iter_num = max_admm_iter if with_collision else 1

            for admm_iter in range(admm_iter_num):

                y_all = np.zeros((self.N,self.pair_num*self.T+self.N*self.T*self.m))
                for i in range(self.N):
                    y_all[i] = self.agent_list[i].get_y()

                for i in range(self.N):
                    curr_agent = self.agent_list[i]
                    curr_agent.update_all_y(y_all)
                    curr_agent.admm_iter(with_collision)
            t4 = time.time()
            all_trajectory, overall_cost = self.exact_line_search(2,with_collision,self.lower_bound,self.upper_bound)

            if self.show_convergence:
                y_all = np.zeros((self.N,self.pair_num*self.T+self.N*self.T*self.m))
                for i in range(self.N):
                    y_all[i] = self.agent_list[i].get_y()
                self.y_var_list.append(np.mean(np.var(y_all,axis=0)))
            
            for i in range(self.N):
                self.agent_list[i].update_traj(all_trajectory[i])

            delta_cost = overall_cost - self.cost_last
            print(ite, self.cost_last, overall_cost)
            
            self.trajectory_list.append(self.get_traj())
            if (with_collision and abs(delta_cost) < criterion2):
                break
            t5 = time.time()
            print(t2-t1,t3-t2,t4-t3,t5-t4)
        
    @staticmethod
    def process(agent,barrier,all_traj,all_y,candidate_traj,ref_trajs,**kwargs):

        N, T, n_m = agent.N, agent.T, agent.n_m
        pair_num = int(N*(N-1)/2) * (len(agent.d_circle)**2)
        arr_traj = np.frombuffer(all_traj.get_obj()).reshape((N,T,n_m))
        arr_y = np.frombuffer(all_y.get_obj()).reshape(N,pair_num*T+N*T*2)
        arr_candidate_traj = np.frombuffer(candidate_traj.get_obj()).reshape((kwargs['max_line_search'],N,T,n_m))
        barrier.wait()

        print("agent %d has started!"%(agent.id))
        

        t0 = time.time()

        with_collision = True
        for ite in range(kwargs['max_iter']):

            agent.sigma = 1 if ite < 3 else 0.03
            agent.rho = 1./N if ite < 3 else 0.03/N
            
            arr_traj[agent.id] = agent.get_traj()
            barrier.wait()
            cost_last = get_cost_static(
                arr_traj,kwargs['d_ref'],ref_trajs,agent.C_matrix,agent.CT_matrix,\
                    with_collision,kwargs['beta'],agent.d_circle,agent.prob)
            
            agent.update_all_traj(arr_traj)
            agent.approximation(kwargs['d_ref'],with_collision,kwargs['beta'])
            agent.reset()

            admm_iter_num = kwargs['max_admm_iter'] if with_collision else 1
            for admm_iter in range(admm_iter_num):
                if with_collision:
                    arr_y[agent.id] = agent.get_y()
                    barrier.wait()
                    agent.update_all_y(arr_y)
                agent.admm_iter(with_collision)

            alpha, gamma = kwargs['alpha'], kwargs['gamma']
            cost_list, trajectory_list = [], []
            for i in range(kwargs['max_line_search']):
                trajectory = agent.compute_traj(alpha)
                trajectory_list.append(trajectory)
                arr_candidate_traj[i][agent.id] = trajectory
                alpha *= gamma
            barrier.wait()

            for i in range(kwargs['max_line_search']):
                cost = get_cost_static(
                    arr_candidate_traj[i],kwargs['d_ref'],ref_trajs,agent.C_matrix,agent.CT_matrix,\
                        with_collision,kwargs['beta'],agent.d_circle,agent.prob)
                cost_list.append(cost)

            index = np.argmin(np.array(cost_list))
            agent.update_traj(trajectory_list[index])
            delta_cost = cost_list[index] - cost_last

            if agent.id == 0:
                print(ite, cost_last, cost_list[index])

            if abs(delta_cost)<kwargs['criterion2']:
                break

        arr_traj[agent.id] = agent.get_traj()
        t12 = time.time()
        print("agent %d has finished! %f second used."%(agent.id,t12-t0))

    def mp_solve(self, max_iter=100, max_admm_iter=25, criterion1=100.0, criterion2=1.0, max_line_search=2):
        barrier = Barrier(self.N)
        all_traj = Array('d',self.N*self.T*self.n_m)
        candidate_traj = Array('d',max_line_search*self.N*self.T*self.n_m)
        all_y = Array('d',self.N*(self.pair_num*self.T+self.N*self.T*self.m))

        hyper_parameters = {'max_iter':max_iter,
                            'max_admm_iter':max_admm_iter,
                            'criterion1':criterion1,
                            'criterion2':criterion2,
                            'max_line_search':max_line_search,
                            'alpha':self.alpha,
                            'gamma':self.gamma,
                            'beta':self.beta,
                            'd_ref':self.d_ref,
                            'lb':self.lower_bound,
                            'ub':self.upper_bound}

        process_pool = [Process(target=self.process,\
            args=(self.agent_list[i],barrier,all_traj,all_y,candidate_traj,self.ref_trajs),\
            kwargs=hyper_parameters) for i in range(self.N)]

        for i in range(self.N):
            process_pool[i].start()
            # os.system("taskset -p -c %d %d" % (i+100, process_pool[i].pid))
            #os.system("sudo renice -20 -p %s" % (process_pool[i].pid))
        
        for i in range(self.N):
            process_pool[i].join()
        
        self.trajectory = np.frombuffer(all_traj.get_obj()).reshape((self.N,self.T,self.n_m))

    def exact_line_search(self, maximum_line_search, with_collision, lower_bound, upper_bound):
        alpha = self.alpha
        all_trajectory_list = []
        overall_cost_list = []
        for k in range(maximum_line_search):
            all_trajectory = np.zeros((self.N,self.T,self.n_m))
            for i in range(self.N):
                all_trajectory[i] = self.agent_list[i].compute_traj(alpha)
            all_trajectory_list.append(all_trajectory)
            overall_cost = self.get_cost(all_trajectory, with_collision, self.beta)
            overall_cost_list.append(overall_cost)
            alpha *= self.gamma

        overall_cost_list = np.array(overall_cost_list)
        index = np.argmin(overall_cost_list)
        return all_trajectory_list[index], overall_cost_list[index]

    def exact_line_search_agent(self, i, maximum_line_search):
        alpha = 1.0
        trajectory_list = []
        overall_cost_list = []
        for k in range(maximum_line_search):
            trajectory = self.agent_list[i].compute_traj(alpha)
            trajectory_list.append(trajectory)
            overall_cost = self.get_cost_agent(i, trajectory)
            overall_cost_list.append(overall_cost)
            alpha *= self.gamma

        overall_cost_list = np.array(overall_cost_list)
        index = np.argmin(overall_cost_list)
        return trajectory_list[index], overall_cost_list[index]

    def get_cost(self, all_trajectory, with_collision, beta):
        return get_cost_static(all_trajectory,self.d_ref,self.ref_trajs,\
            self.C_matrix,self.CT_matrix,with_collision,beta,self.d_circle,self.prob)
    

    def get_traj(self):
        agent_trajectory_all = np.zeros((self.N,self.T,self.n_m))
        for i in range(self.N):
            agent_trajectory_all[i,:,:] = self.agent_list[i].get_traj()
        return agent_trajectory_all

class Agent:
    def __init__(self,n,m,T,init_state,init_input,ref_traj,N,id,lb,ub,d_circle,graph,prob,C,CT):
        self.n, self.m, self.T, self.n_m = n, m, T, n+m
        self.init_state = init_state
        self.init_input = init_input
        self.prob = prob
        self.graph = graph
        self.N = N
        self.id = id
        self.all_trajectory = np.zeros((N,T,n+m),dtype=np.float64)
        self.bound = np.zeros((N,T,m),dtype=np.float64).reshape(-1)
        self.vehicle, self.x_u = multiVehicle(1)
        self.dynamic_model = MultiDynamicModelWrapper(self.vehicle, self.x_u, init_state, init_input, T)
        self.trajectory = self.dynamic_model.eval_traj()
        self.K_matrix = np.zeros((T,m,n),dtype=np.float64)
        self.k_vector = np.zeros((T,m),dtype=np.float64)
        self.d_circle = d_circle
        pair_num = int(N*(N-1)/2) * (len(d_circle)**2)
        self.pair_num = pair_num
        self.p = np.zeros(pair_num*T+N*T*m,dtype=np.float64)
        self.y = np.zeros(pair_num*T+N*T*m,dtype=np.float64)
        self.all_y = np.zeros((N,pair_num*T+N*T*m),dtype=np.float64)
        self.z = np.zeros(pair_num*T+N*T*m,dtype=np.float64)
        self.s = np.zeros(pair_num*T+N*T*m,dtype=np.float64)
        self.r = np.zeros(pair_num*T+N*T*m,dtype=np.float64)
        self.l = np.zeros(pair_num*T,dtype=np.float64)
        self.J = np.zeros((T,pair_num,self.n),dtype=np.float64)
        self.LQR = LQRWrapper(n,m,T,np.zeros(self.n_m,dtype=np.float64))
        self.ref_traj = ref_traj
        
        self.C_matrix = C
        self.CT_matrix = CT

        self.upper_bound = ub
        self.lower_bound = lb

    def get_traj(self):
        return self.trajectory.copy()

    def get_delta_traj(self):
        return self.delta_trajectory.copy()

    def get_y(self):
        return self.y.copy()

    def get_l(self):
        return self.l.copy()

    def get_jacobian(self):
        return self.J.copy()
    
    def update_traj(self, trajectory):
        self.trajectory = trajectory

    def update_all_traj(self, all_trajectory):
        self.all_trajectory = all_trajectory

    def update_all_y(self, all_y):
        self.all_y = all_y
    
    def update_coeff(self, decay):
        self.sigma = self.sigma * decay
        self.rho = self.rho * decay

    def reset(self):
        self.p = np.zeros(self.pair_num*self.T+self.N*self.T*self.m)
        #self.y = np.zeros(self.pair_num*self.T)
        #self.all_y = np.zeros((self.N,self.pair_num*self.T))
        #self.z = np.zeros(self.pair_num*self.T)
        self.s = np.zeros(self.pair_num*self.T+self.N*self.T*self.m)
    
    def approximation(self, d_ref, with_collision, beta):
        self.F_matrix = self.dynamic_model.eval_grad(self.trajectory)
        if with_collision:
            self.l, self.J = compute_l_J(self.n,self.all_trajectory,self.id,d_ref,beta,\
                self.d_circle,self.graph,self.prob)
            self.control = self.all_trajectory[:,:,self.n:]
            self.control_upper_bound = (self.upper_bound - self.control).reshape(-1)
            self.control_lower_bound = (self.lower_bound - self.control).reshape(-1)
        else:
            self.l = np.zeros(self.pair_num*self.T)
            self.J = np.zeros((self.T,self.pair_num,self.n))

    def admm_iter(self, with_collision):


        '''all_y_sum = np.sum(self.all_y,axis=0)
        self.p = self.p + self.rho*(self.N*self.y-all_y_sum)
        self.s = self.s + self.sigma*(self.y-self.z)
        self.r = self.rho * ((self.N-2)*self.y+all_y_sum) + self.sigma * self.z - self.p - self.s'''
        self.p, self.s, self.r = compute_psr(self.all_y,self.p,\
            self.s,self.r,self.y,self.z,self.N,self.rho,self.sigma)
        coeff = 1.0/(self.sigma+2*self.rho*(self.N-1))
        All_C_matrix, All_c_vector, coeff = compute_C(self.trajectory,self.ref_traj,\
            self.J,self.r,coeff,self.id,self.graph,self.prob,self.C_matrix,self.CT_matrix)
        self.LQR.update_F(self.F_matrix)
        self.LQR.update_C(All_C_matrix,All_c_vector)
        self.K_matrix, self.k_vector = self.LQR.backward_pass()
        self.delta_trajectory = self.LQR.forward_pass()
        self.y = compute_y(self.delta_trajectory,self.J,self.r,coeff,self.id)
        self.z = compute_z(self.N,self.s,self.sigma,self.y,\
            self.l,self.control_lower_bound,self.control_upper_bound,self.pair_num,self.T)
        '''division = self.pair_num * self.T
        self.z[:division] = 2.0/(2.0*self.N*self.sigma+1)*(self.N*(self.s[:division]+self.sigma*self.y[:division])+self.l)
        self.z[division:] = np.maximum(self.N*(self.s[division:]+self.sigma*self.y[division:]),self.control_lower_bound)
        self.z[division:] = np.minimum(self.z[division:],self.control_upper_bound)
        self.z[division:] = self.s[division:]/self.sigma+self.y[division:]-self.z[division:]/(self.N*self.sigma)'''

    def compute_traj(self, alpha):
        return self.dynamic_model.update_traj(self.trajectory, self.K_matrix, self.k_vector, self.lower_bound, self.upper_bound, alpha)

def positive_int(value):
    """Parse a strictly positive iteration count for the command line."""
    import argparse

    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    """Run a merge (M) or intersection (C) trajectory planning demo."""
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Distributed iLQR potential Bayesian game trajectory planning demo.",
        epilog=("Examples: python pb_game_DiLQR.py C --no-video; "
                "python pb_game_DiLQR.py M --solver single --no-plot. "
                "Plots are written to result.jpg and result.mp4 in the current directory."),
    )
    parser.add_argument(
        "scene", nargs="?", default="M", type=str.upper, choices=("C", "M"),
        help="C: three-car intersection; M: two-car merge (default: M)",
    )
    parser.add_argument(
        "--solver", choices=("single", "multi", "both"), default="multi",
        help="single process, multiprocessing, or compare both (default: multi)",
    )
    parser.add_argument("--max-iter", type=positive_int, default=500,
                        help="maximum outer iterations (default: 500)")
    parser.add_argument("--admm-iter", type=positive_int, default=2,
                        help="ADMM iterations per outer iteration (default: 2)")
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="absolute cost-change stopping tolerance (default: 0.1)")
    parser.add_argument("--no-video", action="store_true",
                        help="save only result.jpg; skip MP4 export")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip both image and video output")
    args = parser.parse_args(argv)
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be finite and greater than zero")
    if not args.no_plot and not args.no_video and shutil.which("ffmpeg") is None:
        parser.error("MP4 export requires ffmpeg; install it or use --no-video")

    n, m, T = 4, 2, 100
    if args.scene == "C":
        start_points = np.array([
            [3, -18, np.pi / 2, 3],
            [-12, -3, 0, 3],
            [24, 3, np.pi, 3],
        ])
        distribution = [
            [[3.0, 0.1, 1.0]],
            [[2.4, 0.2, 0.1], [3.6, 0.2, 0.9]],
            [[2.4, 0.2, 0.9], [3.6, 0.2, 0.1]],
        ]
        scenario = cross_2
    else:
        start_points = np.array([[0, 0, 0, 3], [0, 4, 0, 3]])
        distribution = [[3.5, 0.2, 0.5], [2.5, 0.2, 0.5]]
        scenario = merge

    start_points, ref_trajs, graph, prob, p_list, draw, figsize = scenario(
        start_points, distribution, T
    )
    N = len(graph)
    print(f"Scene: {args.scene}; type nodes: {N}; horizon: {T} steps (dt=0.1 s)")
    solvers = ("single", "multi") if args.solver == "both" else (args.solver,)
    for solver in solvers:
        optimization = DistributedOptimization(
            n, m, T, start_points, np.zeros((N, T, m)),
            ref_trajs, N, graph, prob, args.scene,
        )
        print(f"Starting {solver} solver (first run includes Numba compilation)...")
        started = time.perf_counter()
        solve = optimization.solve if solver == "single" else optimization.mp_solve
        solve(max_iter=args.max_iter, max_admm_iter=args.admm_iter,
              criterion2=args.tolerance)
        trajectory = (optimization.get_traj() if solver == "single"
                      else optimization.trajectory)
        print(f"{solver} solver finished in {time.perf_counter() - started:.3f} s")

    if not args.no_plot:
        plot(trajectory, np.diagonal(prob), p_list, draw, figsize,
             save=not args.no_video)
        print("Saved result.jpg" + (" and result.mp4" if not args.no_video else ""))
    return trajectory


if __name__ == "__main__":
    main()
