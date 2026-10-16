from numba import njit
import numpy as np
from math import sin, cos, sqrt, asin
import sympy as sp
from sympy import Array, Matrix

class MultiDynamicModelWrapper(object):
    def __init__(self, dynamic_model_function, x_u_var, init_state, init_inputs, T):
        self.init_state = init_state
        self.init_inputs = init_inputs
        self.n = int(init_state.shape[0])
        self.m = int(len(x_u_var) - self.n)
        self.T = T
        # self.dynamic_model_lamdify = njit(sp.lambdify([x_u_var],dynamic_model_function,'numpy'))
        # grad_dynamic_model_function = sp.transpose(sp.derive_by_array(dynamic_model_function, x_u_var))
        # self.grad_dynamic_model_lamdify = njit(sp.lambdify([x_u_var], grad_dynamic_model_function, "numpy"))
        # print(self.dynamic_model_lamdify.__doc__)
        # print(self.grad_dynamic_model_lamdify.__doc__)

    def eval_traj(self, init_state=None, input_traj=None):
        if init_state is None:
            init_state = self.init_state
        if input_traj is None:
            input_traj = self.init_inputs
        return eval_traj_static(init_state, input_traj, self.m, self.n)

    def update_traj(self, old_traj, K_matrix_all, k_vector_all, lb, ub, alpha):
        return update_traj_static(self.m, self.n, old_traj, K_matrix_all, k_vector_all, lb, ub, alpha)

    def eval_grad(self, traj):
        return eval_grad_static(traj)
    
def multiVehicle(num_of_vehicle, h_constant = 0.1, d_constanT = 2.700):
    x_u = Array(sp.symarray('x_u',(num_of_vehicle,6)))

    prod = lambda x,y:Array(Matrix(x).multiply_elementwise(Matrix(y)))[:,0]
    sq = lambda x:x.applyfunc(lambda y:y**2)
    sqrt = lambda x:x.applyfunc(sp.sqrt)
    cos = lambda x:x.applyfunc(sp.cos)
    sin = lambda x:x.applyfunc(sp.sin)
    asin = lambda x:x.applyfunc(sp.asin)
    def add(x,n):
        return x.applyfunc(lambda y:y+n)

    h_constant = 0.1
    d_constanT = d_constanT
    h_d_constanT = h_constant/d_constanT

    b_function=add(h_constant*prod(x_u[:,3],cos(x_u[:,4])),d_constanT)\
        - sqrt(add(-(h_constant**2)*prod(sq(x_u[:,3]),sq(sin(x_u[:,4]))),d_constanT**2))

    system = sp.transpose(Array([
            x_u[:,0] + prod(b_function,cos(x_u[:,2])),
            x_u[:,1] + prod(b_function,sin(x_u[:,2])),
            x_u[:,2] + asin(h_d_constanT*prod(x_u[:,3],sin(x_u[:,4]))),
            x_u[:,3] + h_constant*x_u[:,5],
        ])).reshape(4*num_of_vehicle)

    x = np.array(x_u[:,0:4].reshape(4*num_of_vehicle))
    u = np.array(x_u[:,4:6].reshape(2*num_of_vehicle))
    x_u = Array(np.concatenate((x,u)))

    return system, x_u

class LQRWrapper(object):
    def __init__(self,n,m,T,init_state):
        self.n = n
        self.m = m
        self.m_n = self.n + self.m
        self.T = T
        self.C_matrix = np.zeros((T, self.m_n, self.m_n),dtype=np.float64)
        self.c_vector = np.zeros((T, self.m_n),dtype=np.float64)
        self.F_matrix = np.zeros((T, self.n, self.m_n),dtype=np.float64)
        self.init_state = init_state

    def update_C(self, C_matrix, c_vector):
        self.C_matrix = C_matrix
        self.c_vector = c_vector

    def update_F(self, F_matrix):
        self.F_matrix = F_matrix

    def backward_pass(self):
        self.K_matrix, self.k_vector = backward_pass_static(self.m,self.n,self.T,self.C_matrix,self.c_vector,self.F_matrix)
        return self.K_matrix, self.k_vector

    def forward_pass(self):
        self.trajectory = forward_pass_static(self.m,self.n,self.T,self.K_matrix,self.k_vector,self.F_matrix,self.init_state)
        return self.trajectory
    
    def get_traj(self):
        return self.trajectory.copy()

@njit(cache=True)
def compute_psr(all_y,p,s,r,y,z,N,rho,sigma):
    all_y_sum = np.sum(all_y,axis=0)
    p = p + rho*(N*y-all_y_sum)
    s = s + sigma*(y-z)
    r = rho * ((N-2)*y+all_y_sum) + sigma * z - p - s
    return p,s,r

@njit(cache=True)
def compute_z(N,s,sigma,y,l,lb,ub,pair_num,T):
    new_z = np.zeros_like(y)
    division = pair_num * T
    new_z[:division] = 2.0/(2.0*N*sigma+1)*(N*(s[:division]+sigma*y[:division])+l)
    new_z[division:] = np.maximum(N*(s[division:]+sigma*y[division:]),lb)
    new_z[division:] = np.minimum(new_z[division:],ub)
    new_z[division:] = s[division:]/sigma+y[division:]-new_z[division:]/(N*sigma)
    return new_z

@njit(cache=True)
def compute_y(delta_trajectory,J,r,coeff,id):

    T, pair_num = len(J),len(J[0])
    new_y = coeff*r
    for tau in range(T):
        a = tau*pair_num
        b = (tau+1)*pair_num
        new_y[a:b] += coeff*J[tau]@delta_trajectory[tau,:4]
        c = T*pair_num+id*T*2+tau*2
        new_y[c:c+2] += coeff*delta_trajectory[tau,4:]

    return new_y

@njit(cache=True)
def compute_C(trajectory,ref_traj,J,r,coeff,id,graph,prob,C,CT):

    T, n_m, pair_num = len(trajectory), len(trajectory[0]), len(J[0])
    All_C_matrix = np.zeros((T, n_m, n_m))
    All_c_vector = np.zeros((T, n_m))
    C_host = prob[id,id]*C # host cost is sth like (x_u-r).T@C@(x_u-r)

    for tau in range(T):
        if tau == T-1:
            C_host = prob[id,id]*CT
        a = tau*pair_num
        b = (tau+1)*pair_num
        All_C_matrix[tau] = 2*C_host # second order derivative
        All_C_matrix[tau,:4,:4] += 0.5*coeff*J[tau].T@J[tau]
        All_C_matrix[tau,4:,4:] += 0.5*coeff*np.eye(2)
        All_c_vector[tau] = 2*(trajectory[tau]-ref_traj[tau])@C_host
        All_c_vector[tau,:4] += coeff*J[tau].T@r[a:b]
        c = T*pair_num+id*T*2+tau*2
        All_c_vector[tau,4:] += coeff*r[c:c+2]
    
    return All_C_matrix, All_c_vector, coeff

@njit(cache=True)
def compute_l_J(n,all_trajectory,id,d_ref,beta,d_circle,graph,prob):

    T, N = len(all_trajectory[0]), len(all_trajectory)
    num_circles = len(d_circle)
    pair_num = int(N*(N-1)/2)*(num_circles**2)
    l = np.zeros(pair_num*T)
    J = np.zeros((T,pair_num,n))
    
    x_circle = np.zeros((num_circles,N,T,2))
    cos_all, sin_all = np.cos(all_trajectory[:, :, 2]), np.sin(all_trajectory[:, :, 2])
    for i in range(num_circles):
        x_circle[i,:,:,0] = all_trajectory[:,:,0]+d_circle[i]*cos_all
        x_circle[i,:,:,1] = all_trajectory[:,:,1]+d_circle[i]*sin_all
    dis = np.ones(num_circles**2)

    for tau in range(T):
        count = 0
        for i in range(N):
            # vec_i = x_circle[0,i,tau] - x_circle[1,i,tau]
            # norm_i = sqrt(vec_i[0]**2+vec_i[1]**2)+1e-6
            for j in range(i):
                # vec_j = x_circle[0,j,tau] - x_circle[1,j,tau]
                # norm_j = sqrt(vec_j[0]**2+vec_j[1]**2)+1e-6
                for k in range(num_circles**2):
                    circle_i, circle_j = k//num_circles, k%num_circles
                    dis[k] = sqrt((x_circle[circle_i,i,tau,0]-x_circle[circle_j,j,tau,0])**2
                                    +(x_circle[circle_i,i,tau,1]-x_circle[circle_j,j,tau,1])**2) - d_ref
                    # vec = x_circle[circle_i,i,tau]-x_circle[circle_j,j,tau]
                    # norm = sqrt(vec[0]**2+vec[1]**2)+1e-6
                    # extra_penalty = 0.1 + 0.45*abs(vec_i[0]*vec[0]+vec_i[1]*vec[1])/norm_i/norm\
                    #                     + 0.45*abs(vec_j[0]*vec[0]+vec_j[1]*vec[1])/norm_j/norm 
                    l[tau*pair_num+count+k] = sqrt(prob[id,j])*sqrt(beta)*dis[k]
                    if dis[k] < 0 and i == id and graph[i,j] > 0:
                        
                        J[tau,count+k,:2] = sqrt(prob[id,i])*sqrt(beta)/(dis[k]+d_ref+1e-10)*(x_circle[circle_i,i,tau]-x_circle[circle_j,j,tau])
                        J[tau,count+k,2] = -J[tau,count+k,0]*d_circle[circle_i]*sin_all[i,tau]+J[tau,count+k,1]*d_circle[circle_i]*cos_all[i,tau]

                    if dis[k] < 0 and j == id and graph[i,j] > 0:

                        J[tau,count+k,:2] = sqrt(prob[id,i])*sqrt(beta)/(dis[k]+d_ref+1e-10)*(x_circle[circle_j,j,tau]-x_circle[circle_i,i,tau])
                        J[tau,count+k,2] = -J[tau,count+k,0]*d_circle[circle_j]*sin_all[j,tau]+J[tau,count+k,1]*d_circle[circle_j]*cos_all[j,tau]
                        
                count += num_circles**2
    return l,J

@njit(cache=True)
def get_cost_static(all_trajectory,d_ref,ref_trajs,C_matrix,CT_matrix,with_collision,beta,d_circle,prob):
    host_cost = 0
    collision_cost = 0
    T = len(all_trajectory[0])
    N = len(all_trajectory)

    num_circles = len(d_circle)
    x_circle = np.zeros((num_circles,N,T,2))
    cos_all, sin_all = np.cos(all_trajectory[:, :, 2]), np.sin(all_trajectory[:, :, 2])
    for i in range(num_circles):
        x_circle[i,:,:,0] = all_trajectory[:,:,0]+d_circle[i]*cos_all
        x_circle[i,:,:,1] = all_trajectory[:,:,1]+d_circle[i]*sin_all

    for tau in range(T):
        if tau == T-1:
            C_matrix = CT_matrix
        for i in range(N):
            state = all_trajectory[i][tau]-ref_trajs[i][tau]
            host_cost += prob[i,i]*state@C_matrix@state

            if with_collision:
                for j in range(i):
                    # vec_j = x_circle[0,j,tau] - x_circle[1,j,tau]
                    # norm_j = sqrt(vec_j[0]**2+vec_j[1]**2)+1e-6
                    for k in range(num_circles**2):
                        circle_i, circle_j = k//num_circles, k%num_circles
                        collision_cost += prob[i,j]*beta*(min(sqrt((x_circle[circle_i,i,tau,0]-x_circle[circle_j,j,tau,0])**2
                                                            +(x_circle[circle_i,i,tau,1]-x_circle[circle_j,j,tau,1])**2) - d_ref, 0))**2
                        
    return host_cost + collision_cost


@njit(cache=True)
def forward_pass_static(m,n,T,K_matrix,k_vector,F_matrix,init_state):
    trajectory = np.zeros((T,m+n))
    trajectory[0] = init_state
    for tau in range(T-1):
        x = trajectory[tau,:n]
        u = K_matrix[tau]@x+k_vector[tau]
        x_u = np.concatenate((x,u))
        trajectory[tau,n:] = u
        trajectory[tau+1, :n] = F_matrix[tau,]@x_u
    return trajectory

@njit(cache=True)
def backward_pass_static(m,n,T,C_matrix,c_vector,F_matrix):
    V_matrix = np.zeros((n, n))
    v_vector = np.zeros(n)
    K_matrix_list = np.zeros((T, m, n))
    k_vector_list = np.zeros((T, m))
    for i in range(T-1,-1,-1):
        Q_matrix = C_matrix[i] + F_matrix[i].T@V_matrix@F_matrix[i]
        q_vector = c_vector[i] + F_matrix[i].T@v_vector
        Q_uu = Q_matrix[n:n+m,n:n+m].copy()
        Q_ux = Q_matrix[n:n+m,0:n].copy()
        q_u = q_vector[n:n+m].copy()

        Q_uu_inv = np.linalg.inv(Q_uu)
        K_matrix_list[i] = -Q_uu_inv@Q_ux
        k_vector_list[i] = -Q_uu_inv@q_u

        V_matrix = Q_matrix[:n,:n] - Q_ux.T@Q_uu_inv@Q_ux
        v_vector = q_vector[:n] - Q_ux.T@Q_uu_inv@q_u

    return K_matrix_list, k_vector_list

@njit(cache=True)
def eval_traj_static(init_state, input_traj, m, n):
    T = int(input_traj.shape[0])
    trajectory = np.zeros((T,m+n))
    trajectory[0] = np.concatenate((init_state,input_traj[0]))
    for tau in range(T-1):
        trajectory[tau+1, :n] = np.asarray(dyn(trajectory[tau,:]), dtype=np.float64)
        trajectory[tau+1, n:] = input_traj[tau+1]
    return trajectory

@njit(cache=True)
def update_traj_static(m, n, old_traj, K_matrix_all, k_vector_all, lb, ub, alpha):
    T = int(K_matrix_all.shape[0])
    new_trajectory = np.zeros((T,m+n))
    new_trajectory[0] = old_traj[0]
    for tau in range(T-1):
        delta_x = new_trajectory[tau, :n] - old_traj[tau, :n]
        delta_u = K_matrix_all[tau]@delta_x+alpha*k_vector_all[tau]
        input_u = old_traj[tau, n:] + delta_u
        new_trajectory[tau, n:] = np.clip(input_u,lb,ub)
        new_trajectory[tau+1, :n] = np.asarray(dyn(new_trajectory[tau,:]),dtype=np.float64)
    return new_trajectory

@njit(cache=True)
def eval_grad_static(trajectory):
    T = int(trajectory.shape[0])
    F_matrix_initial = grad_dyn(trajectory[0,:])
    F_matrix_all = np.zeros((T, len(F_matrix_initial), len(F_matrix_initial[0])))
    F_matrix_all[0] = np.asarray(F_matrix_initial, dtype = np.float64)
    for tau in range(1, T):
        F_matrix_all[tau] = np.asarray(grad_dyn(trajectory[tau,:]), dtype = np.float64)
    return F_matrix_all

@njit(cache=True)
def dyn(_Dummy_22):
    # 2.7 wheel base
    [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_22
    return (  # Not supported in Python:
  # ImmutableDenseNDimArray
[x_u_0_0 + (0.1*x_u_0_3*cos(x_u_0_4) - 2.7*sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 2.7)*cos(x_u_0_2), x_u_0_1 + (0.1*x_u_0_3*cos(x_u_0_4) - 2.7*sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 2.7)*sin(x_u_0_2), x_u_0_2 + asin(0.037037037037037*x_u_0_3*sin(x_u_0_4)), x_u_0_3 + 0.1*x_u_0_5])


@njit(cache=True)
def grad_dyn(_Dummy_23):
    # 2.7 wheel base
    [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_23
    return (  # Not supported in Python:
  # ImmutableDenseNDimArray
[[1, 0, -(0.1*x_u_0_3*cos(x_u_0_4) - 2.7*sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 2.7)*sin(x_u_0_2), (0.0037037037037037*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 0.1*cos(x_u_0_4))*cos(x_u_0_2), (0.0037037037037037*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) - 0.1*x_u_0_3*sin(x_u_0_4))*cos(x_u_0_2), 0], [0, 1, (0.1*x_u_0_3*cos(x_u_0_4) - 2.7*sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 2.7)*cos(x_u_0_2), (0.0037037037037037*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 0.1*cos(x_u_0_4))*sin(x_u_0_2), (0.0037037037037037*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) - 0.1*x_u_0_3*sin(x_u_0_4))*sin(x_u_0_2), 0], [0, 0, 1, 0.037037037037037*sin(x_u_0_4)/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 0.037037037037037*x_u_0_3*cos(x_u_0_4)/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 0], [0, 0, 0, 1, 0, 0.1]])
