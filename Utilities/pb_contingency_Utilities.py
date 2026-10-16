from numba import njit
import numpy as np
from math import sin, cos, sqrt
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
def neighbor_index(graph):
    neighbors_sum_row = np.zeros_like(graph)
    for i in range(len(graph[0])):
        count = 0
        for j in range(len(graph[0])):
            count += graph[i,j]
            neighbors_sum_row[i,j] = count
    return neighbors_sum_row

@njit(cache=True)
def compute_psr(all_y,p,s,y,z,rho,sigma,id,graph,segment_length):
    
    neighbors = np.sum(graph,axis=1)
    neighbors_sum = np.cumsum(neighbors)
    neighbors_sum_row = neighbor_index(graph)
    
    pair_num = neighbors[id] * segment_length
    income_y_array = np.zeros(pair_num)

    # Each time step contains four pairings of the two vehicle circles.
    T = segment_length // 4
    for tau in range(T):
        for i in range(len(graph[0])):
            if i == id:
                continue
            
            if graph[i,id] == 0:
                continue

            if i == 0:
                start = tau * neighbors[i] * 4\
                      + (neighbors_sum_row[i,id]-1) * 4
            else:
                start = (neighbors_sum[i-1] * segment_length) + tau * neighbors[i] * 4\
                      + (neighbors_sum_row[i,id]-1) * 4

            end = start + 4
            income_y = all_y[start:end]
            start = tau * neighbors[id] * 4 + (neighbors_sum_row[id,i]-1) * 4
            end = start + 4
            income_y_array[start:end] = income_y
    

    p[:pair_num] = p[:pair_num] + 0.5*rho[:pair_num]*(y[:pair_num]-income_y_array)

    s = s + sigma*(y-z)

    r = sigma * z - p - s
    r[pair_num:] = r[pair_num:] + rho[pair_num:]*y[pair_num:]
    r[:pair_num] = r[:pair_num] + 0.5*rho[:pair_num]*(y[:pair_num]+income_y_array)

    return p,s,r

@njit(cache=True)
def compute_z(s,sigma,y,l,lb,ub,pair_num):
    new_z = np.zeros_like(y)
    division = pair_num 
    new_z[:division] = 2.0/(2.0*sigma[:division]+1)*(s[:division]+sigma[:division]*y[:division]+l)
    new_z[division:] = np.maximum(s[division:]+sigma[division:]*y[division:],lb)
    new_z[division:] = np.minimum(new_z[division:],ub)
    new_z[division:] = s[division:]/sigma[division:]+y[division:]-new_z[division:]/sigma[division:]
    return new_z

@njit(cache=True)
def compute_y(delta_trajectory,J,r,coeff,id,graph,segment_length):

    neighbors = np.sum(graph,axis=1)
    
    T = len(delta_trajectory)
    new_y = coeff*r

    for tau in range(T):
        a = tau * neighbors[id] * 4
        b = (tau+1) * neighbors[id] * 4
        new_y[a:b] += coeff[a:b]*(J[a:b]@delta_trajectory[tau,:4])

    end = neighbors[id] * segment_length

    for tau in range(T):
        c = end+tau*2
        new_y[c:c+2] += coeff[c:c+2]*delta_trajectory[tau,4:]

    return new_y

@njit(cache=True)
def compute_C(trajectory,ref_traj,J,r,coeff,id,graph,segment_length,prob,C):

    neighbors = np.sum(graph,axis=1)

    T, n_m = len(trajectory), len(trajectory[0])
    All_C_matrix = np.zeros((T, n_m, n_m))
    All_c_vector = np.zeros((T, n_m))
    C_host = prob[id,id]*C # host cost is sth like (x_u-r).T@C@(x_u-r)

    end = neighbors[id] * segment_length

    for tau in range(T):
        c = end+tau*2
        All_C_matrix[tau] = 2*C_host # second order derivative
        All_C_matrix[tau,4:,4:] += np.eye(2)@np.diag(coeff[c:c+2])
        All_c_vector[tau] = 2*(trajectory[tau]-ref_traj[tau])@C_host
        All_c_vector[tau,4:] += coeff[c:c+2]*r[c:c+2]
        a = tau * neighbors[id] * 4
        b = (tau+1) * neighbors[id] * 4
        tmp = J[a:b].T@np.diag(coeff[a:b])
        All_C_matrix[tau,:4,:4] += tmp@J[a:b]
        All_c_vector[tau,:4] += tmp@r[a:b]
            
    return All_C_matrix, All_c_vector, coeff

@njit(cache=True)
def compute_l_J(n,all_trajectory,id,d_ref,beta,d_circle,graph,segment_length,prob,contin_type):

    neighbors = np.sum(graph,axis=1)
    neighbors_sum_row = neighbor_index(graph)

    T, N = len(all_trajectory[0]), len(all_trajectory)
    num_circles = len(d_circle)
    pair_num = neighbors[id] * segment_length
    l = np.zeros(pair_num)
    J = np.zeros((pair_num,n))
    
    x_circle = np.zeros((num_circles,N,T,2))
    cos_all, sin_all = np.cos(all_trajectory[:, :, 2]), np.sin(all_trajectory[:, :, 2])
    for i in range(num_circles):
        x_circle[i,:,:,0] = all_trajectory[:,:,0]+d_circle[i]*cos_all
        x_circle[i,:,:,1] = all_trajectory[:,:,1]+d_circle[i]*sin_all
    dis = np.ones(num_circles**2)

    coeff = [50,50,100,10]
    for k in range(4):
        coeff[k] = sqrt(coeff[k])

    for tau in range(T):
        for i in range(len(graph[0])):
            if i == id:
                continue
            
            if graph[id,i] == 0:
                continue
            
            if id in contin_type and i in contin_type:
                
                if tau < 5:
                    
                    for k in range(4):
                        index = tau * neighbors[id]*4 + (neighbors_sum_row[id,i] - 1)*4 + k
                        if id > i:
                            l[index]   = coeff[k]*(all_trajectory[id,tau,k]-all_trajectory[i,tau,k])
                            J[index,k] = coeff[k]
                        else:
                            l[index]   = coeff[k]*(-all_trajectory[id,tau,k]+all_trajectory[i,tau,k])
                            J[index,k] = -coeff[k]
                continue

            for k in range(num_circles**2):
                if id > i:
                    circle_id, circle_i = k//num_circles, k%num_circles
                else:
                    circle_i, circle_id = k//num_circles, k%num_circles
                dis[k] = sqrt((x_circle[circle_id,id,tau,0]-x_circle[circle_i,i,tau,0])**2
                        +(x_circle[circle_id,id,tau,1]-x_circle[circle_i,i,tau,1])**2) - d_ref
                index = tau * neighbors[id]*4 + (neighbors_sum_row[id,i] - 1)*4 + k
                
                if dis[k] < 0:
                    l[index] = sqrt(prob[id,i])*sqrt(beta)*dis[k]
                    J[index,:2] = sqrt(prob[id,i])*sqrt(beta)/(dis[k]+d_ref+1e-10)*(x_circle[circle_id,id,tau]-x_circle[circle_i,i,tau])
                    J[index,2] = -J[index,0]*d_circle[circle_id]*sin_all[id,tau]+J[index,1]*d_circle[circle_id]*cos_all[id,tau]

    return l,J

@njit(cache=True)
def get_cost_static(all_trajectory,d_ref,ref_trajs,C_matrix,with_collision,beta,d_circle,prob,contin_type):
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
            
        for i in range(N):
            state = all_trajectory[i][tau]-ref_trajs[i][tau]
            host_cost += prob[i,i]*state@C_matrix@state

            for j in range(i):
                for k in range(num_circles**2):
                    circle_i, circle_j = k//num_circles, k%num_circles

                    collision_cost += prob[i,j]*beta*(min(sqrt((x_circle[circle_i,i,tau,0]-x_circle[circle_j,j,tau,0])**2
                                                        +(x_circle[circle_i,i,tau,1]-x_circle[circle_j,j,tau,1])**2) - d_ref, 0))**2
                coeff = [50,50,100,10]
                if i in contin_type and j in contin_type and tau < 5:
                    for k in range(4):
                        collision_cost += coeff[k]*(all_trajectory[i,tau,k]-all_trajectory[j,tau,k])**2
            
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
        Q_uu = Q_matrix[n:n+m,n:n+m]
        Q_ux = Q_matrix[n:n+m,0:n]
        q_u = q_vector[n:n+m]

        Q_uu_inv = np.linalg.inv(Q_uu)
        K_matrix_list[i] = -Q_uu_inv@Q_ux
        k_vector_list[i] = -Q_uu_inv@q_u

        tmp = Q_ux.T@Q_uu_inv
        V_matrix = Q_matrix[:n,:n] - tmp@Q_ux
        v_vector = q_vector[:n] - tmp@q_u
        
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

# @njit(cache=True)
# def dyn(_Dummy_48):
#     # 0.1 wheel base
#     [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_48
#     return np.array([x_u_0_0 + (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.01) + 0.1)*cos(x_u_0_2), x_u_0_1 + (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.01) + 0.1)*sin(x_u_0_2), x_u_0_2 + np.arcsin(1.0*x_u_0_3*sin(x_u_0_4)), x_u_0_3 + 0.1*x_u_0_5])

# @njit(cache=True)
# def dyn(_Dummy_42):
#     # 0.25
#     [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_42
#     return np.array([x_u_0_0 + (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.0625) + 0.25)*cos(x_u_0_2), x_u_0_1 + (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.0625) + 0.25)*sin(x_u_0_2), x_u_0_2 + np.arcsin(0.4*x_u_0_3*sin(x_u_0_4)), x_u_0_3 + 0.1*x_u_0_5])

@njit(cache=True)
def dyn(_Dummy_34):
    # 0.5 wheel base
    [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_34
    return np.array([x_u_0_0 + (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.25) + 0.5)*cos(x_u_0_2), x_u_0_1 + (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.25) + 0.5)*sin(x_u_0_2), x_u_0_2 + np.arcsin(0.2*x_u_0_3*sin(x_u_0_4)), x_u_0_3 + 0.1*x_u_0_5])

# @njit(cache=True)
# def dyn(_Dummy_34):
#     [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_34
#     return np.array([x_u_0_0 + (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 1.0) + 1.0)*cos(x_u_0_2), x_u_0_1 + (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 1.0) + 1.0)*sin(x_u_0_2), x_u_0_2 + np.arcsin(0.1*x_u_0_3*sin(x_u_0_4)), x_u_0_3 + 0.1*x_u_0_5])

# @njit(cache=True)
# def dyn(_Dummy_22):
#     # 2.7 wheel base
#     [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_22
#     return (  # Not supported in Python:
#   # ImmutableDenseNDimArray
# [x_u_0_0 + (0.1*x_u_0_3*cos(x_u_0_4) - 2.7*sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 2.7)*cos(x_u_0_2), x_u_0_1 + (0.1*x_u_0_3*cos(x_u_0_4) - 2.7*sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 2.7)*sin(x_u_0_2), x_u_0_2 + asin(0.037037037037037*x_u_0_3*sin(x_u_0_4)), x_u_0_3 + 0.1*x_u_0_5])

# @njit(cache=True)
# def grad_dyn(_Dummy_49):
#     # 0.1 wheel base
#     [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_49
#     return np.array([[1, 0, -(0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.01) + 0.1)*sin(x_u_0_2), (0.01*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.01) + 0.1*cos(x_u_0_4))*cos(x_u_0_2), (0.01*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.01) - 0.1*x_u_0_3*sin(x_u_0_4))*cos(x_u_0_2), 0], [0, 1, (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.01) + 0.1)*cos(x_u_0_2), (0.01*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.01) + 0.1*cos(x_u_0_4))*sin(x_u_0_2), (0.01*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.01) - 0.1*x_u_0_3*sin(x_u_0_4))*sin(x_u_0_2), 0], [0, 0, 1, 1.0*sin(x_u_0_4)/sqrt(-1.0*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 1.0*x_u_0_3*cos(x_u_0_4)/sqrt(-1.0*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 0], [0, 0, 0, 1, 0, 0.1]])

# @njit(cache=True)
# def grad_dyn(_Dummy_43):
#     #0.25
#     [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_43
#     return np.array([[1, 0, -(0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.0625) + 0.25)*sin(x_u_0_2), (0.01*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.0625) + 0.1*cos(x_u_0_4))*cos(x_u_0_2), (0.01*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.0625) - 0.1*x_u_0_3*sin(x_u_0_4))*cos(x_u_0_2), 0], [0, 1, (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.0625) + 0.25)*cos(x_u_0_2), (0.01*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.0625) + 0.1*cos(x_u_0_4))*sin(x_u_0_2), (0.01*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.0625) - 0.1*x_u_0_3*sin(x_u_0_4))*sin(x_u_0_2), 0], [0, 0, 1, 0.4*sin(x_u_0_4)/sqrt(-0.16*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 0.4*x_u_0_3*cos(x_u_0_4)/sqrt(-0.16*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 0], [0, 0, 0, 1, 0, 0.1]])

@njit(cache=True)
def grad_dyn(_Dummy_49):
    # 0.5 wheel base
    [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_49
    return np.array([[1, 0, -(0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.25) + 0.5)*sin(x_u_0_2), (0.01*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.25) + 0.1*cos(x_u_0_4))*cos(x_u_0_2), (0.01*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.25) - 0.1*x_u_0_3*sin(x_u_0_4))*cos(x_u_0_2), 0], [0, 1, (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.25) + 0.5)*cos(x_u_0_2), (0.01*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.25) + 0.1*cos(x_u_0_4))*sin(x_u_0_2), (0.01*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 0.25) - 0.1*x_u_0_3*sin(x_u_0_4))*sin(x_u_0_2), 0], [0, 0, 1, 0.2*sin(x_u_0_4)/sqrt(-0.04*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 0.2*x_u_0_3*cos(x_u_0_4)/sqrt(-0.04*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 0], [0, 0, 0, 1, 0, 0.1]])

# @njit(cache=True)
# def grad_dyn(_Dummy_35):
#     [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_35
#     return np.array([[1, 0, -(0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 1.0) + 1.0)*sin(x_u_0_2), (0.01*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 1.0) + 0.1*cos(x_u_0_4))*cos(x_u_0_2), (0.01*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 1.0) - 0.1*x_u_0_3*sin(x_u_0_4))*cos(x_u_0_2), 0], [0, 1, (0.1*x_u_0_3*cos(x_u_0_4) - sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 1.0) + 1.0)*cos(x_u_0_2), (0.01*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 1.0) + 0.1*cos(x_u_0_4))*sin(x_u_0_2), (0.01*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 1.0) - 0.1*x_u_0_3*sin(x_u_0_4))*sin(x_u_0_2), 0], [0, 0, 1, 0.1*sin(x_u_0_4)/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 0.1*x_u_0_3*cos(x_u_0_4)/sqrt(-0.01*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 0], [0, 0, 0, 1, 0, 0.1]])

# @njit(cache=True)
# def grad_dyn(_Dummy_23):
#     # 2.7 wheel base
#     [x_u_0_0, x_u_0_1, x_u_0_2, x_u_0_3, x_u_0_4, x_u_0_5] = _Dummy_23
#     return (  # Not supported in Python:
#   # ImmutableDenseNDimArray
# [[1, 0, -(0.1*x_u_0_3*cos(x_u_0_4) - 2.7*sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 2.7)*sin(x_u_0_2), (0.0037037037037037*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 0.1*cos(x_u_0_4))*cos(x_u_0_2), (0.0037037037037037*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) - 0.1*x_u_0_3*sin(x_u_0_4))*cos(x_u_0_2), 0], [0, 1, (0.1*x_u_0_3*cos(x_u_0_4) - 2.7*sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 2.7)*cos(x_u_0_2), (0.0037037037037037*x_u_0_3*sin(x_u_0_4)**2/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) + 0.1*cos(x_u_0_4))*sin(x_u_0_2), (0.0037037037037037*x_u_0_3**2*sin(x_u_0_4)*cos(x_u_0_4)/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1) - 0.1*x_u_0_3*sin(x_u_0_4))*sin(x_u_0_2), 0], [0, 0, 1, 0.037037037037037*sin(x_u_0_4)/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 0.037037037037037*x_u_0_3*cos(x_u_0_4)/sqrt(-0.00137174211248285*x_u_0_3**2*sin(x_u_0_4)**2 + 1), 0], [0, 0, 0, 1, 0, 0.1]])
