import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np

class normal:
    def __init__(self,mean,std,coeff):
        self.mean = mean
        self.std  = std
        self.coeff = coeff
    
    def value(self,x):
        coeff = self.coeff*1./np.sqrt(2*np.pi*self.std**2)
        return coeff * np.exp(-(x-self.mean)**2/(2*self.std**2))

class multi_normal:
    
    def __init__(self,distribution):
        self.normals = []
        for tmp in distribution:
            self.normals.append(normal(tmp[0],tmp[1],tmp[2]))
    
    def value(self,x):
        prob = 0
        for n in self.normals:
            prob += n.value(x)
        return prob
    
def cross_2(start_points,distribution,T,long_ref=False):
    ref_v_list_list = []
    N_type = 0
    player_list = []
    player = 0
    for i,dist in zip(range(len(distribution)),distribution):
        ref_v_list = []
        count = 0
        for tmp in dist:
            mean,std,coeff = tmp[0],tmp[1],tmp[2]
            if i == 0:
                list = range(0,1)
            else:
                list = range(-2,3)
            for j in list:
                ref_v_list.append(mean+j*std)
                N_type += 1
                count+= 1
                player_list.append(player)
        player += 1
        ref_v_list_list.append(ref_v_list)
    
    N = len(ref_v_list_list)
    if long_ref:
        length = 3*T
    else:
        length = T
    all_reference_pos_array = np.zeros((N_type,length ,2))
    count = 0
    for i in range(N):
        for v in ref_v_list_list[i]:
            if i == 0:
                x = 3*np.ones(length)
                y = start_points[i,1]+np.arange(length)*0.1*v
                all_reference_pos_array[count] = np.stack([x,y],axis=-1)
            if i == 1:
                y = -3*np.ones(length)
                x = start_points[i,0]+np.arange(length)*0.1*v
                all_reference_pos_array[count] = np.stack([x,y],axis=-1)
            if i == 2:
                y = 3*np.ones(length)
                x = start_points[i,0]-np.arange(length)*0.1*v
                all_reference_pos_array[count] = np.stack([x,y],axis=-1)
            count += 1

    tmp = np.zeros((N_type,length,4))
    ref_trajs = np.concatenate([all_reference_pos_array,tmp],axis=-1)

    graph = np.zeros((N_type,N_type),dtype=np.int64)
    prob_matrix = np.zeros((N_type,N_type))

    multi_n_list = []
    for i in range(N):
        multi_n_list.append(multi_normal(distribution[i]))
    
    prob_list = []
    for i in range(N):
        ref_v_list = ref_v_list_list[i]
        prob = np.zeros(len(ref_v_list))
        for (j,v) in zip(range(len(ref_v_list)),ref_v_list):
            prob[j] = multi_n_list[i].value(v)
        prob = prob/np.sum(prob)
        prob_list += prob.tolist()

    for i in range(N_type):
        for j in range(N_type):

            if i == j:
                prob_matrix[i,i] = prob_list[i]
                continue

            if player_list[i] == player_list[j]:
                continue

            graph[i,j] = 1
            prob_matrix[i,j] = prob_list[i] * prob_list[j]

    new_start_points = np.zeros((N_type,4))
    for i in range(N_type):
        new_start_points[i] = start_points[player_list[i]]

    def add_background(ax):
        
        w = 8
        r = 30
        ax.plot([-r,-w],[w,w],color="dimgray",linestyle='-',linewidth=2)
        ax.plot([w,r],[w,w],color="dimgray",linestyle='-',linewidth=2)
        ax.plot([-r,-w],[-w,-w],color="dimgray",linestyle='-',linewidth=2) 
        ax.plot([w,r],[-w,-w],color="dimgray",linestyle='-',linewidth=2)
        
        
        ax.plot([w,w],[-r,-w],color="dimgray",linestyle='-',linewidth=2)
        ax.plot([w,w],[w,r],color="dimgray",linestyle='-',linewidth=2)
        ax.plot([-w,-w],[-r,-w],color="dimgray",linestyle='-',linewidth=2) 
        ax.plot([-w,-w],[w,r],color="dimgray",linestyle='-',linewidth=2)

        ax.plot([0,0],[-r,-w],color="dimgray",linestyle='-',linewidth=1)
        ax.plot([0,0],[w,r],color="dimgray",linestyle='-',linewidth=1)
        ax.plot([-r,-w],[0,0],color="dimgray",linestyle='-',linewidth=1)
        ax.plot([w,r],[0,0],color="dimgray",linestyle='-',linewidth=1)
        ax.set_xlim([-20,24])
        ax.set_ylim([-20,20])
    
    return new_start_points,ref_trajs,graph, prob_matrix,player_list,add_background,[9.6,8.0]

def cross(start_points,distribution,T):
    
    ref_v_list = []
    player_list = []
    for tmp in distribution:
        mean,std,coeff = tmp[0],tmp[1],tmp[2]
        for j in range(-1,2):
            ref_v_list.append(mean+j*std)
            
    N = len(ref_v_list) + 1
    all_reference_pos_array = np.zeros((N,T,2))
    
    for i in range(N):
        if i == 0:
            x = start_points[0,0]*np.ones(T)
            y = start_points[0,1]+np.arange(T)*0.1*start_points[0,3]
            all_reference_pos_array[i] = np.stack([x,y],axis=-1)
            player_list.append(0)
            continue

        player_list.append(1)
        y = start_points[1,1]*np.ones(T)
        ref_v = ref_v_list[i-1]
        x = start_points[1,0]+np.arange(T)*0.1*ref_v
        all_reference_pos_array[i] = np.stack([x,y],axis=-1)
    
    tmp = np.zeros((N,T,4))
    ref_trajs = np.concatenate([all_reference_pos_array,tmp],axis=-1)
    graph = np.zeros((N,N),dtype=np.int64)
    prob = np.zeros((N,N))
    for i in range(N):
        if i == 0:
            continue
        graph[0,i] = 1
        graph[i,0] = 1
    
    multi_n = multi_normal(distribution)
    prob = np.zeros(N-1)
    for i in range(N-1):
        prob[i] = multi_n.value(ref_v_list[i])
    
    prob = prob/np.sum(prob)
    prob_matrix = np.zeros((N,N))
    for i in range(N):
        if i == 0:
            prob_matrix[i,i] = 1
            continue
        prob_matrix[i,i] = prob[i-1]
        prob_matrix[i,0] = prob[i-1]
        prob_matrix[0,i] = prob[i-1]
        
    new_start_points = np.zeros((N,4))
    for i in range(N):
        if i == 0:
            new_start_points[i] = start_points[i]
            continue
        new_start_points[i] = start_points[-1]
    
    plot_distribution(distribution)

    def add_background(ax):
        
        w = 4.5
        ax.plot([-15,-w],[w,w],'k-',linewidth=1)
        ax.plot([w,15],[w,w],'k-',linewidth=1)
        ax.plot([-15,-w],[-w,-w],'k-',linewidth=1) 
        ax.plot([w,15],[-w,-w],'k-',linewidth=1)
        
        
        ax.plot([w,w],[-15,-w],'k-',linewidth=1)
        ax.plot([w,w],[w,15],'k-',linewidth=1)
        ax.plot([-w,-w],[-15,-w],'k-',linewidth=1) 
        ax.plot([-w,-w],[w,15],'k-',linewidth=1)
        ax.set_xlim([-15,15])
        ax.set_ylim([-15,15])

    return new_start_points,ref_trajs,graph,prob_matrix,player_list,add_background,[6.4,6.4]

def merge(start_points,distribution,T):
    
    ref_v_list = []
    player_list = []
    for tmp in distribution:
        mean,std,coeff = tmp[0],tmp[1],tmp[2]
        for j in range(-2,3):
            ref_v_list.append(mean+j*std)
            
    N = len(ref_v_list) + 1
    all_reference_pos_array = np.zeros((N,T,2))
    
    for i in range(N):
        if i == 0:
            y = 0*np.ones(T)
            x = start_points[0,0]+np.arange(T)*0.1*3
            all_reference_pos_array[i] = np.stack([x,y],axis=-1)
            player_list.append(0)
            continue
        
        player_list.append(1)
        y = 0*np.ones(T)
        ref_v = ref_v_list[i-1]
        x = start_points[1,0]+np.arange(T)*0.1*ref_v
        all_reference_pos_array[i] = np.stack([x,y],axis=-1)
    
    tmp = np.zeros((N,T,4))
    ref_trajs = np.concatenate([all_reference_pos_array,tmp],axis=-1)

    for i in range(N):
        if i == 0:
            ref_trajs[0,:,3] = 3
            continue
        ref_v = ref_v_list[i-1]
        ref_trajs[i,:,3] = ref_v

    graph = np.zeros((N,N),dtype=np.int64)
    prob = np.zeros((N,N))
    for i in range(N):
        if i == 0:
            continue
        graph[0,i] = 1
        graph[i,0] = 1
    
    multi_n = multi_normal(distribution)
    prob = np.zeros(N-1)
    for i in range(N-1):
        prob[i] = multi_n.value(ref_v_list[i])
    
    prob = prob/np.sum(prob)
    prob_matrix = np.zeros((N,N))
    for i in range(N):
        if i == 0:
            prob_matrix[i,i] = 1
            continue
        prob_matrix[i,i] = prob[i-1]
        prob_matrix[i,0] = prob[i-1]
        prob_matrix[0,i] = prob[i-1]
        
    new_start_points = np.zeros((N,4))
    for i in range(N):
        if i == 0:
            new_start_points[i] = start_points[i]
            continue
        new_start_points[i] = start_points[-1]
    
    # plot_distribution(distribution)

    def draw(ax):
        ax.plot([-5,50],[-3,-3],color="dimgray",linestyle='-',linewidth=2)
        ax.plot([-5,50],[2,2],color="dimgray",linestyle='--',linewidth=1)
        ax.plot([-5,50],[7,7],color="dimgray",linestyle='-',linewidth=2)

    return new_start_points,ref_trajs,graph,prob_matrix,player_list,draw,[6.4,1.8]

def contingency(start_points,distribution,lane,ref_speed,T):
    N = len(lane) * len(ref_speed)
    new_start_points = np.zeros((N*2,4))
    for i in range(N):
        new_start_points[i] = start_points[0]
        new_start_points[N+i] = start_points[1]
    
    graph = np.zeros((N*2,N*2),dtype=np.int64)
    prob  = np.zeros((N*2,N*2))
    for i in range(N*2):
        if i < N:
            for j in range(N):
                if j == i:
                    continue
                graph[i,j] = 1
            graph[i,N+i] = 1
            prob[i,i] = distribution[i]
            prob[i,N+i] = distribution[i]
        else:
            graph[i,i-N] = 1
            prob[i,i] = distribution[i-N]
            prob[i,i-N] = distribution[i-N]

    ref_trajs = np.zeros((N*2,T,6))
    for i in range(N*2):
        ref_trajs[i,:,2] = 1.57
        if i < N:
            ref_trajs[i,:,0] = lane[i//len(ref_speed)]
            # ref_trajs[i,:,0] = lane[0]
            ref_trajs[i,:,3] = start_points[0,3]
        else:
            tmp = 0 if (i-N)//len(ref_speed) == 1 else 1
            ref_trajs[i,:,0] = lane[tmp]
            index = ((i-N)%len(ref_speed))
            ref_trajs[i,:,3] = ref_speed[index]

    contin_type = [i for i in range(N)]

    return new_start_points,ref_trajs,graph,prob,contin_type, N*2


def plot_distribution(distribution):
    multi_n = multi_normal(distribution)
    v_list = [i*0.01 for i in range(200,400)]
    p_list = []
    for v in v_list:
        p = multi_n.value(v)
        p_list.append(p)
    
    fig1 = plt.figure(figsize=[6.4,1.8])
    ax1 = plt.axes()
    ax1.set_xlim([2,4])
    ax1.set_ylim([0,2])
    ax1.set_xlabel('v(m/s)')
    ax1.set_ylabel('p')
    ax1.plot(v_list,p_list)
    # ax1.axis('equal')
    fig1.savefig("tmp.jpg",bbox_inches='tight',dpi=600)

def plot(trajectory,prob,player_list,add_background,fig_size,save=True):

    alpha = np.zeros_like(prob)
    for i in range(np.max(player_list)+1):

        sum = []
        for (j,p) in zip(range(len(player_list)),prob):
            if player_list[j] == i:
                sum.append(p)

        a, b = np.max(sum), np.min(sum)
        for (j,p) in zip(range(len(player_list)),prob):
            if player_list[j] == i:
                alpha[j] = (p-b+1e-5)/(a-b+1e-5)*0.8+0.2
            
    N = len(trajectory)
    T = len(trajectory[0])
    colorVeh = np.array([[246, 185, 59],[31, 119, 180],[128,138,135]])/255.0
    # colorVeh = np.array([[31, 119, 180],[246, 185, 59]])/255.0
    # colorVeh = np.array([[52, 152, 219],[243, 156, 18],[46, 204, 113]])/255.0


    plt.tight_layout()
    fig1 = plt.figure(figsize=fig_size)
    ax1 = plt.axes()
    ax1.axis('equal')
    fig2 = plt.figure(figsize=fig_size)
    ax2 = plt.axes()
    ax2.axis('equal')

    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2.xaxis.set_tick_params(labelsize=14)
    ax2.yaxis.set_tick_params(labelsize=14)

    plt.tight_layout()

    def obtain_rectangles(tau, i):
        yaw = trajectory[i,tau,2]
        center = np.array([trajectory[i,tau,0]+1.35*np.cos(yaw), trajectory[i,tau,1]+1.35*np.sin(yaw)])
        length = 4.15
        width = 1.77
        corner_vector = np.array([-length/2.0,-width/2.0])
        index = player_list[i]
        rectangle = Rectangle(center+corner_vector,length,width,
                                        facecolor=colorVeh[index,:],
                                        angle=yaw * 57.3,
                                        rotation_point='center',
                                        alpha=alpha[i],
                                        linewidth = 1,
                                        edgecolor=[0,0,0])
        return rectangle
    
    def polygon(tau,i):
        yaw = trajectory[i,tau,2]
        a = np.array([
        [3.425, 0],
        [1.4,-0.885],
        [-0.725,-0.885],
        [-0.725,0.885],
        [1.4,0.885]
        ])
        tmp = np.zeros((5,2))
        index = player_list[i]
        for j in range(5):
            tmp[j][0] = trajectory[i,tau,0] + np.cos(yaw) * a[j][0] - np.sin(yaw) * a[j][1]
            tmp[j][1] = trajectory[i,tau,1] + np.sin(yaw) * a[j][0] + np.cos(yaw) * a[j][1]
        irregular = patches.Polygon(tmp, closed=True, facecolor=colorVeh[index,:],\
                                    # alpha=alpha[i],
                                    # alpha = 1- tau/100.0,
                                    linewidth = 1,edgecolor=[0,0,0])
        return irregular

    def init():
        ax2.clear()
        add_background(ax2)

    def update(k):
        ax2.clear()
        add_background(ax2)
        for i in range(N):
            index = player_list[i]
            ax2.plot(trajectory[i,:k,0], trajectory[i,:k,1], color=colorVeh[index,:],alpha=alpha[i],zorder=0)
        for i in range(N):
            # rectangle = obtain_rectangles(k,i)
            poly = polygon(k,i)
            ax2.add_patch(poly)
        # if k in [25,50,75,99,33,66]:
        #     fig2.savefig("results/result%d.jpg"%(k),bbox_inches='tight',dpi=600)
    
    add_background(ax1)
    ax1.axis('off')
    for j in range(N):
        index = player_list[j]
        ax1.plot(trajectory[j,:,0], trajectory[j,:,1], color=colorVeh[index,:], linewidth=1.5, alpha=alpha[j])
        

        # Show one vehicle at the final state of each trajectory.
        poly = polygon(T - 1, j)
        ax1.add_patch(poly)
    
    fig1.savefig("result.jpg",bbox_inches='tight',dpi=600)


    # for k in range(T):
    #     if k in [25,50,75,99,33,66]:
    #         ax2.clear()
    #         add_background(ax2)
    #         for i in range(N):
    #             index = player_list[i]
    #             ax2.plot(trajectory[i,:k,0], trajectory[i,:k,1], color=colorVeh[index,:],alpha=alpha[i],zorder=0)
    #         for i in range(N):
    #             # rectangle = obtain_rectangles(k,i)
    #             poly = polygon(k,i)
    #             ax2.add_patch(poly)
            # fig2.savefig("results/result%d.jpg"%(k),bbox_inches='tight',dpi=600)

    if save:
        ani = FuncAnimation(fig2, update, frames = T, init_func = init, interval = 100, repeat = True)
        ani.save("result.mp4", fps=10, dpi=600)
        
def plot_contingency(trajectory, save=True):
    """Plot contingency trajectories, grouped by vehicle.

    Ego branches come first, followed by the other vehicle's branches, as
    returned by contingency(). State-only and state/control trajectories
    are both supported; each state must contain at least x, y and yaw.
    """
    trajectory = np.asarray(trajectory)
    if (trajectory.ndim != 3 or trajectory.shape[0] < 2
            or trajectory.shape[0] % 2 != 0 or trajectory.shape[1] < 1
            or trajectory.shape[2] < 3):
        raise ValueError("trajectory must have shape (2 * num_types, T, D), with D >= 3")

    num_types = len(trajectory) // 2
    colors = np.array([[246, 185, 59], [31, 119, 180]]) / 255.0
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.plot([-4.0, -1.2], [-0.25, -0.25], 'k-', linewidth=2)
    ax.plot([-4.0, -1.2], [0.75, 0.75], 'k-', linewidth=2)
    ax.plot([-4.0, -1.2], [0.25, 0.25], color='gray',
            linestyle='--', dashes=(21, 4), linewidth=1)

    body = np.array([
        [0.3425, 0],
        [0.14, -0.0885],
        [-0.0725, -0.0885],
        [-0.0725, 0.0885],
        [0.14, 0.0885],
    ])
    for index, branch in enumerate(trajectory):
        color = colors[index // num_types]
        linestyle = '-' if index % num_types == 0 else ':'
        # Display longitudinal y horizontally and lateral x vertically.
        ax.plot(branch[:, 1], branch[:, 0], color=color,
                linestyle=linestyle, marker='*', markersize=3, linewidth=1.5)
        x, y, yaw = branch[-1, :3]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                             [np.sin(yaw), np.cos(yaw)]])
        vertices = body @ rotation.T + np.array([x, y])
        ax.add_patch(patches.Polygon(vertices[:, ::-1], closed=True,
                                     facecolor=color, edgecolor='black', linewidth=1))

    fig.tight_layout()
    if save:
        fig.savefig("result.jpg", bbox_inches='tight', dpi=600)
    return fig
