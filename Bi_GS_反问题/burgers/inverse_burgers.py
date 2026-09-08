import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim import Adam, LBFGS
import math
from pyDOE import lhs
import scipy.io
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from datetime import datetime

class Config:
    seed = 2026
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_min, x_max = -1.0, 1.0
    t_min, t_max = 0.0, 1.0

    nu = 0.01 / np.pi

    DEBUG_MODE = False  # 本地调试完毕后，上服务器前把它改成 False
    INVERSE_MODE = True   # 手动切换：False 为正问题，True 为反问题
    RUN_STRATEGY = "moo_vari"  # 可选 "all", 单策略

    if DEBUG_MODE:
        N_f = 2000           
        N_bc = 100           
        N_ic = 200         
        adam_epochs = 500   
        print_step = 100    
        eval_step = 10     

    else:
        N_f = 20000        
        N_bc = 4000         
        N_ic = 2000         
        adam_epochs = 10000   
        print_step = 500
        eval_step = 100      
        
    lr = 1e-3

    weight_update_freq = 500
    oaw_beta = 0.9

    if DEBUG_MODE:
        lbfgs_max_iter = 200     
    else:
        lbfgs_max_iter = 10000   

    lbfgs_lr = 0.5              
    lbfgs_history_size = 50      
    lbfgs_tolerance_grad = 1e-9  
    lbfgs_tolerance_change = 1e-11  

    layers = [2, 64, 64, 64, 1]

    # Bi-GS-PINN param
    gamma = 0.5

    moo_freq = 1000         
    moo_pop_size = 20       
    moo_n_gen = 5          
    moo_alpha = 100        
    moo_crossover_prob = 0.9  
    moo_mutation_prob = 0.1   
    moo_eta_c = 20           
    moo_eta_m = 20            
    moo_epsilon = 1e-3        
    moo_param_bounds = 2.0    
    gn_lr = 1e-3           
    gn_update_freq = 500    
    gn_update_after = 500    
    gn_alpha = 1.5          
    gn_initial_losses_decay = 1.0 



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.seed)
PI = math.pi


print("Loading reference solution...")
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "burgers_shock.mat")
data = scipy.io.loadmat(data_path)

x_ref = data["x"].flatten()
t_ref = data["t"].flatten()
Exact = np.real(data["usol"])

if Exact.shape[0] == len(t_ref) and Exact.shape[1] == len(x_ref):
    Exact = Exact.T
elif Exact.shape[0] == len(x_ref) and Exact.shape[1] == len(t_ref):
    pass
else:
    raise ValueError(f"Shape mismatch: Exact.shape {Exact.shape} vs x {len(x_ref)} t {len(t_ref)}")

if x_ref[0] > x_ref[-1]:
    x_ref = x_ref[::-1]
    Exact = Exact[::-1, :]
if t_ref[0] > t_ref[-1]:
    t_ref = t_ref[::-1]
    Exact = Exact[:, ::-1]

interp = RectBivariateSpline(x_ref, t_ref, Exact)

def exact_solution_burgers(X):
    X_np = X.detach().cpu().numpy()
    x_vals = X_np[:, 0]
    t_vals = X_np[:, 1]
    u = interp.ev(x_vals, t_vals)
    u = u.reshape(-1, 1)
    return torch.tensor(u, dtype=torch.float32).to(X.device)

def sample_interior(n):
    x = lhs(2, n)
    x[:,0] = x[:,0] * (Config.x_max - Config.x_min) + Config.x_min
    x[:,1] = x[:,1] * (Config.t_max - Config.t_min) + Config.t_min
    X = torch.tensor(x, dtype=torch.float32).to(Config.device)
    return X

def sample_boundary(n):
    t = np.linspace(Config.t_min, Config.t_max, n).reshape(n, 1)
    left = np.hstack([-np.ones((n,1)), t])
    right = np.hstack([ np.ones((n,1)), t])
    X = np.vstack([left, right])
    return torch.tensor(X, dtype=torch.float32).to(Config.device)

def sample_initial(n):
    x = lhs(1, n) * (Config.x_max - Config.x_min) + Config.x_min
    X = np.hstack([x, np.zeros((n,1))])
    return torch.tensor(X, dtype=torch.float32).to(Config.device)

def evaluate_full_mesh(net, x_ref, t_ref, exact):
    X_mesh, T_mesh = np.meshgrid(x_ref, t_ref, indexing="ij")
    X_test_np = np.hstack([X_mesh.reshape(-1, 1), T_mesh.reshape(-1, 1)])
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(Config.device)

    with torch.no_grad():
        u_pred = net(X_test_tensor).cpu().numpy().reshape(exact.shape)

    l2_error = np.linalg.norm(u_pred - exact) / np.linalg.norm(exact)
    return l2_error


class PINN(nn.Module):
    def __init__(self, layers, inverse=False):
        super().__init__()
        self.lb = torch.tensor([Config.x_min, Config.t_min], dtype=torch.float32).to(Config.device)
        self.ub = torch.tensor([Config.x_max, Config.t_max], dtype=torch.float32).to(Config.device)

        self.inverse = inverse
        modules = []
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

        if inverse:
            self.nu = nn.Parameter(torch.tensor(0.03, dtype=torch.float32))
        else:
            self.nu = torch.tensor(Config.nu, dtype=torch.float32, device=Config.device)

    def forward(self, x):
        x_norm = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(x_norm)


def pde_loss(net, X):
    # clone+detach 隔离外部叶张量，防止 X.grad 在多轮 backward 中累积溢出
    X_local = X.clone().detach().requires_grad_(True)
    u = net(X_local)

    grad_u = grad(u, X_local, torch.ones_like(u), create_graph=True)[0]
    u_x = grad_u[:,0:1]
    u_t = grad_u[:,1:2]

    u_xx = grad(u_x, X_local, torch.ones_like(u_x), create_graph=True)[0][:,0:1]

    nu = net.nu if net.inverse else Config.nu
    res = u_t + u * u_x - nu * u_xx
    return torch.mean(res**2)

def boundary_loss(net, X):
    return torch.mean(net(X)**2)

def initial_loss(net, X):
    u_true = -torch.sin(PI * X[:,0:1])
    return torch.mean((net(X) - u_true)**2)


def cosine_similarity(g1, g2):
    dot = (g1 * g2).sum()
    norm1 = torch.norm(g1)
    norm2 = torch.norm(g2)
    return dot / (norm1 * norm2 + 1e-8)

def angle_projection(g_i, g_j):
    dot = (g_i * g_j).sum()
    norm_j_sq = (g_j * g_j).sum()
    if norm_j_sq > 0:
        proj = (dot / norm_j_sq) * g_j
        return g_i - proj
    return g_i

def magnitude_similarity(g_i, g_j):
    norm_i = torch.norm(g_i)
    norm_j = torch.norm(g_j)
    return 2 * norm_i * norm_j / (norm_i**2 + norm_j**2 + 1e-8)

def bidirectional_gradient_surgery(g_list, gamma=0.5):
    K = len(g_list)
    # 1. Angle-based projection (PCGrad style)
    g_proj = [g.clone() for g in g_list]
    for i in range(K):
        for j in range(K):
            if i != j:
                if cosine_similarity(g_proj[i], g_proj[j]) < 0:
                    g_proj[i] = angle_projection(g_proj[i], g_proj[j])
    # 2. Magnitude equalization (SAM-GS style)
    sim_sum = 0.0
    for i in range(K):
        for j in range(K):
            sim_sum += magnitude_similarity(g_proj[i], g_proj[j])
    psi = sim_sum / (K*K)
    if psi < gamma:
        norms = [torch.norm(g) for g in g_proj]
        mean_norm = sum(norms) / K
        for i in range(K):
            if norms[i] > 1e-8:
                g_proj[i] = g_proj[i] * (mean_norm / norms[i])
    g_total = sum(g_proj)
    return g_total

def apply_bigspinn_surgery(losses, weights, net, gamma=0.5):
    param_numels = [p.numel() for p in net.parameters()]
    grads = []
    for w, loss in zip(weights, losses):
        g = torch.autograd.grad(w * loss, net.parameters(), retain_graph=True,
                                create_graph=False, allow_unused=True)
        flat_parts = []
        for grad_val, numel in zip(g, param_numels):
            if grad_val is not None:
                flat_parts.append(grad_val.view(-1))
            else:
                flat_parts.append(torch.zeros(numel, device=Config.device))
        grads.append(torch.cat(flat_parts))

    g_total_flat = bidirectional_gradient_surgery(grads, gamma)

    start = 0
    for param in net.parameters():
        numel = param.numel()
        param.grad = g_total_flat[start:start+numel].view(param.shape)
        start += numel


def apply_pcgrad(losses, net):

    grads = []
    param_numels = [p.numel() for p in net.parameters()]
    for loss in losses:
        g = torch.autograd.grad(loss, net.parameters(), retain_graph=True,
                                create_graph=False, allow_unused=True)
        flat_parts = []
        for grad_val, numel in zip(g, param_numels):
            if grad_val is not None:
                flat_parts.append(grad_val.view(-1))
            else:
                flat_parts.append(torch.zeros(numel, device=Config.device))
        grads.append(torch.cat(flat_parts))

    K = len(grads)
    for i in range(K):
        for j in range(K):
            if i != j and cosine_similarity(grads[i], grads[j]) < 0:
                grads[i] = angle_projection(grads[i], grads[j])

    g_total = sum(grads)

    start = 0
    for param in net.parameters():
        numel = param.numel()
        param.grad = g_total[start:start+numel].view(param.shape)
        start += numel

def get_flat_params(net, inverse=False):
    params_list = []
    for p in net.parameters():
        params_list.append(p.data.detach().cpu().numpy().ravel())
    if inverse:
        params_list.append(np.array([net.nu.item()], dtype=np.float32))
    return np.concatenate(params_list)


def set_flat_params(net, flat, inverse=False):
    start = 0
    for p in net.parameters():
        numel = p.numel()
        p.data = torch.tensor(flat[start:start+numel].reshape(p.shape),
                              dtype=torch.float32, device=Config.device)
        start += numel
    if inverse:
        net.nu.data = torch.tensor([flat[start]], dtype=torch.float32, device=Config.device)
        start += 1
    return start 

def evaluate_fitness(net, flat, inverse, X_f, X_bc, X_ic, X_data=None, u_data=None):

    set_flat_params(net, flat, inverse)

    with torch.enable_grad():
        X_f_local = X_f.clone().detach().requires_grad_(True)
        u = net(X_f_local)
        grad_u = grad(u, X_f_local, torch.ones_like(u), create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]
        u_xx = grad(u_x, X_f_local, torch.ones_like(u_x), create_graph=False)[0][:, 0:1]
        nu_val = net.nu if inverse else Config.nu
        res = u_t + u * u_x - nu_val * u_xx
        loss_pde = torch.mean(res**2).item()

    with torch.no_grad():
        loss_bc = torch.mean(net(X_bc)**2).item()
        u_true_ic = -torch.sin(PI * X_ic[:, 0:1])
        loss_ic = torch.mean((net(X_ic) - u_true_ic)**2).item()

    fitness = [loss_pde, loss_bc, loss_ic]
    if inverse and X_data is not None:
        with torch.no_grad():
            u_pred = net(X_data)
            loss_data = torch.mean((u_pred - u_data)**2).item()
        fitness.append(loss_data)

    return np.array(fitness, dtype=np.float64)


def non_dominated_sort(fitness):
    N = fitness.shape[0]
    dominated_count = np.zeros(N, dtype=int)
    dominates_list = [[] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # i dominates j iff all(f_i <= f_j) and any(f_i < f_j)
            if np.all(fitness[i] <= fitness[j]) and np.any(fitness[i] < fitness[j]):
                dominates_list[i].append(j)
            elif np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                dominated_count[i] += 1

    fronts = []
    current_front = [i for i in range(N) if dominated_count[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def crowding_distance(fitness, front):

    if len(front) <= 2:
        return np.full(len(front), np.inf)

    M = fitness.shape[1]
    dist = np.zeros(len(front))
    front_arr = np.array(front)

    for m in range(M):
        obj_values = fitness[front_arr, m]
        sorted_idx = np.argsort(obj_values)
        obj_sorted = obj_values[sorted_idx]

        dist[sorted_idx[0]] = np.inf
        dist[sorted_idx[-1]] = np.inf

        obj_range = obj_sorted[-1] - obj_sorted[0]
        if obj_range > 0:
            for k in range(1, len(front) - 1):
                dist[sorted_idx[k]] += (obj_sorted[k+1] - obj_sorted[k-1]) / obj_range

    return dist


def tournament_selection(fronts, crowding_dists, fitness, n_select):
    rank = np.zeros(fitness.shape[0], dtype=int)
    crowd = np.zeros(fitness.shape[0])
    for f_idx, front in enumerate(fronts):
        for i_idx, ind in enumerate(front):
            rank[ind] = f_idx
            crowd[ind] = crowding_dists[f_idx][i_idx]

    selected = []
    for _ in range(n_select):
        a, b = np.random.choice(fitness.shape[0], 2, replace=False)
        if rank[a] < rank[b]:
            selected.append(a)
        elif rank[b] < rank[a]:
            selected.append(b)
        else:
            selected.append(a if crowd[a] >= crowd[b] else b)

    return selected


def sbx_crossover(p1, p2, eta_c=20, prob=0.9):
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if np.random.rand() < prob:
            if np.abs(p2[i] - p1[i]) > 1e-14:
                if np.random.rand() < 0.5:
                    beta = 2.0 * np.random.rand()
                    if beta <= 1.0:
                        beta_q = beta ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - beta)) ** (1.0 / (eta_c + 1.0))
                else:
                    beta = 2.0 * np.random.rand()
                    if beta <= 1.0:
                        beta_q = (1.0 / (2.0 - beta)) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = beta ** (1.0 / (eta_c + 1.0))
                c1[i] = 0.5 * ((1 + beta_q) * p1[i] + (1 - beta_q) * p2[i])
                c2[i] = 0.5 * ((1 - beta_q) * p1[i] + (1 + beta_q) * p2[i])
    return c1, c2


def polynomial_mutation(ind, bounds, eta_m=20, prob=0.1):
    mutated = ind.copy()
    for i in range(len(ind)):
        if np.random.rand() < prob:
            delta = np.random.rand()
            if delta < 0.5:
                delta_q = (2.0 * delta) ** (1.0 / (eta_m + 1.0)) - 1.0
            else:
                delta_q = 1.0 - (2.0 * (1.0 - delta)) ** (1.0 / (eta_m + 1.0))
            mutated[i] += delta_q * (bounds - (-bounds))
            mutated[i] = np.clip(mutated[i], -bounds, bounds)
    return mutated


def nsga2_pareto_search(net, inverse, X_f, X_bc, X_ic, X_data, u_data):
    pop_size = Config.moo_pop_size
    n_gen = Config.moo_n_gen
    bounds = Config.moo_param_bounds
    n_params_net = sum(p.numel() for p in net.parameters())
    n_vars = n_params_net + (1 if inverse else 0)


    population = []
    current_flat = get_flat_params(net, inverse)
    noise_scale = 0.05 * bounds 
    for _ in range(pop_size // 2):
        noisy = current_flat + np.random.randn(n_vars) * noise_scale
        noisy = np.clip(noisy, -bounds, bounds)
        population.append(noisy)

    for _ in range(pop_size - pop_size // 2):
        random_ind = np.random.uniform(-bounds, bounds, n_vars)
        population.append(random_ind)
    population = np.array(population, dtype=np.float64)

    fitness = np.array([evaluate_fitness(net, ind, inverse, X_f, X_bc, X_ic, X_data, u_data)
                        for ind in population])

    for gen in range(n_gen):
        fronts = non_dominated_sort(fitness)
        crowd_dists = [crowding_distance(fitness, f) for f in fronts]

        parent_idx = tournament_selection(fronts, crowd_dists, fitness, pop_size)

        offspring = []
        for k in range(0, pop_size, 2):
            p1 = population[parent_idx[k]]
            p2 = population[parent_idx[min(k+1, pop_size-1)]]
            c1, c2 = sbx_crossover(p1, p2, Config.moo_eta_c, Config.moo_crossover_prob)
            c1 = polynomial_mutation(c1, bounds, Config.moo_eta_m, Config.moo_mutation_prob)
            c2 = polynomial_mutation(c2, bounds, Config.moo_eta_m, Config.moo_mutation_prob)
            offspring.append(c1)
            offspring.append(c2)
        offspring = np.array(offspring[:pop_size], dtype=np.float64)
        off_fitness = np.array([evaluate_fitness(net, ind, inverse, X_f, X_bc, X_ic, X_data, u_data)
                                for ind in offspring])

        merged_pop = np.vstack([population, offspring])
        merged_fitness = np.vstack([fitness, off_fitness])

        merged_fronts = non_dominated_sort(merged_fitness)
        merged_crowd = [crowding_distance(merged_fitness, f) for f in merged_fronts]

        new_pop = []
        new_fitness = []
        for front in merged_fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
                new_fitness.extend([merged_fitness[i] for i in front])
            else:
                f_idx = merged_fronts.index(front)
                remaining = pop_size - len(new_pop)
                cd = merged_crowd[f_idx]
                sorted_by_cd = sorted(zip(front, cd), key=lambda x: x[1], reverse=True)
                for ind_idx, _ in sorted_by_cd[:remaining]:
                    new_pop.append(ind_idx)
                    new_fitness.append(merged_fitness[ind_idx])
                break

        population = merged_pop[new_pop]
        fitness = np.array(new_fitness, dtype=np.float64)

    set_flat_params(net, population[0], inverse)

    final_fronts = non_dominated_sort(fitness)
    pareto_fitness = np.array([fitness[i] for i in final_fronts[0]])

    return pareto_fitness


def compute_vari_weights(pareto_fitness, loss_history_buffer, num_losses, inverse=False):
    M = num_losses
    P = pareto_fitness.shape[0]


    if len(loss_history_buffer) == 0:
        L_j_pre = np.mean(pareto_fitness, axis=0)
    else:
        hist_arr = np.array(loss_history_buffer)  
        L_j_pre = np.min(hist_arr, axis=0)


    bar_F_j = np.mean(pareto_fitness, axis=0)     
    F_min = np.min(pareto_fitness, axis=0)
    F_max = np.max(pareto_fitness, axis=0)

    sigma_j = np.zeros(M)
    for j in range(M):
        if F_max[j] - F_min[j] > 1e-12:
            F_norm = (pareto_fitness[:, j] - F_min[j]) / (F_max[j] - F_min[j])
        else:
            F_norm = np.zeros(P)
        sigma_j[j] = np.std(F_norm) + 1e-12


    r_j = bar_F_j / (L_j_pre + 1e-12)
    r_j = np.maximum(r_j, 1e-12)


    s_j = sigma_j / r_j


    if inverse:
        L_data_current = bar_F_j[-1] if len(loss_history_buffer) == 0 else loss_history_buffer[-1][-1]
        if L_data_current > Config.moo_epsilon:
            T = np.ones(M)
            T[-1] = 0.5  # data loss 温度更低，差异化更大
        else:
            T = np.ones(M)
    else:
        T = np.ones(M)

    exp_scores = np.exp(np.clip(s_j / T, -50, 50))
    lambda_j = M * exp_scores / np.sum(exp_scores)

    return torch.tensor(lambda_j, dtype=torch.float32, device=Config.device)


def moo_vari_update(net, inverse, X_f, X_bc, X_ic, X_data, u_data,
                    loss_history_buffer, num_losses):

    pareto_fitness = nsga2_pareto_search(net, inverse, X_f, X_bc, X_ic, X_data, u_data)
    weights = compute_vari_weights(pareto_fitness, loss_history_buffer, num_losses, inverse)

    return weights


def apply_gradnorm(losses, net, weights, initial_losses=None, lr=1e-4, alpha=0.0):
    params_list = list(net.parameters())
    grad_norm_tensor = params_list[-2]

    num_losses = len(losses)
    init_loss_weights_for_sum = weights.sum().detach()
    loss_weights = weights.detach().clone().requires_grad_(True)

    grad_norms = []
    for weight, loss in zip(loss_weights, losses):
        gradients, = torch.autograd.grad(weight * loss, grad_norm_tensor,
                                         create_graph=True, retain_graph=True)
        grad_norms.append(gradients.norm(p=2))
    grad_norms = torch.stack(grad_norms)  # (num_losses,)

    grad_norm_average = grad_norms.mean().detach()

    if alpha > 0 and initial_losses is not None:
        # Restoring force: gradient_target = G_avg × (relative_training_rate)^{-alpha}
        with torch.no_grad():
            loss_ratio = torch.stack([l.detach() for l in losses]) / (initial_losses + 1e-12)
            relative_training_rate = torch.nn.functional.normalize(loss_ratio, p=1, dim=0) * num_losses
            gradient_target = (grad_norm_average * (relative_training_rate ** -alpha)).detach()
    else:
        gradient_target = grad_norm_average.expand(num_losses).detach()

    grad_norm_loss = F.l1_loss(grad_norms, gradient_target)

    loss_weights_grad = torch.autograd.grad(grad_norm_loss, loss_weights)[0]

    updated_loss_weights = loss_weights.detach() - loss_weights_grad * lr
    updated_loss_weights = torch.clamp(updated_loss_weights, min=1e-8)
    renormalized_loss_weights = torch.nn.functional.normalize(updated_loss_weights, p=1, dim=0) * init_loss_weights_for_sum

    return renormalized_loss_weights.detach()


def main(strategy="standard", inverse=False, seed=0, timestamp=""):
    net = PINN(Config.layers, inverse=inverse).to(Config.device)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    prefix = "inverse_" if inverse else ""
    opt_tag = f"A{Config.adam_epochs}_L{Config.lbfgs_max_iter}"
    if timestamp:
        current_save_dir = os.path.join(current_dir, "results", f"{prefix}{strategy}_{opt_tag}_{timestamp}", f"seed_{seed}")
    else:
        current_save_dir = os.path.join(current_dir, "results", f"{prefix}{strategy}_{opt_tag}", f"seed_{seed}")

    os.makedirs(current_save_dir, exist_ok=True)

    X_f = sample_interior(Config.N_f)
    X_bc = sample_boundary(Config.N_bc)
    X_ic = sample_initial(Config.N_ic)

    # For inverse problem: data points
    if inverse:
        N_data = 200
        x_data = np.random.uniform(Config.x_min, Config.x_max, (N_data, 1))
        t_data = np.random.uniform(Config.t_min, Config.t_max, (N_data, 1))
        X_data_np = np.hstack([x_data, t_data])
        X_data = torch.tensor(X_data_np, dtype=torch.float32).to(Config.device)
        u_data = exact_solution_burgers(X_data)
        noise = 0.01 * torch.randn_like(u_data)
        u_data = u_data + noise
    else:
        X_data = None
        u_data = None


    print("\n" + "="*60)
    print("Stage 1: Adam Training")
    print("="*60)

    optimizer = Adam(net.parameters(), lr=Config.lr)
    start = time.time()

    history_iter = []
    history_loss = []
    history_l2 = []

    iter_to_1e3 = -1
    current_iter = 0

    num_losses = 4 if inverse else 3
    weights = torch.ones(num_losses, device=Config.device) / num_losses

    loss_history_buffer = []

    gn_initial_losses = None

    def update_weights_smooth(prev_weights, optimal_weights, beta=0.9):
        return beta * prev_weights + (1 - beta) * optimal_weights

    for epoch in range(Config.adam_epochs):
        loss_pde = pde_loss(net, X_f)
        loss_bc = boundary_loss(net, X_bc)
        loss_ic = initial_loss(net, X_ic)

        losses = [loss_pde, loss_bc, loss_ic]
        if inverse:
            u_data_pred = net(X_data)
            loss_data = torch.mean((u_data_pred - u_data)**2)
            losses.append(loss_data)

        optimizer.zero_grad()

        if strategy == "standard":
            loss = sum(losses)
            loss.backward()
            optimizer.step()
        elif strategy == "pcgrad":
            apply_pcgrad(losses, net)
            optimizer.step()
            loss = sum(losses).detach()
        elif strategy == "moo_vari":
            if epoch > 0 and epoch % Config.moo_freq == 0:
                weights = moo_vari_update(net, inverse, X_f, X_bc, X_ic,
                                          X_data, u_data,
                                          loss_history_buffer, num_losses)
                if epoch % Config.print_step == 0 or Config.DEBUG_MODE:
                    print(f"  [MOO-VARI] Updated weights: {weights.detach().cpu().numpy()}")
            loss = sum(w * l for w, l in zip(weights, losses))
            loss.backward()
            optimizer.step()
        elif strategy == "gradnorm":
            if epoch >= Config.gn_update_after and epoch % Config.gn_update_freq == 0:
                if gn_initial_losses is None:
                    gn_initial_losses = torch.stack([l.detach() for l in losses])
                elif Config.gn_alpha > 0 and Config.gn_initial_losses_decay < 1.0:
                    current_losses = torch.stack([l.detach() for l in losses])
                    gn_initial_losses = gn_initial_losses * Config.gn_initial_losses_decay + \
                                        current_losses * (1.0 - Config.gn_initial_losses_decay)
                weights = apply_gradnorm(losses, net, weights, gn_initial_losses,
                                         Config.gn_lr, Config.gn_alpha)
                if epoch % Config.print_step == 0 or Config.DEBUG_MODE:
                    print(f"  [GradNorm] Updated weights: {weights.detach().cpu().numpy()}")
            loss = sum(w * l for w, l in zip(weights, losses))
            loss.backward()
            optimizer.step()
        elif strategy == "bi_gs":
            params = list(net.parameters())

            grad_norms = []
            for l in losses:
                grad_l = torch.autograd.grad(l, params, retain_graph=True, create_graph=False, allow_unused=True)
                norm = sum(p.norm().item()**2 for p in grad_l if p is not None)**0.5
                grad_norms.append(norm)
            grad_norms = torch.tensor(grad_norms, device=Config.device)

            if epoch % Config.weight_update_freq == 0:
                inv_sq = 1.0 / (grad_norms**2 + 1e-8)
                opt_weights = inv_sq / inv_sq.sum()
                weights = update_weights_smooth(weights, opt_weights, Config.oaw_beta)

            apply_bigspinn_surgery(losses, weights, net, Config.gamma)
            optimizer.step()
            loss = sum(w * l for w, l in zip(weights, losses)).detach()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


        if strategy == "moo_vari":
            loss_vals = [l.detach().item() for l in losses]
            loss_history_buffer.append(loss_vals)
            if len(loss_history_buffer) > Config.moo_alpha:
                loss_history_buffer.pop(0)

        if current_iter % Config.eval_step == 0:
            net.eval()
            l2 = evaluate_full_mesh(net, x_ref, t_ref, Exact)
            net.train()

            history_iter.append(current_iter)
            history_loss.append(loss.item())
            history_l2.append(l2)

            if l2 < 1e-3 and iter_to_1e3 == -1:
                iter_to_1e3 = current_iter

            if current_iter % Config.print_step == 0:
                nu_str = f"| nu={net.nu.item():.6f}" if inverse else ""
                print(f"Adam Epoch {epoch:5d} | Loss={loss.item():.2e} | L2={l2:.3e} {nu_str}")

        current_iter += 1

    adam_time = time.time() - start
    print(f"\nAdam finished. Time: {adam_time:.2f} sec")


    if Config.lbfgs_max_iter > 0:
        print("\n" + "="*60)
        print("Stage 2: L-BFGS Fine-tuning")
        print("="*60)

        final_weights = weights.detach().clone()

        optimizer_lbfgs = LBFGS(
            net.parameters(),
            lr=Config.lbfgs_lr,
            max_iter=Config.lbfgs_max_iter,
            max_eval=None,  
            history_size=Config.lbfgs_history_size,
            tolerance_grad=Config.lbfgs_tolerance_grad,
            tolerance_change=Config.lbfgs_tolerance_change,
            line_search_fn="strong_wolfe"  
        )

        lbfgs_loss_history = []
        lbfgs_l2_history = []
        lbfgs_iter_count = [0]  

        def closure():
            optimizer_lbfgs.zero_grad()

            loss_pde = pde_loss(net, X_f)
            loss_bc = boundary_loss(net, X_bc)
            loss_ic = initial_loss(net, X_ic)

            losses_lbfgs = [loss_pde, loss_bc, loss_ic]
            if inverse:
                u_data_pred = net(X_data)
                loss_data = torch.mean((u_data_pred - u_data) ** 2)
                losses_lbfgs.append(loss_data)

            loss = sum(w * l for w, l in zip(final_weights, losses_lbfgs))
            loss.backward()

            lbfgs_iter_count[0] += 1
            if lbfgs_iter_count[0] % 500 == 0 or lbfgs_iter_count[0] == 1:
                with torch.no_grad():
                    l2 = evaluate_full_mesh(net, x_ref, t_ref, Exact)
                lbfgs_loss_history.append(loss.item())
                lbfgs_l2_history.append(l2)
                nu_str = f"| nu={net.nu.item():.6f}" if inverse else ""
                print(f"L-BFGS Iter {lbfgs_iter_count[0]:5d} | Loss={loss.item():.2e} | L2={l2:.3e} {nu_str}")

            return loss

        net.train()
        lbfgs_start = time.time()
        optimizer_lbfgs.step(closure)
        lbfgs_time = time.time() - lbfgs_start

        net.eval()
        l2_lbfgs_final = evaluate_full_mesh(net, x_ref, t_ref, Exact)
        print(f"\nL-BFGS finished. Time: {lbfgs_time:.2f} sec")
        print(f"L-BFGS Final L2: {l2_lbfgs_final:.3e}  (total L-BFGS iters: {lbfgs_iter_count[0]})")

        adam_final_iter = history_iter[-1] if history_iter else 0
        for i, (loss_val, l2_val) in enumerate(zip(lbfgs_loss_history, lbfgs_l2_history)):
            history_iter.append(adam_final_iter + (i + 1) * 500)
            history_loss.append(loss_val)
            history_l2.append(l2_val)

    print(f"\nTraining Complete! Total Time: {time.time()-start:.2f} sec")

    net.eval()
    l2_final = evaluate_full_mesh(net, x_ref, t_ref, Exact)

    if Config.DEBUG_MODE:
        print("\n[WARNING] DEBUG MODE ENABLED -- Results are NOT for paper!")

    print("\n" + "="*60)
    print("    [FINAL EXPERIMENTAL METRICS (For Paper)]")
    print("="*60)
    val_10_3 = l2_final * 1000
    print("【Table 1: Predictive Accuracy】")
    print(f"  --> {strategy.upper()}-PINN L2 Error: {val_10_3:.2f} (x 10^-3)  <-- Check this vs 1.1")
    print(f"      (Raw L2 value: {l2_final:.3e})")

    print("\n【Table 2: Convergence Behavior】")
    if iter_to_1e3 != -1:
        print(f"  --> Iterations to 1e-3: {iter_to_1e3}  <-- Check this vs 6,500")
    else:
        print(f"  --> Iterations to 1e-3: Did not reach 1e-3. Final={l2_final:.3e}")
    print("="*60)


    print("\nGenerating Plots...")
    plot_name = {"standard": "PINN (Uniform)", "pcgrad": "PCGrad", "bi_gs": "Bi-GS-PINN", "moo_vari": "MOO-VARI-PINN", "gradnorm": "GradNorm"}[strategy]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_iter, history_loss, 'b-', label='Total Loss', linewidth=1.5)
    if Config.lbfgs_max_iter > 0:
        plt.axvline(x=Config.adam_epochs, color='r', linestyle='--', alpha=0.7, label='Adam -> L-BFGS')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve ({plot_name})')
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_iter, history_l2, 'g-', label='Relative L2 Error', linewidth=1.5)
    plt.axhline(y=1e-3, color='k', linestyle='-', label='Tolerance ($10^{-3}$)')
    if Config.lbfgs_max_iter > 0:
        plt.axvline(x=Config.adam_epochs, color='r', linestyle='--', alpha=0.7, label='Adam -> L-BFGS')
    if iter_to_1e3 != -1:
        plt.scatter([iter_to_1e3], [1e-3], color='red', zorder=5)
        plt.text(iter_to_1e3, 1.5e-3, f' {iter_to_1e3} iters', color='red', fontsize=10)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Relative L2 Error')
    plt.title(f'Validation L2 Error Curve ({plot_name})')
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(current_save_dir, "Convergence_Curve.png"), dpi=300)
    plt.close()

    X_mesh, T_mesh = np.meshgrid(x_ref, t_ref, indexing="ij")
    X_test_np = np.hstack([X_mesh.reshape(-1, 1), T_mesh.reshape(-1, 1)])
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(Config.device)

    with torch.no_grad():
        u_pred = net(X_test_tensor).cpu().numpy().reshape(Exact.shape)

    error_matrix = np.abs(Exact - u_pred)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    h1 = plt.pcolormesh(T_mesh, X_mesh, Exact, cmap='jet', shading='gouraud')
    plt.colorbar(h1)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Exact u(t,x)')

    plt.subplot(1, 3, 2)
    h2 = plt.pcolormesh(T_mesh, X_mesh, u_pred, cmap='jet', shading='gouraud')
    plt.colorbar(h2)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(f'Predicted u(t,x) ({plot_name})')

    plt.subplot(1, 3, 3)
    h3 = plt.pcolormesh(T_mesh, X_mesh, error_matrix, cmap='jet', shading='gouraud')
    plt.colorbar(h3)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Absolute Error')

    plt.tight_layout()
    plt.savefig(os.path.join(current_save_dir, "Burgers_Heatmap.png"), dpi=300)
    plt.close()

    with open(os.path.join(current_save_dir, "config.txt"), "w") as f:
        f.write(f"strategy = {strategy}\n")
        f.write(f"inverse_mode = {inverse}\n")
        f.write(f"seed = {seed}\n")
        f.write(f"N_f = {Config.N_f}\n")
        f.write(f"N_bc = {Config.N_bc}\n")
        f.write(f"N_ic = {Config.N_ic}\n")
        f.write(f"adam_epochs = {Config.adam_epochs}\n")
        f.write(f"lr = {Config.lr}\n")
        if Config.lbfgs_max_iter > 0:
            f.write(f"lbfgs_max_iter = {Config.lbfgs_max_iter}\n")
            f.write(f"lbfgs_lr = {Config.lbfgs_lr}\n")
            f.write(f"lbfgs_history_size = {Config.lbfgs_history_size}\n")
        if strategy == "bi_gs":
            f.write(f"gamma = {Config.gamma}\n")
            f.write(f"weight_update_freq = {Config.weight_update_freq}\n")
            f.write(f"oaw_beta = {Config.oaw_beta}\n")

    print(f"Plots saved to {current_save_dir}")

    torch.save(net.state_dict(), os.path.join(current_save_dir, "model_final.pth"))

    return l2_final, iter_to_1e3, net.nu.item() if inverse else None

if __name__ == "__main__":
    if Config.RUN_STRATEGY == "all":
        strategies_to_test = ["standard", "pcgrad", "bi_gs", "moo_vari", "gradnorm"]
    elif "," in Config.RUN_STRATEGY:
        strategies_to_test = [s.strip() for s in Config.RUN_STRATEGY.split(",")]
    else:
        strategies_to_test = [Config.RUN_STRATEGY]

    inverse_mode = Config.INVERSE_MODE

    num_runs = 2 if Config.DEBUG_MODE else 10
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    mode_str = "INVERSE" if inverse_mode else "FORWARD"
    for strategy in strategies_to_test:
        print(f"\n" + "="*60)
        print(f"=== STARTING EVALUATION FOR STRATEGY: {strategy.upper()} ({mode_str}) ===")
        print("="*60)

        key = f"{strategy}_{mode_str}"
        l2_list = []
        iter_list = []
        nu_list = []

        for seed in range(max(5, num_runs) if not Config.DEBUG_MODE else num_runs):
            print(f"\n" + "-"*40)
            print(f"--- {strategy.upper()} - Run {seed} ---")
            print("-"  *40)
            set_seed(seed)

            l2_final, iter_to_1e3, nu_final = main(strategy=strategy, inverse=inverse_mode, seed=seed, timestamp=run_timestamp)

            l2_list.append(l2_final)
            if iter_to_1e3 != -1:
                iter_list.append(iter_to_1e3)
            if inverse_mode:
                nu_list.append(nu_final)

        results[key] = {
            "l2_mean": np.mean(l2_list) * 1000,
            "l2_std": np.std(l2_list) * 1000,
            "iters_mean": int(np.mean(iter_list)) if len(iter_list) > 0 else -1,
            "nu_mean": np.mean(nu_list) if inverse_mode else None,
            "nu_std": np.std(nu_list) if inverse_mode else None
        }


    lines = []
    lines.append("="*80)
    lines.append("=== FINAL BENCHMARK RESULTS (For Copy-Pasting into Paper) ===")
    lines.append("="*80)
    lines.append("| Method                 | Mode    | L2 Error (x 10^-3) | Iter to 1e-3 | Est. nu (true: 0.003183) |")
    lines.append("|------------------------|---------|--------------------|--------------|--------------------------|")

    mode_str_table = "Inverse" if inverse_mode else "Forward"
    mode_key = "INVERSE" if inverse_mode else "FORWARD"
    for strategy in strategies_to_test:
        key = f"{strategy}_{mode_key}"
        res = results[key]
        l2_str = f"{res['l2_mean']:.1f} ± {res['l2_std']:.1f}"
        iter_str = f"{res['iters_mean']}" if res['iters_mean'] != -1 else "N/A"
        name_str = {"standard": "PINN (uniform)", "pcgrad": "PCGrad", "bi_gs": "Bi-GS-PINN", "moo_vari": "MOO-VARI-PINN", "gradnorm": "GradNorm"}[strategy]
        nu_str = f"{res['nu_mean']:.6f} ± {res['nu_std']:.6f}" if inverse_mode else "N/A"
        lines.append(f"| {name_str:<22} | {mode_str_table:<7} | {l2_str:<18} | {iter_str:<12} | {nu_str:<24} |")
    lines.append("="*80)


    for line in lines:
        print(line)

    table_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(table_dir, exist_ok=True)
    table_path = os.path.join(table_dir, f"benchmark_table_{run_timestamp}.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nBenchmark table saved to: {table_path}")
