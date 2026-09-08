# ablation_3DMMS_Bi_GS_PINN.py
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
import matplotlib.pyplot as plt
from datetime import datetime

class Config:
    seed = 2026
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 计算域：x, y, p, t ∈ [0, 1]
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    p_min, p_max = 0.0, 1.0
    t_min, t_max = 0.0, 1.0

    # 物理参数
    g = 9.81
    rho0 = 1.225
    f0 = 1e-4
    beta_phys = 1e-11

    # MMS 解参数
    A = 1.0
    B = 1.0
    C = 0.1
    D = 1.0
    alpha_mms = 0.1
    beta_mms = 0.1
    gamma_mms = 0.1

    # 无量纲系数
    x_0 = 1.0
    t_0 = 1.0
    p_0 = 1.0
    u_0 = 1.0
    h_0 = 1.0

    # ==========================================
    # 🏃 运行模式切换：本地测试 vs 服务器训练
    # ==========================================
    DEBUG_MODE = False  # 本地调试完毕后，上服务器前把它改成 False
    # 消融变体: "full", "no_angle", "no_mag", "no_oaw", "no_smooth", "no_moo", "all_ablation"
    RUN_STRATEGY = "no_mag"  # 消融实验策略选择

    if DEBUG_MODE:
        N_f = 2000        
        N_bc = 600   
        N_ic = 400   

        adam_epochs = 500 
        print_step = 100
        eval_step = 10

    else:
        N_f = 20000     
        N_bc = 6000      
        N_ic = 4000   

        adam_epochs = 15000
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

    layers = [4, 64, 64, 64, 4] 

    # Bi-GS-PINN param
    gamma = 0.5


    USE_ANGLE_PROJECTION = True       
    USE_MAGNITUDE_EQUALIZATION = True 
    USE_OAW_WEIGHTS = True            
    OAW_BETA = 0.9  


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

def exact_solution(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    p = X[:, 2:3]
    t = X[:, 3:4]

    A = Config.A; B = Config.B; C = Config.C; D = Config.D
    alpha = Config.alpha_mms; beta_m = Config.beta_mms; gamma = Config.gamma_mms

    u = A * torch.sin(PI * x) * torch.cos(PI * y) * torch.exp(-t) * torch.exp(-alpha * p)
    v = B * torch.cos(PI * x) * torch.sin(PI * y) * torch.exp(-t) * torch.exp(-beta_m * p)
    w = C * torch.sin(PI * p) * torch.cos(PI * x) * torch.sin(PI * y) * torch.exp(-t)
    h = D * torch.cos(PI * x) * torch.cos(PI * y) * torch.exp(-t) * torch.exp(-gamma * p)

    return u, v, w, h


def forcing_terms(X):
    X = X.clone().detach().requires_grad_(True)
    u, v, w, h = exact_solution(X)

    def d(f, idx):
        return grad(f, X, torch.ones_like(f), create_graph=True)[0][:, idx:idx+1]

    u_t, u_x, u_y, u_p = d(u, 3), d(u, 0), d(u, 1), d(u, 2)
    v_t, v_x, v_y, v_p = d(v, 3), d(v, 0), d(v, 1), d(v, 2)
    w_p = d(w, 2)
    h_x, h_y = d(h, 0), d(h, 1)

    y_coord = X[:, 1:2]
    cor = Config.f0 + Config.beta_phys * y_coord * Config.x_0

    x0 = Config.x_0; t0 = Config.t_0; p0 = Config.p_0
    u0 = Config.u_0; h0 = Config.h_0

    Fu = (x0 / u0 / t0) * u_t + u * u_x + v * u_y + (x0 / u0 / t0) * w * u_p \
         - (x0 / u0) * cor * v + (h0 / (u0**2)) * h_x

    Fv = (x0 / u0 / t0) * v_t + u * v_x + v * v_y + (x0 / u0 / t0) * w * v_p \
         + (x0 / u0) * cor * u + (h0 / (u0**2)) * h_y

    Fc = u_x + v_y + (x0 / p0) * w_p

    return Fu.detach(), Fv.detach(), Fc.detach()


def sample_interior(n):
    X_np = lhs(4, n)
    X_np[:, 0] = X_np[:, 0] * (Config.x_max - Config.x_min) + Config.x_min
    X_np[:, 1] = X_np[:, 1] * (Config.y_max - Config.y_min) + Config.y_min
    X_np[:, 2] = X_np[:, 2] * (Config.p_max - Config.p_min) + Config.p_min
    X_np[:, 3] = X_np[:, 3] * (Config.t_max - Config.t_min) + Config.t_min
    X = torch.tensor(X_np, dtype=torch.float32).to(Config.device)
    return X

def sample_boundary(n):
    n_side = max(n // 6, 1)
    samples = []
    for val, dim in [(0.0, 0), (1.0, 0), (0.0, 1), (1.0, 1), (0.0, 2), (1.0, 2)]:
        side = lhs(4, n_side)
        side[:, dim] = val 
        side[:, 0] = side[:, 0] * (Config.x_max - Config.x_min) + Config.x_min
        side[:, 1] = side[:, 1] * (Config.y_max - Config.y_min) + Config.y_min
        side[:, 2] = side[:, 2] * (Config.p_max - Config.p_min) + Config.p_min
        side[:, 3] = side[:, 3] * (Config.t_max - Config.t_min) + Config.t_min
        samples.append(side)
    X_np = np.vstack(samples)
    return torch.tensor(X_np, dtype=torch.float32).to(Config.device)

def sample_initial(n):
    X_np = lhs(4, n)
    X_np[:, 3] = Config.t_min  # t 固定为 0
    X_np[:, 0] = X_np[:, 0] * (Config.x_max - Config.x_min) + Config.x_min
    X_np[:, 1] = X_np[:, 1] * (Config.y_max - Config.y_min) + Config.y_min
    X_np[:, 2] = X_np[:, 2] * (Config.p_max - Config.p_min) + Config.p_min
    return torch.tensor(X_np, dtype=torch.float32).to(Config.device)

def sample_test_4d(N, device=None):
    X_np = lhs(4, N)
    X_np[:, 0] = X_np[:, 0] * (Config.x_max - Config.x_min) + Config.x_min
    X_np[:, 1] = X_np[:, 1] * (Config.y_max - Config.y_min) + Config.y_min
    X_np[:, 2] = X_np[:, 2] * (Config.p_max - Config.p_min) + Config.p_min
    X_np[:, 3] = X_np[:, 3] * (Config.t_max - Config.t_min) + Config.t_min
    X = torch.tensor(X_np, dtype=torch.float32)
    return X.to(device) if device else X

def evaluate_l2_4d(net, X_test, device='cpu'):
    net.eval()
    with torch.no_grad():
        pred = net(X_test)
        u_p, v_p, w_p, h_p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
        u_t, v_t, w_t, h_t = exact_solution(X_test)

        l2_u = torch.norm(u_p - u_t) / (torch.norm(u_t) + 1e-12)
        l2_v = torch.norm(v_p - v_t) / (torch.norm(v_t) + 1e-12)
        l2_w = torch.norm(w_p - w_t) / (torch.norm(w_t) + 1e-12)
        l2_h = torch.norm(h_p - h_t) / (torch.norm(h_t) + 1e-12)
    net.train()
    return l2_u.item(), l2_v.item(), l2_w.item(), l2_h.item()


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.lb = torch.tensor([Config.x_min, Config.y_min, Config.p_min, Config.t_min],
                               dtype=torch.float32).to(Config.device)
        self.ub = torch.tensor([Config.x_max, Config.y_max, Config.p_max, Config.t_max],
                               dtype=torch.float32).to(Config.device)
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, X):
        X_norm = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(X_norm)


def pde_residuals(net, X, forcing_fn=None):
    X = X.clone().detach().requires_grad_(True)
    pred = net(X)
    u, v, w, h = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]

    def d(f, idx):
        return grad(f, X, torch.ones_like(f), create_graph=True)[0][:, idx:idx+1]

    u_t, u_x, u_y, u_p = d(u, 3), d(u, 0), d(u, 1), d(u, 2)
    v_t, v_x, v_y, v_p = d(v, 3), d(v, 0), d(v, 1), d(v, 2)
    w_p = d(w, 2)
    h_x, h_y = d(h, 0), d(h, 1)

    yc = X[:, 1:2]
    cor = Config.f0 + Config.beta_phys * yc * Config.x_0

    x0 = Config.x_0; t0 = Config.t_0; p0 = Config.p_0
    u0 = Config.u_0; h0 = Config.h_0

    Ru = (x0 / u0 / t0) * u_t + u * u_x + v * u_y + (x0 / u0 / t0) * w * u_p \
         - (x0 / u0) * cor * v + (h0 / (u0**2)) * h_x

    Rv = (x0 / u0 / t0) * v_t + u * v_x + v * v_y + (x0 / u0 / t0) * w * v_p \
         + (x0 / u0) * cor * u + (h0 / (u0**2)) * h_y

    Rc = u_x + v_y + (x0 / p0) * w_p

    if forcing_fn is not None:
        Fu, Fv, Fc = forcing_fn(X)
        Ru = Ru - Fu
        Rv = Rv - Fv
        Rc = Rc - Fc

    return Ru, Rv, Rc

def boundary_loss(net, X):
    pred = net(X)
    exact = torch.cat(exact_solution(X), dim=1)
    return torch.mean((pred - exact) ** 2)

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
    g_proj = [g.clone() for g in g_list]

    if Config.USE_ANGLE_PROJECTION:
        for i in range(K):
            for j in range(K):
                if i != j:
                    if cosine_similarity(g_proj[i], g_proj[j]) < 0:
                        g_proj[i] = angle_projection(g_proj[i], g_proj[j])

    if Config.USE_MAGNITUDE_EQUALIZATION:
        sim_sum = 0.0
        for i in range(K):
            for j in range(K):
                sim_sum += magnitude_similarity(g_proj[i], g_proj[j])
        psi = sim_sum / (K * K)
        if psi < gamma:
            norms = [torch.norm(g) for g in g_proj]
            mean_norm = sum(norms) / K
            for i in range(K):
                if norms[i] > 1e-8:
                    g_proj[i] = g_proj[i] * (mean_norm / norms[i])
    g_total = sum(g_proj)
    return g_total

def apply_bigspinn_surgery(losses, weights, net, gamma=0.5):

    param_shapes = [p.shape for p in net.parameters()]
    param_numels = [p.numel() for p in net.parameters()]

    grads = []
    for w, loss in zip(weights, losses):
        g = torch.autograd.grad(w * loss, net.parameters(), retain_graph=True,
                                create_graph=False, allow_unused=True)
        flat_parts = []
        for grad_val, shape, numel in zip(g, param_shapes, param_numels):
            if grad_val is not None:
                flat_parts.append(grad_val.view(-1))
            else:
                flat_parts.append(torch.zeros(numel, device=Config.device))
        flat_grad = torch.cat(flat_parts)
        grads.append(flat_grad)

    g_total_flat = bidirectional_gradient_surgery(grads, gamma)

    start = 0
    for param in net.parameters():
        numel = param.numel()
        param.grad = g_total_flat[start:start + numel].view(param.shape)
        start += numel


def apply_pcgrad(losses, net):
    param_shapes = [p.shape for p in net.parameters()]
    param_numels = [p.numel() for p in net.parameters()]

    grads = []
    for loss in losses:
        g = torch.autograd.grad(loss, net.parameters(), retain_graph=True,
                                create_graph=False, allow_unused=True)

        flat_parts = []
        for grad_val, shape, numel in zip(g, param_shapes, param_numels):
            if grad_val is not None:
                flat_parts.append(grad_val.view(-1))
            else:
                flat_parts.append(torch.zeros(numel, device=Config.device))
        flat_grad = torch.cat(flat_parts)
        grads.append(flat_grad)

    K = len(grads)
    for i in range(K):
        for j in range(K):
            if i != j and cosine_similarity(grads[i], grads[j]) < 0:
                grads[i] = angle_projection(grads[i], grads[j])

    g_total = sum(grads)

    start = 0
    for param in net.parameters():
        numel = param.numel()
        param.grad = g_total[start:start + numel].view(param.shape)
        start += numel

def get_flat_params(net):
    params_list = []
    for p in net.parameters():
        params_list.append(p.data.detach().cpu().numpy().ravel())
    return np.concatenate(params_list)


def set_flat_params(net, flat):
    start = 0
    for p in net.parameters():
        numel = p.numel()
        p.data = torch.tensor(flat[start:start+numel].reshape(p.shape),
                              dtype=torch.float32, device=Config.device)
        start += numel


def evaluate_fitness(net, flat, X_f, X_bc, X_ic, forcing_cache):
    set_flat_params(net, flat)

    with torch.enable_grad():
        X_f_local = X_f.clone().detach().requires_grad_(True)
        pred = net(X_f_local)
        u, v, w, h = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]

        def d(f, idx):
            return grad(f, X_f_local, torch.ones_like(f), create_graph=False, retain_graph=True)[0][:, idx:idx+1]

        u_t, u_x, u_y, u_p = d(u, 3), d(u, 0), d(u, 1), d(u, 2)
        v_t, v_x, v_y, v_p = d(v, 3), d(v, 0), d(v, 1), d(v, 2)
        w_p = d(w, 2)
        h_x, h_y = d(h, 0), d(h, 1)

        yc = X_f_local[:, 1:2]
        cor = Config.f0 + Config.beta_phys * yc * Config.x_0

        x0 = Config.x_0; t0 = Config.t_0; p0 = Config.p_0
        u0 = Config.u_0; h0 = Config.h_0

        Ru = (x0 / u0 / t0) * u_t + u * u_x + v * u_y + (x0 / u0 / t0) * w * u_p \
             - (x0 / u0) * cor * v + (h0 / (u0**2)) * h_x
        Rv = (x0 / u0 / t0) * v_t + u * v_x + v * v_y + (x0 / u0 / t0) * w * v_p \
             + (x0 / u0) * cor * u + (h0 / (u0**2)) * h_y
        Rc = u_x + v_y + (x0 / p0) * w_p

        Fu, Fv, Fc = forcing_cache
        Ru -= Fu; Rv -= Fv; Rc -= Fc

        loss_Ru = torch.mean(Ru**2).item()
        loss_Rv = torch.mean(Rv**2).item()
        loss_Rc = torch.mean(Rc**2).item()

    with torch.no_grad():
        pred_bc = net(X_bc)
        exact_bc = torch.cat(exact_solution(X_bc), dim=1)
        loss_bc = torch.mean((pred_bc - exact_bc)**2).item()

        pred_ic = net(X_ic)
        exact_ic = torch.cat(exact_solution(X_ic), dim=1)
        loss_ic = torch.mean((pred_ic - exact_ic)**2).item()

    return np.array([loss_Ru, loss_Rv, loss_Rc, loss_bc, loss_ic], dtype=np.float64)


def non_dominated_sort(fitness):
    N = fitness.shape[0]
    dominated_count = np.zeros(N, dtype=int)
    dominates_list = [[] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
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


def nsga2_pareto_search(net, X_f, X_bc, X_ic, forcing_cache):
    pop_size = Config.moo_pop_size
    n_gen = Config.moo_n_gen
    bounds = Config.moo_param_bounds
    n_vars = sum(p.numel() for p in net.parameters())

    population = []
    current_flat = get_flat_params(net)
    noise_scale = 0.05 * bounds
    for _ in range(pop_size // 2):
        noisy = current_flat + np.random.randn(n_vars) * noise_scale
        noisy = np.clip(noisy, -bounds, bounds)
        population.append(noisy)
    for _ in range(pop_size - pop_size // 2):
        random_ind = np.random.uniform(-bounds, bounds, n_vars)
        population.append(random_ind)
    population = np.array(population, dtype=np.float64)

    fitness = np.array([evaluate_fitness(net, ind, X_f, X_bc, X_ic, forcing_cache)
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

        off_fitness = np.array([evaluate_fitness(net, ind, X_f, X_bc, X_ic, forcing_cache)
                                for ind in offspring])

        merged_pop = np.vstack([population, offspring])
        merged_fitness = np.vstack([fitness, off_fitness])
        merged_fronts = non_dominated_sort(merged_fitness)
        merged_crowd = [crowding_distance(merged_fitness, f) for f in merged_fronts]

        new_pop, new_fitness = [], []
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

    set_flat_params(net, population[0])
    final_fronts = non_dominated_sort(fitness)
    pareto_fitness = np.array([fitness[i] for i in final_fronts[0]])
    return pareto_fitness


def compute_vari_weights(pareto_fitness, loss_history_buffer, num_losses):
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

    exp_scores = np.exp(np.clip(s_j, -50, 50))
    lambda_j = M * exp_scores / np.sum(exp_scores)

    return torch.tensor(lambda_j, dtype=torch.float32, device=Config.device)


def moo_vari_update(net, X_f, X_bc, X_ic, forcing_cache,
                    loss_history_buffer, num_losses):

    pareto_fitness = nsga2_pareto_search(net, X_f, X_bc, X_ic, forcing_cache)
    weights = compute_vari_weights(pareto_fitness, loss_history_buffer, num_losses)
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
    grad_norms = torch.stack(grad_norms)

    grad_norm_average = grad_norms.mean().detach()

    if alpha > 0 and initial_losses is not None:
        with torch.no_grad():
            loss_ratio = torch.stack([l.detach() for l in losses]) / (initial_losses + 1e-12)
            relative_training_rate = F.normalize(loss_ratio, p=1, dim=0) * num_losses
            gradient_target = (grad_norm_average * (relative_training_rate ** -alpha)).detach()
    else:
        gradient_target = grad_norm_average.expand(num_losses).detach()

    grad_norm_loss = F.l1_loss(grad_norms, gradient_target)
    loss_weights_grad = torch.autograd.grad(grad_norm_loss, loss_weights)[0]

    updated_loss_weights = loss_weights.detach() - loss_weights_grad * lr
    updated_loss_weights = torch.clamp(updated_loss_weights, min=1e-8)
    renormalized_loss_weights = F.normalize(updated_loss_weights, p=1, dim=0) * init_loss_weights_for_sum

    return renormalized_loss_weights.detach()

def main(strategy="standard", seed=0, timestamp=""):
    net = PINN(Config.layers).to(Config.device)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    opt_tag = f"A{Config.adam_epochs}_L{Config.lbfgs_max_iter}"
    if timestamp:
        current_save_dir = os.path.join(current_dir, "results", f"ablation_{strategy}_{opt_tag}_{timestamp}", f"seed_{seed}")
    else:
        current_save_dir = os.path.join(current_dir, "results", f"ablation_{strategy}_{opt_tag}", f"seed_{seed}")

    os.makedirs(current_save_dir, exist_ok=True)

    # 采样
    X_f = sample_interior(Config.N_f)     
    X_bc = sample_boundary(Config.N_bc)   
    X_ic = sample_initial(Config.N_ic)    
    X_test = sample_test_4d(5000, Config.device)

    print("Precomputing forcing terms...")
    Fu, Fv, Fc = forcing_terms(X_f)
    forcing_cache = (Fu, Fv, Fc)


    print("\n" + "=" * 60)
    print("Stage 1: Adam Training")
    print("=" * 60)

    optimizer = Adam(net.parameters(), lr=Config.lr)
    start = time.time()

    history_iter = []
    history_loss = []
    history_l2 = []

    iter_to_1e3 = -1
    current_iter = 0

    num_losses = 5
    weights = torch.ones(num_losses, device=Config.device) / num_losses

    loss_history_buffer = []

    gn_initial_losses = None

    def update_weights_smooth(prev_weights, optimal_weights, beta=0.9):
        return beta * prev_weights + (1 - beta) * optimal_weights

    for epoch in range(Config.adam_epochs):
        Ru, Rv, Rc = pde_residuals(net, X_f, lambda X: forcing_cache)
        loss_Ru = torch.mean(Ru ** 2)
        loss_Rv = torch.mean(Rv ** 2)
        loss_Rc = torch.mean(Rc ** 2)
        loss_bc = boundary_loss(net, X_bc)
        loss_ic = boundary_loss(net, X_ic)

        losses = [loss_Ru, loss_Rv, loss_Rc, loss_bc, loss_ic]

        optimizer.zero_grad()

        if strategy == "standard":
            loss = sum(losses)
            loss.backward()
            optimizer.step()
        elif strategy == "bi_gs":
            params = list(net.parameters())

            if Config.USE_OAW_WEIGHTS:
                grad_norms = []
                for l in losses:
                    grad_l = torch.autograd.grad(l, params, retain_graph=True,
                                                  create_graph=False, allow_unused=True)
                    norm = sum(p.norm().item()**2 for p in grad_l if p is not None)**0.5
                    grad_norms.append(norm)
                grad_norms = torch.tensor(grad_norms, device=Config.device)

                if epoch % Config.weight_update_freq == 0:
                    inv_sq = 1.0 / (grad_norms**2 + 1e-8)
                    opt_weights = inv_sq / inv_sq.sum()
                    weights = update_weights_smooth(weights, opt_weights, Config.OAW_BETA)

            apply_bigspinn_surgery(losses, weights, net, Config.gamma)
            optimizer.step()
            loss = sum(w * l for w, l in zip(weights, losses)).detach()
        elif strategy == "pcgrad":
            apply_pcgrad(losses, net)
            optimizer.step()
            loss = sum(losses).detach()
        elif strategy == "moo_vari":
            if epoch > 0 and epoch % Config.moo_freq == 0:
                weights = moo_vari_update(net, X_f, X_bc, X_ic, forcing_cache,
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
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if strategy == "moo_vari":
            loss_vals = [l.detach().item() for l in losses]
            loss_history_buffer.append(loss_vals)
            if len(loss_history_buffer) > Config.moo_alpha:
                loss_history_buffer.pop(0)

        if current_iter % Config.eval_step == 0:
            l2_u, l2_v, l2_w, l2_h = evaluate_l2_4d(net, X_test, Config.device)
            l2_avg = (l2_u + l2_v + l2_w + l2_h) / 4.0

            history_iter.append(current_iter)
            history_loss.append(loss.item())
            history_l2.append(l2_avg)

            if l2_avg < 1e-3 and iter_to_1e3 == -1:
                iter_to_1e3 = current_iter

            if current_iter % Config.print_step == 0:
                print(f"Adam Epoch {epoch:5d} | Loss={loss.item():.2e} | L2_avg={l2_avg:.3e} | "
                      f"u={l2_u:.3e} v={l2_v:.3e} w={l2_w:.3e} h={l2_h:.3e}")

        current_iter += 1

    adam_time = time.time() - start
    print(f"\nAdam finished. Time: {adam_time:.2f} sec")

    if Config.lbfgs_max_iter > 0:
        print("\n" + "=" * 60)
        print("Stage 2: L-BFGS Fine-tuning")
        print("=" * 60)

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

            Ru, Rv, Rc = pde_residuals(net, X_f, lambda X: forcing_cache)
            loss_Ru = torch.mean(Ru ** 2)
            loss_Rv = torch.mean(Rv ** 2)
            loss_Rc = torch.mean(Rc ** 2)
            loss_bc = boundary_loss(net, X_bc)
            loss_ic = boundary_loss(net, X_ic)

            losses_lbfgs = [loss_Ru, loss_Rv, loss_Rc, loss_bc, loss_ic]

            loss = sum(w * l for w, l in zip(final_weights, losses_lbfgs))
            loss.backward()

            lbfgs_iter_count[0] += 1
            if lbfgs_iter_count[0] % 500 == 0 or lbfgs_iter_count[0] == 1:
                with torch.no_grad():
                    l2_u, l2_v, l2_w, l2_h = evaluate_l2_4d(net, X_test, Config.device)
                    l2_avg = (l2_u + l2_v + l2_w + l2_h) / 4.0
                lbfgs_loss_history.append(loss.item())
                lbfgs_l2_history.append(l2_avg)
                print(f"L-BFGS Iter {lbfgs_iter_count[0]:5d} | Loss={loss.item():.2e} | L2_avg={l2_avg:.3e} | "
                      f"u={l2_u:.3e} v={l2_v:.3e} w={l2_w:.3e} h={l2_h:.3e}")

            return loss

        net.train()
        lbfgs_start = time.time()
        optimizer_lbfgs.step(closure)
        lbfgs_time = time.time() - lbfgs_start

        l2_u, l2_v, l2_w, l2_h = evaluate_l2_4d(net, X_test, Config.device)
        l2_lbfgs_final = (l2_u + l2_v + l2_w + l2_h) / 4.0
        print(f"\nL-BFGS finished. Time: {lbfgs_time:.2f} sec")
        print(f"L-BFGS Final L2_avg: {l2_lbfgs_final:.3e}  (total L-BFGS iters: {lbfgs_iter_count[0]})")
        print(f"  L2_u={l2_u:.3e} L2_v={l2_v:.3e} L2_w={l2_w:.3e} L2_h={l2_h:.3e}")

        adam_final_iter = history_iter[-1] if history_iter else 0
        for i, (loss_val, l2_val) in enumerate(zip(lbfgs_loss_history, lbfgs_l2_history)):
            history_iter.append(adam_final_iter + (i + 1) * 500)
            history_loss.append(loss_val)
            history_l2.append(l2_val)

    print(f"\nTraining Complete! Total Time: {time.time()-start:.2f} sec")

    l2_u, l2_v, l2_w, l2_h = evaluate_l2_4d(net, X_test, Config.device)
    l2_avg_final = (l2_u + l2_v + l2_w + l2_h) / 4.0

    if Config.DEBUG_MODE:
        print("\n[WARNING] DEBUG MODE ENABLED -- Results are NOT for paper!")

    print("\n" + "=" * 60)
    print("    [FINAL EXPERIMENTAL METRICS (For Paper)]")
    print("=" * 60)
    val_10_3 = l2_avg_final * 1000
    print("[Table 1: Predictive Accuracy]")
    print(f"  --> {strategy.upper()}-PINN L2 Error (avg u,v,w,h): {val_10_3:.2f} (x 10^-3)")
    print(f"      L2_u={l2_u:.3e}, L2_v={l2_v:.3e}, L2_w={l2_w:.3e}, L2_h={l2_h:.3e}")

    print("\n[Table 2: Convergence Behavior]")
    if iter_to_1e3 != -1:
        print(f"  --> Iterations to 1e-3: {iter_to_1e3}")
    else:
        print(f"  --> Iterations to 1e-3: Did not reach 1e-3. Final L2_avg={l2_avg_final:.3e}")
    print("=" * 60)


    print("\nGenerating Plots...")
    plot_name = {"standard": "PINN (Uniform)", "pcgrad": "PCGrad", "bi_gs": "Bi-GS-PINN", "moo_vari": "MOO-VARI-PINN", "gradnorm": "GradNorm"}[strategy]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history_iter, history_loss, 'b-', label='Total Loss', linewidth=1.5)
    if Config.lbfgs_max_iter > 0:
        axes[0].axvline(x=Config.adam_epochs, color='r', linestyle='--', alpha=0.7, label='Adam -> L-BFGS')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training Loss Curve ({plot_name})')
    axes[0].grid(True, which="both", ls=":", alpha=0.5)
    axes[0].legend()

    axes[1].plot(history_iter, history_l2, 'g-', label='Relative L2 Error (avg)', linewidth=1.5)
    axes[1].axhline(y=1e-3, color='k', linestyle='-', label='Tolerance ($10^{-3}$)')
    if Config.lbfgs_max_iter > 0:
        axes[1].axvline(x=Config.adam_epochs, color='r', linestyle='--', alpha=0.7, label='Adam -> L-BFGS')
    if iter_to_1e3 != -1:
        axes[1].scatter([iter_to_1e3], [1e-3], color='red', zorder=5)
        axes[1].text(iter_to_1e3, 1.5e-3, f' {iter_to_1e3} iters', color='red', fontsize=10)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Relative L2 Error')
    axes[1].set_title(f'Validation L2 Error Curve ({plot_name})')
    axes[1].grid(True, which="both", ls=":", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(current_save_dir, "Convergence_Curve.png"), dpi=300)
    plt.close()

    # 保存配置
    with open(os.path.join(current_save_dir, "config.txt"), "w") as f:
        f.write(f"equation = 4D MMS Atmospheric System\n")
        f.write(f"strategy = {strategy}\n")
        f.write(f"seed = {seed}\n")
        f.write(f"N_f = {Config.N_f}\n")
        f.write(f"N_bc = {Config.N_bc}\n")
        f.write(f"N_ic = {Config.N_ic}\n")
        f.write(f"adam_epochs = {Config.adam_epochs}\n")
        f.write(f"lr = {Config.lr}\n")
        f.write(f"layers = {Config.layers}\n")
        if Config.lbfgs_max_iter > 0:
            f.write(f"lbfgs_max_iter = {Config.lbfgs_max_iter}\n")
            f.write(f"lbfgs_lr = {Config.lbfgs_lr}\n")
            f.write(f"lbfgs_history_size = {Config.lbfgs_history_size}\n")
        if strategy == "bi_gs":
            f.write(f"gamma = {Config.gamma}\n")
            f.write(f"weight_update_freq = {Config.weight_update_freq}\n")
            f.write(f"oaw_beta = {Config.oaw_beta}\n")
        f.write(f"\n--- Final L2 Errors ---\n")
        f.write(f"L2_u = {l2_u:.6e}\n")
        f.write(f"L2_v = {l2_v:.6e}\n")
        f.write(f"L2_w = {l2_w:.6e}\n")
        f.write(f"L2_h = {l2_h:.6e}\n")
        f.write(f"L2_avg = {l2_avg_final:.6e}\n")

    print(f"Plots saved to {current_save_dir}")
    torch.save(net.state_dict(), os.path.join(current_save_dir, "model_final.pth"))

    return l2_avg_final, iter_to_1e3

if __name__ == "__main__":
    ablation_map = {
        "full":       ("Full Bi-GS-PINN",             True,  True,  True,  0.9),
        "no_angle":   ("-- Angle projection",         False, True,  True,  0.9),
        "no_mag":     ("-- Magnitude equalization",   True,  False, True,  0.9),
        "no_oaw":     ("-- Closed-form weights",      True,  True,  False, 0.9),
        "no_smooth":  ("-- Smoothing (beta=0)",       True,  True,  True,  0.0),
    }

    if Config.RUN_STRATEGY == "all_ablation":
        variants_to_run = list(ablation_map.keys())
    else:
        variants_to_run = [Config.RUN_STRATEGY]

    num_runs = 2 if Config.DEBUG_MODE else 5
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    for variant_key in variants_to_run:
        name, use_angle, use_mag, use_oaw, beta = ablation_map[variant_key]
        Config.USE_ANGLE_PROJECTION = use_angle
        Config.USE_MAGNITUDE_EQUALIZATION = use_mag
        Config.USE_OAW_WEIGHTS = use_oaw
        Config.OAW_BETA = beta

        print(f"\n{'='*60}")
        print(f"=== ABLATION: {name} ===")
        print(f"{'='*60}")

        l2_list = []
        for seed in range(max(5, num_runs) if not Config.DEBUG_MODE else num_runs):
            print(f"  Run {seed}...")
            set_seed(seed)
            l2_final, _ = main(strategy="bi_gs", seed=seed, timestamp=run_timestamp)
            l2_list.append(l2_final)

        results[name] = {
            "l2_mean": np.mean(l2_list) * 1000,
            "l2_std": np.std(l2_list) * 1000,
        }

    # 打印消融表格
    print("\n" + "=" * 60)
    print("=== ABLATION STUDY RESULTS (Table format) ===")
    print("=" * 60)
    print(f"{'Variant':<30} | {'L2 error (x 10^-3)'}")
    print("-" * 52)
    for variant_key in variants_to_run:
        name, _, _, _, _ = ablation_map[variant_key]
        res = results[name]
        print(f"{name:<30} | {res['l2_mean']:.1f} +- {res['l2_std']:.1f}")
    print("=" * 60)
