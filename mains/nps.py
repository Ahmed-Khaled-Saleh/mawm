# neural_program_search_full.py
# Full neural-guided program search with parametrized primitives, padding, and training loop.
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Tuple

# -------------------------
# Config / hyperparameters
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Z_DIM = 64
ACTION_DIM = 8
PROG_HID = 128
PRIM_EMB = 32
PARAM_EMB = 16
PROG_RNN_HID = 128
MSG_DIM = 48
MAX_PARAMS = 2        # maximum params per primitive; everything is padded to this
GRID_SIZE = 5
BEAM_WIDTH = 5
PROP_TOPK = 6
MAX_PROG_LEN = 5
LAMBDA_Z = 1.0
LAMBDA_R = 1.0
LEARNING_RATE = 1e-4

# -------------------------
# Primitive templates (name, arity)
# -------------------------
PRIMITIVE_TEMPLATES = [
    ("AgentAt", 2),
    ("GoalAt", 2),
    ("ObstacleAt", 2),
    ("ItemAt", 2),
    ("Near", 0),       # boolean style (no params)
    ("CanMove", 1),    # direction (0..3)
]
PRIM_NAME_TO_IDX = {name: i for i, (name, ar) in enumerate(PRIMITIVE_TEMPLATES)}
NUM_PRIMS = len(PRIMITIVE_TEMPLATES)
print(PRIM_NAME_TO_IDX)
# -------------------------
# Program representation
# -------------------------
class Program:
    def __init__(self, tokens: List[Tuple[int, List[float]]] = None):
        # tokens: list of (prim_idx, params_list)
        self.tokens = tokens or []

    def extend(self, prim_idx: int, params: List[float]):
        return Program(self.tokens + [(int(prim_idx), [float(p) for p in params])])

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        if len(self.tokens) == 0:
            return "<EMPTY>"
        toks = []
        for pidx, params in self.tokens:
            name = PRIMITIVE_TEMPLATES[pidx][0]
            toks.append(f"{name}{tuple(params)}")
        return " | ".join(toks)

# -------------------------
# encode_program_for_network: guaranteed shapes
# prim_ids: (1, L), param_tensor: (1, L, MAX_PARAMS)
# -------------------------
def encode_program_for_network(program: Program, max_params: int = MAX_PARAMS, device=DEVICE):
    L = len(program)
    if L == 0:
        prim_ids = torch.zeros((1, 0), dtype=torch.long, device=device)                # (1,0)
        param_tensor = torch.zeros((1, 0, max_params), dtype=torch.float32, device=device)  # (1,0,maxp)
        return prim_ids, param_tensor

    prim_ids_list = []
    params_list = []
    for (prim_idx, params) in program.tokens:
        prim_ids_list.append(int(prim_idx))
        p = list(params)[:max_params]
        if len(p) < max_params:
            p = p + [0.0] * (max_params - len(p))
        params_list.append(p)
    prim_ids = torch.tensor([prim_ids_list], dtype=torch.long, device=device)           # (1, L)
    param_tensor = torch.tensor([params_list], dtype=torch.float32, device=device)      # (1, L, max_params)
    return prim_ids, param_tensor

# -------------------------
# ProgramEncoder (Enc_ψ)
# -------------------------
class ProgramEncoder(nn.Module):
    def __init__(self, num_prims=NUM_PRIMS, prim_emb_dim=PRIM_EMB, param_emb_dim=PARAM_EMB,
                 rnn_hid=PROG_RNN_HID, out_dim=MSG_DIM, max_params=MAX_PARAMS):
        super().__init__()
        self.prim_emb = nn.Embedding(num_prims, prim_emb_dim)
        self.param_proj = nn.Linear(max_params, param_emb_dim)
        self.rnn = nn.GRU(prim_emb_dim + param_emb_dim, rnn_hid, batch_first=True)
        self.out_proj = nn.Linear(rnn_hid, out_dim)
        self.max_params = max_params

    def forward(self, prim_ids: torch.LongTensor, param_tensor: torch.FloatTensor):
        # prim_ids: (B, L), param_tensor: (B, L, max_params)
        B, L = prim_ids.shape
        prim_e = self.prim_emb(prim_ids)                         # (B, L, prim_emb)
        param_flat = param_tensor.view(B * L, -1)                # (B*L, max_params)
        param_e = F.relu(self.param_proj(param_flat))           # (B*L, param_emb)
        param_e = param_e.view(B, L, -1)                        # (B, L, param_emb)
        x = torch.cat([prim_e, param_e], dim=-1)                # (B, L, prim+param)
        if L == 0:
            return torch.zeros((B, self.out_proj.out_features), device=prim_ids.device, dtype=torch.float32)
        _, h = self.rnn(x)                                      # h: (1, B, rnn_hid)
        h = h.squeeze(0)                                        # (B, rnn_hid)
        out = self.out_proj(h)                                  # (B, out_dim)
        return out

# -------------------------
# Proposer Qφ
# Predicts next-primitive logits and continuous params in [0,1]
# Accepts fixed-length prev_params tensor of shape (B, MAX_PARAMS)
# -------------------------
class Proposer(nn.Module):
    def __init__(self, obs_dim=Z_DIM, prim_emb_dim=PRIM_EMB, param_emb_dim=PARAM_EMB, hidden=PROG_HID,
                 num_prims=NUM_PRIMS, max_params=MAX_PARAMS):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.prev_prim_emb = nn.Embedding(num_prims + 1, prim_emb_dim)  # +1 for SOS
        self.prev_param_proj = nn.Linear(max_params, param_emb_dim)
        self.comb = nn.Linear(hidden + prim_emb_dim + param_emb_dim, hidden)
        self.next_prim = nn.Linear(hidden, num_prims)
        self.next_param = nn.Linear(hidden, max_params)
        self.sos_idx = num_prims  # index reserved for SOS

    def forward_step(self, z_obs: torch.FloatTensor, prev_prim_idx: torch.LongTensor, prev_params: torch.FloatTensor):
        """
        z_obs: (B, obs_dim)
        prev_prim_idx: (B,) long
        prev_params: (B, MAX_PARAMS) float
        returns:
            prim_logits: (B, num_prims)
            param_pred: (B, MAX_PARAMS) in (0,1)
        """
        h_obs = F.relu(self.obs_proj(z_obs))
        prev_prim_emb = self.prev_prim_emb(prev_prim_idx)
        prev_param_e = F.relu(self.prev_param_proj(prev_params))
        cat = torch.cat([h_obs, prev_prim_emb, prev_param_e], dim=-1)
        h = F.relu(self.comb(cat))
        logits = self.next_prim(h)
        param_pred = torch.sigmoid(self.next_param(h))
        return logits, param_pred

# -------------------------
# WorldModel θ
# -------------------------
class WorldModel(nn.Module):
    def __init__(self, z_dim=Z_DIM, action_dim=ACTION_DIM, prog_dim=MSG_DIM, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + action_dim + prog_dim, hidden)
        self.fc_state = nn.Linear(hidden, z_dim)
        self.fc_reward = nn.Linear(hidden, 1)

    def forward(self, z_t: torch.FloatTensor, a_t: torch.FloatTensor, prog_emb: torch.FloatTensor):
        x = torch.cat([z_t, a_t, prog_emb], dim=-1)
        h = F.relu(self.fc1(x))
        z_next = self.fc_state(h)
        r = self.fc_reward(h).squeeze(-1)
        return z_next, r

# -------------------------
# Energy (batch)
# -------------------------
def compute_energy_batch(z_pred_b, z_target_b, r_pred_b, r_target_b, lambda_z=LAMBDA_Z, lambda_r=LAMBDA_R):
    diff = torch.abs(z_pred_b - z_target_b).mean(dim=1)  # (Nc,)
    e_state = lambda_z * diff
    e_reward = lambda_r * ((r_pred_b - r_target_b) ** 2)
    return e_state + e_reward  # (Nc,)

# -------------------------
# Neural-guided beam search (per sample) - fixed param lengths + batching
# -------------------------
def neural_guided_beam_search(z_obs: torch.FloatTensor,
                              z_target: torch.FloatTensor,
                              a_t: torch.FloatTensor,
                              r_target: torch.FloatTensor,
                              proposer: Proposer,
                              encoder: ProgramEncoder,
                              world_model: WorldModel,
                              beam_width: int = BEAM_WIDTH,
                              topk: int = PROP_TOPK,
                              max_prog_len: int = MAX_PROG_LEN,
                              grid_size: int = GRID_SIZE,
                              device=DEVICE):
    """
    z_obs, z_target: 1D tensors (z_dim,)
    a_t: 1D tensor (action_dim,)
    r_target: scalar tensor ()
    """
    z_obs = z_obs.to(device).unsqueeze(0)        # (1, z_dim)
    z_target = z_target.to(device).unsqueeze(0)  # (1, z_dim)
    a_t = a_t.to(device).unsqueeze(0)            # (1, action_dim)
    r_target = r_target.to(device).unsqueeze(0)  # (1,)

    sos_idx = proposer.sos_idx
    zero_params = torch.zeros((MAX_PARAMS,), device=device, dtype=torch.float32)

    # Beam entries: (Program, prev_prim_idx(int), prev_params(tensor MAX_PARAMS), cumulative_score(float))
    beam = [(Program(), sos_idx, zero_params, 0.0)]

    best_program = Program()
    best_energy = float("inf")

    with torch.no_grad():
        for depth in range(1, max_prog_len + 1):
            all_candidates = []
            # Prepare proposer inputs batched across current beam
            Bbeam = len(beam)
            prev_idx_batch = torch.tensor([entry[1] for entry in beam], dtype=torch.long, device=device)  # (Bbeam,)
            prev_params_batch = torch.stack([entry[2] for entry in beam], dim=0)                         # (Bbeam, MAX_PARAMS)
            z_obs_batch = z_obs.repeat(Bbeam, 1)                                                         # (Bbeam, z_dim)

            prim_logits_batch, param_pred_batch = proposer.forward_step(z_obs_batch, prev_idx_batch, prev_params_batch)
            # prim_logits_batch: (Bbeam, num_prims), param_pred_batch: (Bbeam, MAX_PARAMS)

            # For each beam parent, expand top-k primitives
            expansions = []
            for parent_i, (prog_parent, prev_idx, prev_params, parent_score) in enumerate(beam):
                probs = F.softmax(prim_logits_batch[parent_i], dim=-1)  # (num_prims,)
                top_vals, top_idx = torch.topk(probs, k=min(topk, probs.size(0)), dim=-1)
                for k_i in range(top_idx.size(0)):
                    prim_idx = int(top_idx[k_i].item())
                    arity = PRIMITIVE_TEMPLATES[prim_idx][1]
                    # pull param predictions for this parent and convert to instance
                    pred_params = param_pred_batch[parent_i][:arity].cpu().numpy().tolist() if arity > 0 else []
                    # discretize param predictions to grid coordinates or keep floats; here we round to nearest int in grid
                    inst_params = []
                    for pval in pred_params:
                        scaled = float(pval) * (grid_size - 1)
                        inst_params.append(float(round(scaled)))
                    # pad inst_params to MAX_PARAMS later in encoder
                    new_prog = prog_parent.extend(prim_idx, inst_params)
                    expansions.append((parent_i, new_prog, prim_idx, inst_params, parent_score))

            if len(expansions) == 0:
                break

            # Batch-evaluate all expansions with encoder + world_model
            cand_programs = [e[1] for e in expansions]
            # Encode each program, collect prim_ids and param tensors padded to L_max
            prim_ids_list = []
            param_tensors_list = []
            L_list = []
            for prog in cand_programs:
                prim_ids, param_tensor = encode_program_for_network(prog, max_params=MAX_PARAMS, device=device)
                prim_ids_list.append(prim_ids)           # shape (1, L)
                param_tensors_list.append(param_tensor)  # shape (1, L, MAX_PARAMS)
                L_list.append(prim_ids.shape[1])
            L_max = max(L_list) if len(L_list) > 0 else 0
            # Create batched padded tensors
            Nc = len(cand_programs)
            prim_ids_padded = torch.zeros((Nc, L_max), dtype=torch.long, device=device)
            param_padded = torch.zeros((Nc, L_max, MAX_PARAMS), dtype=torch.float32, device=device)
            for i_p, (prim_ids, param_t) in enumerate(zip(prim_ids_list, param_tensors_list)):
                Li = prim_ids.shape[1]
                if Li > 0:
                    prim_ids_padded[i_p, :Li] = prim_ids.squeeze(0)
                    param_padded[i_p, :Li, :] = param_t.squeeze(0)
            # Get program embeddings in batch
            prog_emb_batch = encoder(prim_ids_padded, param_padded)  # (Nc, MSG_DIM)
            # Prepare z and a repeated for batch
            z_b = z_obs.repeat(Nc, 1)
            a_b = a_t.repeat(Nc, 1)
            z_target_b = z_target.repeat(Nc, 1)
            r_target_b = r_target.repeat(Nc)
            # Evaluate world model
            z_pred_b, r_pred_b = world_model(z_b, a_b, prog_emb_batch)
            energies = compute_energy_batch(z_pred_b, z_target_b, r_pred_b, r_target_b)  # (Nc,)

            # Map back energies to expansions and form candidate beam entries
            for idx_e, entry in enumerate(expansions):
                parent_i, new_prog, prim_idx, inst_params, parent_score = entry
                energy = float(energies[idx_e].item())
                new_score = parent_score - energy  # higher is better
                # prepare prev_params tensor for next step: padded to MAX_PARAMS
                prev_params_next = [float(x) for x in inst_params][:MAX_PARAMS]
                if len(prev_params_next) < MAX_PARAMS:
                    prev_params_next = prev_params_next + [0.0] * (MAX_PARAMS - len(prev_params_next))
                prev_params_tensor = torch.tensor(prev_params_next, dtype=torch.float32, device=device)
                all_candidates.append((new_prog, prim_idx, prev_params_tensor, new_score))

            # After collecting all_candidates across parents, prune to beam_width
            if len(all_candidates) == 0:
                break
            all_candidates.sort(key=lambda x: x[3], reverse=True)
            beam = all_candidates[:beam_width]
            # update best
            for prog_cand, pidx, pparams, sc in beam:
                energy_here = -sc
                if energy_here < best_energy:
                    best_energy = energy_here
                    best_program = prog_cand
        # end depth loop

    if best_program is None:
        return Program(), float("inf")
    return best_program, best_energy

# -------------------------
# Training loop (wake-phase demo)
# - For each batch: run search per sample to get P*, then update world_model+encoder (M-step) and proposer (imitation)
# -------------------------
def train_demo(num_iters=20, batch_size=4):
    # Instantiate models
    encoder = ProgramEncoder().to(DEVICE)
    proposer = Proposer().to(DEVICE)
    world_model = WorldModel().to(DEVICE)

    wm_opt = torch.optim.Adam(list(world_model.parameters()) + list(encoder.parameters()), lr=LEARNING_RATE)
    prop_opt = torch.optim.Adam(proposer.parameters(), lr=LEARNING_RATE)

    for it in range(num_iters):
        # Dummy batch: replace with your real CTDE data
        z_sender = torch.randn(batch_size, Z_DIM, device=DEVICE)
        z_receiver_next = z_sender + 0.05 * torch.randn_like(z_sender)
        actions = torch.randn(batch_size, ACTION_DIM, device=DEVICE)
        rewards = torch.zeros(batch_size, device=DEVICE)

        inferred_programs = []
        inferred_msgs = []

        # 1) Wake: neural-guided search per-sample to infer P*
        for b in range(batch_size):
            z_obs = z_sender[b]
            z_target = z_receiver_next[b]
            a_t = actions[b]
            r_t = rewards[b]
            P_star, energy = neural_guided_beam_search(z_obs, z_target, a_t, r_t,
                                                       proposer, encoder, world_model,
                                                       beam_width=BEAM_WIDTH, topk=PROP_TOPK,
                                                       max_prog_len=MAX_PROG_LEN,
                                                       grid_size=GRID_SIZE, device=DEVICE)
            print(f"Iter {it} Sample {b} Inferred Program: {P_star} Energy: {energy:.4f}")
            inferred_programs.append(P_star)
            # get program embedding for training
            prim_ids, param_tensor = encode_program_for_network(P_star, max_params=MAX_PARAMS, device=DEVICE)
            # pad to batch later; for now compute message
            if prim_ids.shape[1] == 0:
                msg = torch.zeros((MSG_DIM,), device=DEVICE)
            else:
                msg = encoder(prim_ids, param_tensor).squeeze(0)  # (MSG_DIM,)
            inferred_msgs.append(msg)

        # 2) M-step: update world model + encoder using inferred messages
        wm_opt.zero_grad()
        msgs_b = torch.stack(inferred_msgs, dim=0)  # (B, MSG_DIM)
        z_pred, r_pred = world_model(z_sender, actions, msgs_b)
        loss_z = F.l1_loss(z_pred, z_receiver_next)
        loss_r = F.mse_loss(r_pred, rewards)
        loss_wm = LAMBDA_Z * loss_z + LAMBDA_R * loss_r
        loss_wm.backward()
        wm_opt.step()

        # 3) Imitation: train proposer to predict P* (teacher forcing)
        # Build training tensors (variable-length sequences -> pad)
        # We train proposer on next-token prediction: for each program we construct sequences of (prev_prim_idx, prev_params) -> (next_prim_idx, next_params)
        prop_opt.zero_grad()
        ce_loss = 0.0
        tot_tokens = 0
        for b, prog in enumerate(inferred_programs):
            L = len(prog)
            if L == 0:
                continue
            # initial prev: SOS
            prev_idx = proposer.sos_idx
            prev_params = torch.zeros((MAX_PARAMS,), device=DEVICE)
            for t in range(L):
                target_prim_idx, target_params = prog.tokens[t]
                z_obs_b = z_sender[b].unsqueeze(0)
                prev_idx_t = torch.tensor([prev_idx], dtype=torch.long, device=DEVICE)
                prev_params_t = prev_params.unsqueeze(0)  # (1, MAX_PARAMS)
                logits, param_pred = proposer.forward_step(z_obs_b, prev_idx_t, prev_params_t)
                # cross-entropy for prim
                target = torch.tensor([target_prim_idx], dtype=torch.long, device=DEVICE)
                ce_loss += F.cross_entropy(logits, target)
                # regression loss for params (if any)
                if PRIMITIVE_TEMPLATES[target_prim_idx][1] > 0:
                    # prepare target param vector padded to MAX_PARAMS
                    tgt = list(target_params)[:MAX_PARAMS]
                    if len(tgt) < MAX_PARAMS:
                        tgt = tgt + [0.0] * (MAX_PARAMS - len(tgt))
                    tgt = torch.tensor([tgt], dtype=torch.float32, device=DEVICE)
                    ce_loss += F.mse_loss(param_pred, tgt)  # MSE on continuous [0,1] space (we earlier discretized, but here we train proposer to match)
                tot_tokens += 1
                # update prev for next step
                prev_idx = int(target_prim_idx)
                nxt_params = list(target_params)[:MAX_PARAMS]
                if len(nxt_params) < MAX_PARAMS:
                    nxt_params = nxt_params + [0.0] * (MAX_PARAMS - len(nxt_params))
                prev_params = torch.tensor(nxt_params, dtype=torch.float32, device=DEVICE)
        if tot_tokens > 0:
            loss_prop = ce_loss / tot_tokens
            loss_prop.backward()
            prop_opt.step()

        if it % 1 == 0:
            print(f"iter {it} wm_loss {loss_wm.item():.4f} prop_loss {(loss_prop.item() if tot_tokens>0 else 0.0):.4f}")

    return encoder, proposer, world_model

# -------------------------
# Run demo
# -------------------------
if __name__ == "__main__":
    encoder, proposer, world_model = train_demo(num_iters=12, batch_size=6)
    print("Demo finished.")
