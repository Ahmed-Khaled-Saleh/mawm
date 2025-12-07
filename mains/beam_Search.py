import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MAWM.core import Program, PRIMITIVE_TEMPLATES
from MAWM.models.program_embedder import batchify_programs
from MAWM.models.program_encoder import ProgramEncoder
from MAWM.models.program_synthizer import Proposer

@torch.no_grad()
def neural_guided_beam_search2(
    z,
    program_embedder: nn.Module,
    proposer: nn.Module,
    program_encoder: nn.Module,
    score_fn,
    MAX_PARAMS=3,
    beam_width=5,
    topk=6,
    max_prog_len=5,
    grid_size=7,
    device="cuda"
):
    """
    Corrected neural-guided beam search.
    Keeps your parameter logic untouched.
    """

    # --- normalize z shape ---
    z = z.to(device)
    if z.dim() == 1:
        z = z.unsqueeze(0)

    sos_idx = proposer.sos_idx
    zero_params = torch.full((MAX_PARAMS,), -1, device=device)

    # initial program: [SOS]
    init_prog = Program(tokens=[(sos_idx, zero_params.tolist())])
    beam = [(init_prog, 0.0)]  # (program, score)

    best_program = init_prog
    best_score = -float("inf")

    # ---- start beam search ----
    for depth in range(1, max_prog_len + 1):

        # ---- 1) build batch of prefixes ----
        prefix_programs = [p for (p, _) in beam]
        prev_idx_batch, prev_params_batch = batchify_programs(prefix_programs)

        B = prev_idx_batch.shape[0]

        p_vec = program_embedder(prev_idx_batch.to(device),
                                 prev_params_batch.to(device))

        # replicate z
        z_batch = z.repeat(B, 1)

        # proposer forward
        def get_proposer(seq_len):
            return Proposer(
                    obs_dim= z.shape[-1],
                    num_prims= len(PRIMITIVE_TEMPLATES),
                    max_params= 2,
                    seq_len= seq_len,
                    prog_emb_dim_x= 32,
                    prog_emb_dim_y= 32,
                    prog_emb_dim_prims= 32,
                )

        proposer = get_proposer(prev_idx_batch.shape[-1])
        prim_logits_batch, param_pred_batch = proposer.forward_step(z_batch, p_vec)

        # ---- 2) expand each beam entry ----
        expansions = []
        for b_i, (prog_parent, parent_score) in enumerate(beam):
            prim_logprobs = F.log_softmax(prim_logits_batch[b_i], dim=-1)
            top_vals, top_idx = torch.topk(prim_logprobs, k=topk)

            for k_i in range(topk):

                prim_idx = int(top_idx[k_i].item())
                prim_logp = float(top_vals[k_i].item())
                arity = PRIMITIVE_TEMPLATES[prim_idx][1]

                # ----- parameter instantiation -----
                instantiations = []
                if arity > 0:
                    pred_params = param_pred_batch[b_i][:arity].cpu().numpy()
                    for _ in range(3):
                        if np.random.rand() < 0.7:
                            inst = [
                                float(round(float(p) * (grid_size - 1)))
                                for p in pred_params
                            ]
                        else:
                            inst = [
                                float(np.random.randint(0, grid_size))
                                for _ in range(arity)
                            ]
                        instantiations.append(inst)
                else:
                    instantiations.append([])

                # ----- create new beam children -----
                for inst_params in instantiations:
                    new_prog = prog_parent.extend(prim_idx, inst_params)
                    expansions.append((new_prog, parent_score + prim_logp))

        if not expansions:
            break

        # ---- 3) Score all expanded programs ----
        cand_programs = [p for (p, _) in expansions]
        print(cand_programs)
        print(len(cand_programs))
        print(cand_programs[0])
        prim_ids_list, param_list, Llist = [], [], []

        for prog in cand_programs:
            prim_ids, param_t = batchify_programs([prog])
            prim_ids_list.append(prim_ids)
            param_list.append(param_t)
            Llist.append(prim_ids.shape[1])

        # Padding
        Lmax = max(Llist)
        Nc = len(cand_programs)
        # print(Lmax, Nc)
        prim_ids_padded = torch.full((Nc, Lmax), -1, device=device, dtype=torch.long)
        params_padded = torch.full((Nc, Lmax, MAX_PARAMS), -1, device=device, dtype=torch.long)
        # print(prim_ids_padded.shape, params_padded.shape)
        for i_p, (pids, pt) in enumerate(zip(prim_ids_list, param_list)):
            Li = pids.shape[1]
            prim_ids_padded[i_p, :Li] = pids.squeeze(0)
            params_padded[i_p, :Li] = pt.squeeze(0)

        print("After padding:")
        # print(prim_ids_padded.shape, params_padded.shape)

        def get_pencoder(seq_len):
            return ProgramEncoder(num_primitives= len(PRIMITIVE_TEMPLATES),
                    param_cardinalities= [grid_size, grid_size],
                    seq_len= seq_len,
                    max_params_per_primitive= 2)
        
        with torch.no_grad():
            # program_encoder = get_pencoder(prim_ids_padded.shape[-1])
            print(prim_ids_list, len(prim_ids_list), prim_ids_list[0].dtype, prim_ids_list[0].shape) # 10, 64, [1, 2]
            print(torch.stack(prim_ids_list).shape)
            # print(torch.tensor(prim_ids_list, dtype=torch.long).shape)
            program_encoder = get_pencoder(torch.tensor(prim_ids_list, dtype=torch.long).shape[-1])
            #prog_emb_batch = program_encoder(prim_ids_padded, params_padded)
            prog_emb_batch = program_encoder(torch.tensor(prim_ids_list, dtype=torch.long), torch.tensor(param_list, dtype=torch.long))

        scores = score_fn(z, prog_emb_batch)  # shape (Nc,1) or (Nc,)

        # ---- 4) attach scores and prune ----
        candidates = []
        for idx_c, (prog, base_score) in enumerate(expansions):
            total = base_score + float(scores[idx_c].item())
            candidates.append((prog, total))

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_program, best_score = max(candidates, key=lambda x: x[1])
        beam = candidates[:beam_width]
        

    return best_program, best_score
