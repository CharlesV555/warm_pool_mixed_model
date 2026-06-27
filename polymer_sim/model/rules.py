from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polymer_sim.model.species import SpeciesSpace


@dataclass(slots=True)
class ReactionRuleTables:
    left_join: np.ndarray
    right_join: np.ndarray
    split_left_monomer: np.ndarray
    split_left_rest: np.ndarray
    split_right_rest: np.ndarray
    split_right_monomer: np.ndarray
    can_split: np.ndarray
    left_char_idx: np.ndarray
    right_char_idx: np.ndarray


def build_reaction_rule_tables(space: SpeciesSpace) -> ReactionRuleTables:
    n_monomers = space.n_monomers
    n_species = space.n_species

    left_join = np.full((n_monomers, n_species), -1, dtype=np.int64) # 将所有反应物种与单体连接的结果初始化为-1，表示无效连接
    right_join = np.full((n_species, n_monomers), -1, dtype=np.int64)

    for m in range(n_monomers):
        m_name = space.species_names[m]
        for sid, s_name in enumerate(space.species_names): # 遍历所有物种，尝试将单体m与物种name连接（字符拼接）
            if space.lengths[sid] + 1 > space.max_len:
                continue
            left_join[m, sid] = space.name_to_idx[m_name + s_name] # 对于能发生反应的物质，记录连接后的物种ID
            right_join[sid, m] = space.name_to_idx[s_name + m_name]

    split_left_monomer = np.full(n_species, -1, dtype=np.int64)
    split_left_rest = np.full(n_species, -1, dtype=np.int64)
    split_right_rest = np.full(n_species, -1, dtype=np.int64)
    split_right_monomer = np.full(n_species, -1, dtype=np.int64)
    left_char_idx = np.full(n_species, -1, dtype=np.int64)
    right_char_idx = np.full(n_species, -1, dtype=np.int64)
    can_split = space.lengths > 1 # 只有长度大于1的物种才可以发生分解反应，这里是一个mask

    for sid, name in enumerate(space.species_names):
        left_char_idx[sid] = space.name_to_idx[name[0]] # 记录加聚的反应物中多聚物的id（即name的第一个字符对应的单体id）
        right_char_idx[sid] = space.name_to_idx[name[-1]] # -1代表name的最后一个字符对应的单体id，如果能裂解，后面有用↓
        if not can_split[sid]:
            continue
        split_left_monomer[sid] = space.name_to_idx[name[0]] # 如果能裂解，左侧裂解下来的是什么单体
        split_left_rest[sid] = space.name_to_idx[name[1:]] # 如果能裂解，左侧裂解下来的其余部分是什么
        split_right_rest[sid] = space.name_to_idx[name[:-1]] # 如果能裂解，右侧裂解下来的其余部分是什么
        split_right_monomer[sid] = space.name_to_idx[name[-1]] # 如果能裂解，右侧裂解下来的单体是什么

    return ReactionRuleTables(
        left_join=left_join,
        right_join=right_join,
        split_left_monomer=split_left_monomer,
        split_left_rest=split_left_rest,
        split_right_rest=split_right_rest,
        split_right_monomer=split_right_monomer,
        can_split=can_split,
        left_char_idx=left_char_idx,
        right_char_idx=right_char_idx,
    )
