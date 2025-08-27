from typing import Dict

def fuse(rule_flag: int, ml_prob: float, tau_rule: float = 0.5, tau_ml: float = 0.5) -> int:
    # Treat rule_flag as boolean 0/1; ml_prob in [0,1]
    return int(bool(rule_flag) or (ml_prob >= tau_ml))

def fuse_multi(rule_flags: Dict[str,int], ml_probs: Dict[str,float], tau_ml: float = 0.5):
    out = {}
    for k, rule in rule_flags.items():
        out[k] = int(bool(rule) or (ml_probs.get(k, 0.0) >= tau_ml))
    return out
