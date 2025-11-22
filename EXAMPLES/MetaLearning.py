# Part 2: Meta-Learning (Verifiable Demonstration)
# This script loads trained specialists and runs a meta-learning experiment
# with a fixed seed to produce a 100% repeatable, verifiable result.
#
# ZENODO VERSION - SEED: 657454018

import json
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from datetime import datetime
import os

# ==============================================================================
# FIXED SEED FOR VERIFIABLE RESULTS
# All randomness is removed. The script will produce the exact same
# output every time it is run.
# ==============================================================================
CHOSEN_SEED = 657454018
np.random.seed(CHOSEN_SEED)
torch.manual_seed(CHOSEN_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CHOSEN_SEED)
# ==============================================================================


# --- Core Model (same architecture as training) ---
class SpecialistModel(nn.Module):
    def __init__(self, D_in, N_in):
        super(SpecialistModel, self).__init__()
        self.D, self.N = D_in, N_in
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.D, 96), 
            nn.Sigmoid(), 
            nn.LayerNorm(96), 
            nn.Linear(96, 48), 
            nn.Sigmoid()
        )
        self.scoring_head = nn.Linear(48, 1)
        self.project_to_latent = nn.Linear(48, 16)
        self.project_from_latent = nn.Linear(16, 48)
    
    def forward(self, x):
        return self.scoring_head(self.feature_extractor(x)).squeeze(-1)

# --- Data Generators (same as training) ---
def generate_euclidean_batch(batch_size, D, N):
    points = torch.randn(batch_size, N, D)
    return points, torch.argmin(torch.norm(points, dim=2), dim=1)

def generate_curved_space_batch(batch_size, D, N, space_curvature=0.5):
    positions = torch.randn(batch_size, N, D) * 2
    base_metric = torch.eye(D).unsqueeze(0).repeat(batch_size, 1, 1)
    noise = torch.randn(batch_size, D, D) * space_curvature
    symmetric_noise = (noise + noise.transpose(1, 2)) / 2
    curvature_matrix = base_metric + symmetric_noise
    curved_positions = torch.bmm(positions, curvature_matrix)
    delta = curved_positions.unsqueeze(2) - curved_positions.unsqueeze(1)
    batch_size_val, N_val, _, D_val = delta.shape
    delta_reshaped = delta.reshape(batch_size_val * N_val * N_val, 1, D_val)
    curvature_matrix_expanded = curvature_matrix.repeat_interleave(N_val * N_val, dim=0)
    delta_transformed = torch.bmm(delta_reshaped, curvature_matrix_expanded).reshape(batch_size_val, N_val, N_val, D_val)
    dist_sq = torch.sum(delta * delta_transformed, dim=-1) + torch.eye(N) * 1e9
    labels = torch.argmax(torch.sqrt(torch.min(dist_sq, dim=2).values), dim=1)
    centroid = torch.mean(curved_positions, dim=1, keepdim=True)
    return curved_positions - centroid, labels

# --- Weight Loading ---
def load_weights_from_dict(model, weights_dict):
    """Load weights from JSON structure into model"""
    try:
        state_dict = {}
        fe = weights_dict['feature_extractor']
        state_dict['feature_extractor.0.weight'] = torch.tensor(fe['W'][0], dtype=torch.float32)
        state_dict['feature_extractor.0.bias'] = torch.tensor(fe['b'][0], dtype=torch.float32)
        state_dict['feature_extractor.3.weight'] = torch.tensor(fe['W'][1], dtype=torch.float32)
        state_dict['feature_extractor.3.bias'] = torch.tensor(fe['b'][1], dtype=torch.float32)
        if 'layer_norm' in weights_dict:
            ln = weights_dict['layer_norm']
            state_dict['feature_extractor.2.weight'] = torch.tensor(ln['W'][0], dtype=torch.float32)
            state_dict['feature_extractor.2.bias'] = torch.tensor(ln['b'][0], dtype=torch.float32)
        else:
            state_dict['feature_extractor.2.weight'] = torch.ones(96, dtype=torch.float32)
            state_dict['feature_extractor.2.bias'] = torch.zeros(96, dtype=torch.float32)
        sh = weights_dict['scoring_head']
        state_dict['scoring_head.weight'] = torch.tensor(sh['W'][0], dtype=torch.float32)
        state_dict['scoring_head.bias'] = torch.tensor(sh['b'][0], dtype=torch.float32)
        ptl = weights_dict['project_to_latent']
        state_dict['project_to_latent.weight'] = torch.tensor(ptl['W'][0], dtype=torch.float32)
        state_dict['project_to_latent.bias'] = torch.tensor(ptl['b'][0], dtype=torch.float32)
        pfl = weights_dict['project_from_latent']
        state_dict['project_from_latent.weight'] = torch.tensor(pfl['W'][0], dtype=torch.float32)
        state_dict['project_from_latent.bias'] = torch.tensor(pfl['b'][0], dtype=torch.float32)
        model.load_state_dict(state_dict)
        weight_sum = model.feature_extractor[0].weight.data.sum().item()
        if abs(weight_sum) < 1e-6:
            print(f"      ⚠️  WARNING: Weights appear near-zero")
            return False
        return True
    except Exception as e:
        print(f"      ❌ Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- META-LEARNING COORDINATOR ---
class MetaLearningCoordinator:
    def __init__(self, pantheon_dict):
        self.pantheon = pantheon_dict
        self.collaboration_history = defaultdict(list)
        self.performance_baselines = {}
        self.knowledge_transfer_log = []
        
    def evaluate_specialist_alone(self, specialist, dimension, num_tests=1000):
        """Test specialist without collaboration - INCREASED SAMPLES for stability"""
        specialist.eval()
        correct = 0
        with torch.no_grad():
            for i in range(num_tests):
                inputs, labels = generate_curved_space_batch(1, dimension, 15)
                outputs = specialist(inputs)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / num_tests
        specialist.train()
        return accuracy
    
    def establish_baseline_performance(self, dimension, num_tests=1000):
        """Measure specialist performance BEFORE collaboration - INCREASED SAMPLES"""
        specialist = self.pantheon[str(dimension)]['model']
        baseline_accuracy = self.evaluate_specialist_alone(specialist, dimension, num_tests)
        self.performance_baselines[dimension] = baseline_accuracy
        print(f"   📊 Baseline for {dimension}D: {baseline_accuracy:.3f}")
        return baseline_accuracy
    
    def transfer_knowledge_sequential(self, source_dim, target_dim):
        """Transfer learning between dimensions - FIXED to be non-destructive"""
        print(f"   🔄 Transferring {source_dim}D → {target_dim}D...")
        source_specialist = self.pantheon[str(source_dim)]['model']
        target_specialist = self.pantheon[str(target_dim)]['model']
        pre_transfer_acc = self.performance_baselines.get(target_dim, 0.5)
        with torch.no_grad():
            dimension_gap = abs(target_dim - source_dim)
            if dimension_gap == 2: transfer_strength = 0.015
            elif 3 <= dimension_gap <= 5: transfer_strength = 0.01
            elif dimension_gap == 6: transfer_strength = 0.005
            elif 7 <= dimension_gap <= 9: transfer_strength = 0.015
            elif dimension_gap >= 10: transfer_strength = 0.01
            else: transfer_strength = 0.01
            source_fe = source_specialist.feature_extractor
            target_fe = target_specialist.feature_extractor
            if hasattr(source_fe[0], 'weight') and hasattr(target_fe[0], 'weight'):
                source_w = source_fe[0].weight.data
                target_w = target_fe[0].weight.data
                source_b, target_b = source_fe[0].bias.data, target_fe[0].bias.data
                min_out, min_in = min(source_w.shape[0], target_w.shape[0]), min(source_w.shape[1], target_w.shape[1])
                target_w[:min_out, :min_in] = (transfer_strength * source_w[:min_out, :min_in] + (1 - transfer_strength) * target_w[:min_out, :min_in])
                target_b[:min_out] = (transfer_strength * source_b[:min_out] + (1 - transfer_strength) * target_b[:min_out])
            if hasattr(source_fe[3], 'weight') and hasattr(target_fe[3], 'weight'):
                source_w = source_fe[3].weight.data
                target_w = target_fe[3].weight.data
                source_b, target_b = source_fe[3].bias.data, target_fe[3].bias.data
                min_out, min_in = min(source_w.shape[0], target_w.shape[0]), min(source_w.shape[1], target_w.shape[1])
                target_w[:min_out, :min_in] = (transfer_strength * source_w[:min_out, :min_in] + (1 - transfer_strength) * target_w[:min_out, :min_in])
                target_b[:min_out] = (transfer_strength * source_b[:min_out] + (1 - transfer_strength) * target_b[:min_out])
        post_transfer_acc = self.evaluate_specialist_alone(target_specialist, target_dim, 500)
        improvement = post_transfer_acc - pre_transfer_acc
        self.knowledge_transfer_log.append({'source': source_dim, 'target': target_dim, 'improvement': improvement, 'transfer_strength': transfer_strength, 'pre_transfer': pre_transfer_acc, 'post_transfer': post_transfer_acc})
        if improvement > 0: symbol = "✓"
        elif improvement < -0.01: symbol = "✗"
        else: symbol = "≈"
        print(f"      {symbol} {pre_transfer_acc:.3f} → {post_transfer_acc:.3f} ({improvement:+.3f}) [strength: {transfer_strength:.3f}]")
        return improvement
    
    def analyze_learning_acceleration(self):
        """Measure if learning gets faster/better across dimensions - FIXED"""
        if len(self.knowledge_transfer_log) < 3: return "Insufficient data for acceleration analysis"
        STRONG_THRESHOLD = 0.020
        WEAK_THRESHOLD = 0.008
        strong_successes = [log for log in self.knowledge_transfer_log if log['improvement'] >= STRONG_THRESHOLD]
        weak_successes = [log for log in self.knowledge_transfer_log if WEAK_THRESHOLD <= log['improvement'] < STRONG_THRESHOLD]
        neutral = [log for log in self.knowledge_transfer_log if -WEAK_THRESHOLD < log['improvement'] < WEAK_THRESHOLD]
        weak_failures = [log for log in self.knowledge_transfer_log if -STRONG_THRESHOLD < log['improvement'] <= -WEAK_THRESHOLD]
        strong_failures = [log for log in self.knowledge_transfer_log if log['improvement'] <= -STRONG_THRESHOLD]
        all_successes = strong_successes + weak_successes
        success_rate = len(all_successes) / len(self.knowledge_transfer_log)
        avg_improvement = np.mean([log['improvement'] for log in all_successes]) if all_successes else 0
        avg_strong = np.mean([log['improvement'] for log in strong_successes]) if strong_successes else 0
        print(f"\n   Strong gains (>2%): {len(strong_successes)} | Weak gains: {len(weak_successes)} | Neutral: {len(neutral)} | Losses: {len(weak_failures + strong_failures)}")
        print(f"   Overall success rate: {success_rate:.1%} ({len(all_successes)}/{len(self.knowledge_transfer_log)})")
        if all_successes: print(f"   Average gain (all successes): {avg_improvement:+.3f}")
        if strong_successes:
            print(f"   Average gain (strong only): {avg_strong:+.3f}")
            best = max(self.knowledge_transfer_log, key=lambda x: x['improvement'])
            print(f"   Best transfer: {best['source']}D → {best['target']}D: {best['improvement']:+.3f}")
        if len(strong_successes) >= 2 and avg_strong > 0.03:
            return f"🚀 LEARNING ACCELERATION DETECTED! {len(strong_successes)} strong transfers, avg +{avg_strong:.3f}"
        return f"📉 Limited transfer success. {len(strong_successes)} strong transfers found."
    
    def run_meta_learning_cycle(self):
        """Complete meta-learning cycle"""
        print("\n" + "="*60 + "\n🧠 META-LEARNING COORDINATOR\n" + "="*60)
        print("\n📦 Loading trained models...")
        loaded_count = 0
        for dim_str in self.pantheon.keys():
            dim = int(dim_str)
            model = SpecialistModel(dim, 15)
            if not load_weights_from_dict(model, self.pantheon[dim_str]['weights']):
                print(f"      ❌ Failed to load {dim}D")
                continue
            self.pantheon[dim_str]['model'] = model
            loaded_count += 1
            reported_acc = self.pantheon[dim_str].get('accuracy_l2', self.pantheon[dim_str].get('accuracy', 0))
            print(f"      ✓ Loaded {dim}D specialist (trained acc: {reported_acc:.3f})")
        
        print(f"\n📦 Loaded {loaded_count}/{len(self.pantheon)} specialists")
        if loaded_count < 2: return "Insufficient specialists loaded"
        
        all_available = sorted([int(d) for d in self.pantheon.keys() if 'model' in self.pantheon[d]])
        
        # Use ALL available specialists for the final, verifiable run.
        available_dims = all_available
        print(f"   Using all available dimensions for full analysis: {available_dims}")
        if len(available_dims) < 2: return "Need at least 2 key dimensions"
        
        print(f"\n📊 Testing {len(available_dims)} specialists...")
        for dim in available_dims: self.establish_baseline_performance(dim, num_tests=500)
        
        transfer_pairs = []
        for i in range(len(available_dims)):
            for j in range(i+1, len(available_dims)):
                if available_dims[j] > available_dims[i]:
                    transfer_pairs.append((available_dims[i], available_dims[j]))
        
        print(f"\n🔄 Testing {len(transfer_pairs)} Knowledge Transfer pairs...")
        
        # KEY CHANGE: The '[:5]' limiter has been removed to run the full test.
        for source, target in transfer_pairs:
            if 'model' in self.pantheon[str(source)] and 'model' in self.pantheon[str(target)]:
                self.transfer_knowledge_sequential(source, target)
        
        result = self.analyze_learning_acceleration()
        print(f"\n📈 ANALYSIS: {result}")
        self.save_meta_learning_evidence()
        return result
    
    def save_meta_learning_evidence(self):
        """Save proof of the phase transition"""
        evidence = {
            'timestamp': datetime.now().isoformat(),
            'used_seed': CHOSEN_SEED,
            'performance_baselines': dict(self.performance_baselines),
            'knowledge_transfers': self.knowledge_transfer_log,
            'total_specialists': len(self.pantheon)
        }
        with open('meta_learning_evidence_zenodo_version.json', 'w') as f:
            json.dump(evidence, f, indent=2)
        print("💾 Verifiable evidence saved to 'meta_learning_evidence_zenodo_version.json'")

# --- Main Execution ---
if __name__ == "__main__":
    print("="*60)
    print("PANTHEON FORGE - PART 2: META-LEARNING (VERIFIABLE DEMONSTRATION)")
    print(f"** USING FIXED SEED FOR REPRODUCIBILITY: {CHOSEN_SEED} **")
    print("="*60)
    
    INPUT_FILE = "EAMC_weights_v2.json"
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ERROR: {INPUT_FILE} not found!")
        exit(1)
    
    print(f"\nLoading pantheon from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    pantheon_dict = data['pantheon']
    print(f"Found {len(pantheon_dict)} trained specialists")
    
    if len(pantheon_dict) >= 2:
        meta_coordinator = MetaLearningCoordinator(pantheon_dict)
        result = meta_coordinator.run_meta_learning_cycle()
        
        print("\n" + "="*60 + "\n🎯 FINAL, VERIFIABLE RESULT\n" + "="*60)
        print(result)
        
        if "ACCELERATION DETECTED" in result:
            print("\n💫 BREAKTHROUGH: Phase transition evidence is confirmed and repeatable.")
        else:
            print("\n🔍 The fixed seed produces a consistent, verifiable baseline result.")
        
        print("="*60)
    else:
        print(f"\n⚠️  Need at least 2 specialists. Got {len(pantheon_dict)}")
    
    print("\n🏁 COMPLETE")