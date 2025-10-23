#!/usr/bin/env python3
"""Test the learning engine dimension fix"""

import sys
import torch
sys.path.insert(0, 'cosmos_advanced')

from learning_engine import FewShotLearner

print('🧪 Testing FewShotLearner dimension fix...')

dim = 64
learner = FewShotLearner(dim, num_shots=3)

# Test with different tensor shapes
print(f"🔍 Testing with dim={dim}")

# Create a query (batch_size=1, features=dim)
query = torch.randn(1, dim)
print(f"📝 Query shape: {query.shape}")

# Create support set (individual examples as 1D tensors)
support_set = [torch.randn(dim) for _ in range(3)]
print(f"📝 Support set shapes: {[s.shape for s in support_set]}")

try:
    # This should trigger the bilinear layer
    output = learner(query, support_set)
    print(f'✅ FewShotLearner forward pass successful!')
    print(f'📊 Output shape: {output.shape}')
    
except Exception as e:
    print(f'❌ Error in FewShotLearner: {e}')
    import traceback
    traceback.print_exc()