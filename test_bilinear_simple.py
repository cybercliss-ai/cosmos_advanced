#!/usr/bin/env python3
"""Test just the bilinear layer fix"""

import sys
import torch
sys.path.insert(0, 'cosmos_advanced')

print('🧪 Testing bilinear layer fix...')

# Test the specific issue: bilinear layer with mismatched dimensions
dim = 64
bilinear = torch.nn.Bilinear(dim, dim, 1)

# Simulate the tensors from the learning engine
# query_encoded: 2D tensor (batch_size, dim)
query_encoded = torch.randn(1, dim)
print(f"📝 query_encoded shape: {query_encoded.shape}")

# proto: 1D tensor (dim,) - this was the issue
proto_1d = torch.randn(dim)
print(f"📝 proto (1D) shape: {proto_1d.shape}")

# Test the OLD way (should fail)
try:
    dist = bilinear(query_encoded, proto_1d)
    print("❌ Old way should have failed but didn't")
except Exception as e:
    print(f"✅ Old way failed as expected: {e}")

# Test the NEW way (with dimension fix)
try:
    if proto_1d.dim() == 1:
        proto_expanded = proto_1d.unsqueeze(0).expand(query_encoded.size(0), -1)
    else:
        proto_expanded = proto_1d
    print(f"📝 proto_expanded shape: {proto_expanded.shape}")
    
    dist = bilinear(query_encoded, proto_expanded)
    print(f'✅ New way works! Distance shape: {dist.shape}')
    
except Exception as e:
    print(f"❌ New way still fails: {e}")
    import traceback
    traceback.print_exc()