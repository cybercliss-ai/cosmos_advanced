#!/usr/bin/env python3
"""Test script to verify imports work correctly"""

import sys
import os

# Add the cosmos_advanced directory to the path
sys.path.insert(0, '/workspace/cosmos_advanced')

try:
    print("🧪 Testing imports...")
    
    # Test config_system import
    from config_system import CosmosAdvancedConfig
    print("✅ config_system imported successfully")
    
    # Test cosmos_model_advanced import
    from core.cosmos_model_advanced import CosmosAdvancedModel
    print("✅ cosmos_model_advanced imported successfully")
    
    # Test model creation
    config = CosmosAdvancedConfig(
        dim=64, 
        n_layers=1, 
        n_heads=4, 
        n_kv_heads=2, 
        vocab_size=100
    )
    print("✅ Config created successfully")
    
    model = CosmosAdvancedModel(config)
    print("✅ Model created successfully")
    
    # Test a simple forward pass
    import torch
    x = torch.randint(0, 100, (1, 10))
    
    with torch.no_grad():
        output = model(x)
    
    print(f"✅ Forward pass successful! Output shape: {output[0].shape}")
    
    print("\n🎉 All tests passed! The imports and basic functionality work correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()