#!/usr/bin/env python3
"""اختبار بسيط لـ PyTorch"""
import torch

print("🔍 اختبار PyTorch...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# اختبار tensor operation بسيط
x = torch.randn(2, 3)
y = torch.randn(2, 3)
z = torch.matmul(x, y.t())

print(f"✅ PyTorch يعمل! Tensor shapes: {x.shape}, {y.shape}")
print(f"✅ Matrix multiplication نجح: {z.shape}")

# اختبار النموذج
print("\n🔧 اختبار Cosmos Advanced...")
try:
    from cosmos_advanced.config_system import CosmosAdvancedConfig
    from cosmos_advanced.cosmos_model_advanced import CosmosAdvancedAGI
    
    config = CosmosAdvancedConfig(
        dim=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=100
    )
    
    model = CosmosAdvancedAGI(config)
    
    # اختبار forward pass
    input_ids = torch.randint(0, 100, (1, 10))
    
    print(f"✅ تم إنشاء النموذج بنجاح!")
    print(f"📊 عدد المعاملات: {model.get_num_parameters():,}")
    
    with torch.no_grad():
        logits, diagnostics = model(input_ids)
        
    print(f"✅ Forward pass نجح!")
    print(f"📊 Output shape: {logits.shape}")
    print(f"🎯 جميع الاختبارات نجحت!")
    
except Exception as e:
    print(f"❌ خطأ: {e}")
    import traceback
    traceback.print_exc()