#!/usr/bin/env python3
"""اختبار PyTorch مع cosmos_advanced بعد تثبيت uv"""
import torch

print("🔍 اختبار البيئة...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# اختبار النموذج
print("\n🔧 اختبار Cosmos Advanced...")
try:
    from cosmos_advanced.config_system import CosmosAdvancedConfig
    from cosmos_advanced.cosmos_model_advanced import CosmosAdvancedModel
    
    config = CosmosAdvancedConfig(
        dim=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=100
    )
    
    model = CosmosAdvancedModel(config)
    
    # اختبار forward pass
    input_ids = torch.randint(0, 100, (1, 10))
    
    print(f"✅ تم إنشاء النموذج بنجاح!")
    print(f"📊 عدد المعاملات: {sum(p.numel() for p in model.parameters()):,}")
    
    with torch.no_grad():
        logits, diagnostics = model(input_ids)
        
    print(f"✅ Forward pass نجح!")
    print(f"📊 Output shape: {logits.shape}")
    
    # اختبار Safety System
    print(f"\n🛡️ اختبار Safety System...")
    safety_output = diagnostics.get('safety_report', {})
    if safety_output:
        print(f"✅ Safety System يعمل!")
        print(f"🛡️ Safety report keys: {list(safety_output.keys())}")
    else:
        print("ℹ️ Safety system لم يُنتج تقرير (قد يكون طبيعي)")
    
    print(f"\n🎯 جميع الاختبارات نجحت!")
    
except Exception as e:
    print(f"❌ خطأ: {e}")
    import traceback
    traceback.print_exc()