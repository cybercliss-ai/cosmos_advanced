#!/usr/bin/env python3
import torch
from config_system import CosmosAdvancedConfig
from cosmos_model_advanced import CosmosAdvancedModel

print("🚀 بدء اختبار مصغر لـ Cosmos Advanced")
print("="*50)

# إنشاء تكوين مع قيم صغيرة للاختبار
config = CosmosAdvancedConfig(
    dim=64,
    n_layers=1,
    n_heads=4,
    n_kv_heads=2,  # قيمة مختلفة لـ GQA
    vocab_size=100
)

print(f"⚙️  إعدادات النموذج:")
print(f"   - n_heads: {config.n_heads}")
print(f"   - n_kv_heads: {config.n_kv_heads}")
print(f"   - repeat_times: {config.n_heads // config.n_kv_heads}")

# إنشاء النموذج
print("\n🔧 إنشاء النموذج...")
model = CosmosAdvancedModel(config)
model.eval()

# مدخل تجريبي صغير
print("\n📝 إنشاء المدخل التجريبي...")
input_ids = torch.randint(0, config.vocab_size, (1, 5))  # batch=1, seq_len=5
print(f"   - شكل المدخل: {input_ids.shape}")

# تشغيل النموذج
print("\n▶️  تشغيل النموذج...")
try:
    with torch.no_grad():
        print("📊 المدخلات إلى النموذج...")
        print(f"   input_ids.shape: {input_ids.shape}")
        print(f"   input_ids: {input_ids}")
        
        print("\n🧠 تشغيل forward pass...")
        output = model(input_ids)
        print(f"✅ نجح! شكل الإخراج: {output[0].shape if isinstance(output, tuple) else output.shape}")
        
except Exception as e:
    print(f"❌ خطأ: {e}")
    import traceback
    traceback.print_exc()

print("\n🏁 انتهاء الاختبار")