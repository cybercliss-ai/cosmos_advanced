#!/usr/bin/env python3
import torch
from config_system import CosmosAdvancedConfig
from core.cosmos_model_advanced import CosmosAdvancedModel, CosmosAdvancedModel

print("🚀 اختبار أساسي لـ Cosmos بدون safety system")
print("="*60)

# إنشاء تكوين مع قيم صغيرة للاختبار
config = CosmosAdvancedConfig(
    dim=64,
    n_layers=1,
    n_heads=4,
    n_kv_heads=2,  # قيمة مختلفة لـ GQA
    vocab_size=100,
    max_sequence_length=128
)

print(f"⚙️  إعدادات النموذج:")
print(f"   - n_heads: {config.n_heads}")
print(f"   - n_kv_heads: {config.n_kv_heads}")
print(f"   - repeat_times: {config.n_heads // config.n_kv_heads}")

# إنشاء النموذج الأساسي فقط
print("\n🔧 إنشاء النموذج...")
try:
    # إنشاء نموذج بطريقة بسيطة
    model = CosmosAdvancedModel(config)
    
    # تعطيل safety system
    model.safety_system = None
    
    model.eval()
    print("✅ تم إنشاء النموذج بنجاح!")

    # مدخل تجريبي صغير
    print("\n📝 إنشاء المدخل التجريبي...")
    input_ids = torch.randint(0, config.vocab_size, (1, 5))  # batch=1, seq_len=5
    print(f"   - شكل المدخل: {input_ids.shape}")

    # تشغيل النموذج
    print("\n▶️  تشغيل النموذج...")
    with torch.no_grad():
        print("🧠 تشغيل forward pass...")
        
        # تشغيل forward pass بسيط
        x = model.tok_embeddings(input_ids)
        print(f"📊 بعد embedding: {x.shape}")
        
        # تشغيل طبقة واحدة
        x = model.layers[0](x)
        print(f"📊 بعد طبقة التحويل: {x.shape}")
        
        # تطبيق norm
        x = model.norm(x)
        print(f"📊 بعد norm: {x.shape}")
        
        # تطبيق output projection
        logits = model.output(x)
        print(f"📊 logits: {logits.shape}")
        
        print("✅ نجح الاختبار الأساسي!")
        
except Exception as e:
    print(f"❌ خطأ: {e}")
    import traceback
    traceback.print_exc()

print("\n🏁 انتهاء الاختبار الأساسي")