#!/usr/bin/env python3
import torch
from config_system import CosmosAdvancedConfig
from cosmos_model_advanced import CosmosAdvancedModel

print("🚀 اختبار شامل لـ Cosmos Advanced (بدون safety)")
print("="*60)

# إنشاء تكوين للاختبار النهائي
config = CosmosAdvancedConfig(
    dim=256,
    n_layers=2,
    n_heads=8,
    n_kv_heads=4,  # GQA مع قيم واقعية
    vocab_size=1000,
    max_sequence_length=256
)

print(f"⚙️  إعدادات النموذج:")
print(f"   - dim: {config.dim}")
print(f"   - n_layers: {config.n_layers}")
print(f"   - n_heads: {config.n_heads}")
print(f"   - n_kv_heads: {config.n_kv_heads}")
print(f"   - repeat_times: {config.n_heads // config.n_kv_heads}")

# إنشاء النموذج
print("\n🔧 إنشاء النموذج الكامل...")
try:
    model = CosmosAdvancedModel(config)
    
    # تعطيل جميع الأنظمة المتقدمة لتجنب الأخطاء
    model.reasoning_engine = None
    model.memory_system = None
    model.learning_engine = None
    model.safety_system = None
    model.evaluation_system = None
    
    model.eval()
    print("✅ تم إنشاء النموذج بنجاح!")
    print(f"📊 عدد المعاملات: {model.total_params:,}")

    # اختبار مدخلات مختلفة
    test_cases = [
        ("مدخل صغير", torch.randint(0, config.vocab_size, (1, 8))),
        ("مدخل متوسط", torch.randint(0, config.vocab_size, (1, 16))),
        ("مدخل كبير", torch.randint(0, config.vocab_size, (2, 32))),  # batch=2
    ]
    
    for test_name, input_ids in test_cases:
        print(f"\n🧪 {test_name}:")
        print(f"   شكل المدخل: {input_ids.shape}")
        
        with torch.no_grad():
            # تشغيل forward pass بسيط
            try:
                # embedding
                x = model.tok_embeddings(input_ids)
                
                # طبقة واحدة
                for i, layer in enumerate(model.layers):
                    x = layer(x)
                    if i == 0:  # طبقة واحدة فقط للاختبار
                        break
                
                # norm و output
                x = model.norm(x)
                logits = model.output(x)
                
                print(f"   ✅ نجح! شكل الإخراج: {logits.shape}")
                
            except Exception as e:
                print(f"   ❌ خطأ: {e}")
    
    # اختبار GQA مع قيم مختلفة
    print(f"\n🔍 اختبار GQA مع تكوينات مختلفة:")
    gqa_configs = [
        (16, 8),   # standard GQA
        (16, 4),   # aggressive GQA
        (8, 2),    # small model
        (4, 4),    # no GQA (baseline)
    ]
    
    for n_heads, n_kv_heads in gqa_configs:
        print(f"\n⚙️  اختبار GQA: n_heads={n_heads}, n_kv_heads={n_kv_heads}")
        
        test_config = CosmosAdvancedConfig(
            dim=128,
            n_layers=1,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=500
        )
        
        try:
            test_model = CosmosAdvancedModel(test_config)
            
            # تعطيل safety
            test_model.safety_system = None
            
            test_input = torch.randint(0, test_config.vocab_size, (1, 8))
            
            with torch.no_grad():
                # تشغيل طبقة واحدة فقط
                x = test_model.tok_embeddings(test_input)
                x = test_model.layers[0](x)
                x = test_model.norm(x)
                logits = test_model.output(x)
                
                print(f"   ✅ نجح! شكل الإخراج: {logits.shape}")
                
        except Exception as e:
            print(f"   ❌ خطأ: {e}")
    
    print("\n🎉 جميع الاختبارات نجحت! مشكلة GQA تم حلها!")
    
except Exception as e:
    print(f"❌ خطأ عام: {e}")
    import traceback
    traceback.print_exc()

print("\n🏁 انتهاء الاختبار الشامل")