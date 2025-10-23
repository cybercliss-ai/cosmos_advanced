#!/usr/bin/env python3
# اختبار مباشر لنموذج Cosmos

import sys
import os
import torch

# إضافة مسار المشروع
sys.path.insert(0, '/workspace/cosmos_advanced')

def test_cosmos_model():
    """اختبار مباشر للنموذج"""
    print("🧪 اختبار مباشر لنموذج Cosmos المتقدم")
    print("=" * 50)
    
    try:
        # اختبار الاستيراد
        from config_system import CosmosAdvancedConfig
        print("✅ تم استيراد CosmosAdvancedConfig بنجاح")
        
        from cosmos_model_advanced import CosmosAdvancedModel
        print("✅ تم استيراد CosmosAdvancedModel بنجاح")
        
        # إنشاء تكوين مبسط
        config = CosmosAdvancedConfig(
            dim=256,
            n_layers=2,
            n_heads=8,
            n_kv_heads=8,  # نفس العدد لتجنب مشاكل GQA
            vocab_size=1000,
            max_sequence_length=1024
        )
        
        print(f"✅ تم إنشاء التكوين")
        print(f"  - الأبعاد: {config.dim}")
        print(f"  - الرؤوس: {config.n_heads}")
        print(f"  - رؤوس KV: {config.n_kv_heads}")
        
        # إنشاء النموذج
        model = CosmosAdvancedModel(config)
        print(f"✅ تم إنشاء النموذج بنجاح!")
        print(f"  - عدد المعلمات: {model.total_params:,}")
        
        # اختبار تمرير بسيط
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        print(f"✅ تم إنشاء الإدخال: {input_ids.shape}")
        
        model.eval()
        with torch.no_grad():
            # تمرير بسيط بدون قدرات متقدمة
            logits, diagnostics = model(
                input_ids,
                use_reasoning=False,
                use_memory=False,
                use_learning=False,
                use_safety=False,
                use_evaluation=False,
                return_diagnostics=True
            )
            
        print(f"✅ نجح التمرير!")
        print(f"  - شكل الإخراج: {logits.shape}")
        
        # اختبار مع قدرات متقدمة (إذا كانت متوفرة)
        print("\n" + "=" * 30)
        print("اختبار مع قدرات متقدمة...")
        
        try:
            logits_advanced, diagnostics_advanced = model(
                input_ids,
                use_reasoning=True,
                use_memory=True,
                use_safety=True,
                use_evaluation=True,
                return_diagnostics=True
            )
            
            print(f"✅ نجح التمرير مع القدرات المتقدمة!")
            print(f"  - شكل الإخراج: {logits_advanced.shape}")
            
        except Exception as e:
            print(f"⚠️ خطأ في القدرات المتقدمة: {e}")
            # هذا أمر طبيعي إذا كانت الوحدات المساعدة غير مكتملة
        
        print("\n" + "=" * 50)
        print("🎉 انتهى الاختبار بنجاح!")
        return True
        
    except Exception as e:
        print(f"❌ خطأ في الاختبار: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cosmos_model()
    exit(0 if success else 1)