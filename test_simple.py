# test_simple.py - اختبار مبسط للنموذج
import torch
from config_system import CosmosAdvancedConfig
from cosmos_model_advanced import CosmosAdvancedModel

def test_basic_model():
    """اختبار النموذج الأساسي"""
    print("اختبار النموذج الأساسي...")
    
    # إنشاء تكوين مبسط
    config = CosmosAdvancedConfig(
        dim=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=8,  # نفس العدد لتجنب المشاكل
        vocab_size=1000,
        max_sequence_length=1024
    )
    
    print(f"التهيئة:")
    print(f"  - الأبعاد: {config.dim}")
    print(f"  - الطبقات: {config.n_layers}")
    print(f"  - الرؤوس: {config.n_heads}")
    print(f"  - رؤوس KV: {config.n_kv_heads}")
    
    # إنشاء النموذج
    model = CosmosAdvancedModel(config)
    print(f"تم إنشاء النموذج بنجاح!")
    print(f"عدد المعلمات: {model.total_params:,}")
    
    # اختبار التمرير
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"اختبار التمرير...")
    print(f"شكل الإدخال: {input_ids.shape}")
    
    model.eval()
    with torch.no_grad():
        try:
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
            print(f"شكل الإخراج: {logits.shape}")
            if diagnostics and 'model_info' in diagnostics:
                print(f"معلومات النموذج متوفرة")
            
        except Exception as e:
            print(f"❌ خطأ في التمرير: {e}")
            import traceback
            traceback.print_exc()

def test_with_reasoning():
    """اختبار مع قدرات التفكير"""
    print("\nاختبار مع قدرات التفكير...")
    
    config = CosmosAdvancedConfig(
        dim=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=8,  # نفس العدد لتجنب المشاكل
        vocab_size=2000,
        max_sequence_length=1024
    )
    
    model = CosmosAdvancedModel(config)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    with torch.no_grad():
        try:
            logits, diagnostics = model(
                input_ids,
                use_reasoning=True,
                use_memory=True,
                use_safety=False,
                use_evaluation=False,
                return_diagnostics=True
            )
            
            print(f"✅ نجح التمرير مع التفكير!")
            print(f"شكل الإخراج: {logits.shape}")
            
        except Exception as e:
            print(f"❌ خطأ في التمرير مع التفكير: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🧪 اختبار نموذج Cosmos المتقدم\n")
    
    test_basic_model()
    test_with_reasoning()
    
    print("\n✅ انتهى الاختبار!")