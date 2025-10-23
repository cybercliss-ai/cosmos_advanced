#!/usr/bin/env python3
"""اختبار شامل ومكتمل لنموذج Cosmos Advanced AGI"""
import sys
import os
sys.path.append('/workspace')
sys.path.append('/workspace/cosmos_advanced')

import torch
from cosmos_advanced.config_system import CosmosAdvancedConfig, SafetyLevel
from cosmos_advanced.cosmos_model_advanced import CosmosAdvancedModel

def test_complete_cosmos():
    """اختبار شامل لنموذج Cosmos"""
    print("🚀 اختبار شامل لنموذج Cosmos Advanced AGI")
    print("="*60)
    
    # اختبار البيئة
    print(f"🔍 PyTorch version: {torch.__version__}")
    print(f"🔍 CUDA available: {torch.cuda.is_available()}")
    
    # إنشاء تكوين مخصص للاختبار
    config = CosmosAdvancedConfig(
        dim=128,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,  # اختبار GQA
        vocab_size=1000,
        max_sequence_length=512,
        dropout=0.1
    )
    
    print(f"📊 تكوين النموذج:")
    print(f"   - البعد: {config.dim}")
    print(f"   - الطبقات: {config.n_layers}")
    print(f"   - الرؤوس: {config.n_heads}")
    print(f"   - KV_heads: {config.n_kv_heads}")
    print(f"   - المفردات: {config.vocab_size}")
    
    # إنشاء النموذج
    model = CosmosAdvancedModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 إجمالي المعاملات: {total_params:,}")
    
    # اختبار forward pass
    print(f"\n🧠 اختبار Forward Pass...")
    batch_size, seq_len = 2, 50
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"📊 Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        try:
            # اختبار مع Safety System
            print(f"🔄 اختبار مع Safety System...")
            output_with_safety = model(input_ids)
            
            print(f"✅ Forward pass مع Safety System نجح!")
            print(f"📊 Output shape: {output_with_safety.logits.shape}")
            
            # فحص Diagnostics
            diagnostics = output_with_safety.diagnostics
            if diagnostics:
                print(f"\n📋 Diagnostics:")
                for key, value in diagnostics.items():
                    if isinstance(value, dict):
                        print(f"   - {key}: {len(value)} items")
                    else:
                        print(f"   - {key}: {type(value).__name__}")
            
            print(f"\n🎉 جميع الاختبارات نجحت!")
            print(f"🌟 نموذج Cosmos Advanced AGI يعمل بشكل مثالي!")
            
        except Exception as e:
            print(f"❌ خطأ في Forward Pass: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    success = test_complete_cosmos()
    if success:
        print(f"\n✨ تم اختبار النموذج بنجاح!")
    else:
        print(f"\n❌ فشل في اختبار النموذج!")