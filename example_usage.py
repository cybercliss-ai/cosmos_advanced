# example_usage.py - أمثلة استخدام نموذج Cosmos المتقدم
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from .core.config_system import (
    CosmosAdvancedConfig,
    ReasoningMode,
    LearningMode,
    SafetyLevel,
    VerbosityLevel
)
from core.cosmos_model_advanced import CosmosAdvancedModel

def example_1_basic_usage():
    """مثال 1: استخدام أساسي"""
    print("="*50)
    print("مثال 1: الاستخدام الأساسي")
    print("="*50)
    
    # إنشاء تكوين افتراضي
    config = CosmosAdvancedConfig()
    
    # إنشاء النموذج
    model = CosmosAdvancedModel(config)
    model.eval()
    
    # مدخل تجريبي
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    # التمرير
    with torch.no_grad():
        logits, diagnostics = model(
            input_ids,
            use_reasoning=True,
            use_memory=True,
            use_safety=True,
            use_evaluation=True,
            return_diagnostics=True
        )
    
    print(f"شكل الإخراج: {logits.shape}")
    print(f"عدد المعلمات: {model.total_params:,}")
    
    if diagnostics:
        print("\nتشخيصات النموذج:")
        if 'reasoning' in diagnostics:
            print(f"  - وضع التفكير: {diagnostics['reasoning'].get('mode', 'N/A')}")
        if 'memory' in diagnostics:
            print(f"  - حلقات مشابهة: {diagnostics['memory'].get('similar_episodes_found', 0)}")
        if 'safety' in diagnostics:
            print(f"  - آمن: {diagnostics.get('is_safe', True)}")
            print(f"  - عدد التحذيرات: {diagnostics['safety'].get('num_warnings', 0)}")
    
    print()

def example_2_creative_mode():
    """مثال 2: الوضع الإبداعي"""
    print("="*50)
    print("مثال 2: الوضع الإبداعي")
    print("="*50)
    
    config = CosmosAdvancedConfig()
    
    # تطبيق الإعدادات المسبقة للإبداع
    config.get_preset("creative")
    
    print("إعدادات الإبداع:")
    print(f"  - مستوى الإبداع: {config.creativity.creativity_level}")
    print(f"  - درجة الحرارة: {config.generation.temperature}")
    print(f"  - وضع التفكير: {config.reasoning.mode.value}")
    
    model = CosmosAdvancedModel(config)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    with torch.no_grad():
        generated, gen_info = model.generate(
            input_ids,
            max_new_tokens=50,
            use_reasoning=True
        )
    
    print(f"\nتم توليد {gen_info['num_tokens_generated']} توكن")
    print()

def example_3_safe_mode():
    """مثال 3: الوضع الآمن الصارم"""
    print("="*50)
    print("مثال 3: الوضع الآمن الصارم")
    print("="*50)
    
    config = CosmosAdvancedConfig()
    config.get_preset("safe")
    
    print("إعدادات الأمان:")
    print(f"  - مستوى الأمان: {config.safety.safety_level.value}")
    print(f"  - تصفية المحتوى: {config.safety.content_filtering}")
    print(f"  - منع الاختراق: {config.safety.jailbreak_prevention}")
    
    model = CosmosAdvancedModel(config)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    with torch.no_grad():
        logits, diagnostics = model(
            input_ids,
            use_safety=True,
            return_diagnostics=True
        )
    
    if diagnostics and 'safety' in diagnostics:
        safety_report = diagnostics['safety']
        print(f"\nتقرير الأمان:")
        print(f"  - آمن: {diagnostics.get('is_safe', True)}")
        print(f"  - عدد الفحوصات: {len(safety_report.get('checks_performed', []))}")
        print(f"  - التدخلات: {len(safety_report.get('interventions', []))}")
    
    print()

def example_4_custom_config():
    """مثال 4: تكوين مخصص"""
    print("="*50)
    print("مثال 4: التكوين المخصص")
    print("="*50)
    
    config = CosmosAdvancedConfig(
        # معمارية النموذج
        dim=512,
        n_layers=4,
        n_heads=8,
        
        # متقدم
        diffusion=False
    )
    
    # تخصيص التفكير
    config.reasoning.mode = ReasoningMode.TREE_OF_THOUGHTS
    config.reasoning.thinking_depth = 5
    config.reasoning.extended_thinking = True
    
    # تخصيص التعلم
    config.learning.mode = LearningMode.FEW_SHOT
    config.learning.few_shot_examples = 10
    config.learning.adaptation_speed = 0.7
    
    # تخصيص التواصل
    config.communication.verbosity = VerbosityLevel.DETAILED
    config.communication.explanation_depth = 5
    
    print("التكوين المخصص:")
    print(f"  - الأبعاد: {config.dim}")
    print(f"  - الطبقات: {config.n_layers}")
    print(f"  - وضع التفكير: {config.reasoning.mode.value}")
    print(f"  - عمق التفكير: {config.reasoning.thinking_depth}")
    
    model = CosmosAdvancedModel(config)
    print(f"\nعدد المعلمات: {model.total_params:,}")
    print()

def example_5_save_and_load():
    """مثال 5: حفظ وتحميل النموذج"""
    print("="*50)
    print("مثال 5: حفظ وتحميل النموذج")
    print("="*50)
    
    # إنشاء نموذج
    config = CosmosAdvancedConfig(dim=256, n_layers=2)
    model = CosmosAdvancedModel(config)
    
    # حفظ النموذج
    save_path = "./cosmos_advanced_model"
    model.save_pretrained(save_path)
    
    # تحميل النموذج
    loaded_model = CosmosAdvancedModel.from_pretrained(save_path)
    
    print(f"\nتم تحميل النموذج بنجاح!")
    print()

def example_6_learning_from_examples():
    """مثال 6: التعلم من الأمثلة"""
    print("="*50)
    print("مثال 6: التعلم من الأمثلة")
    print("="*50)
    
    config = CosmosAdvancedConfig()
    config.learning.mode = LearningMode.FEW_SHOT
    config.learning.few_shot_examples = 5
    
    model = CosmosAdvancedModel(config)
    model.eval()
    
    # إنشاء أمثلة تجريبية
    examples = [
        (torch.randn(config.dim), torch.randn(config.dim))
        for _ in range(5)
    ]
    
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    with torch.no_grad():
        logits, diagnostics = model(
            input_ids,
            use_learning=True,
            examples=examples,
            return_diagnostics=True
        )
    
    if diagnostics and 'learning' in diagnostics:
        print(f"وضع التعلم: {diagnostics['learning'].get('mode', 'N/A')}")
        print(f"عدد الأمثلة: {diagnostics['learning'].get('num_examples', 0)}")
    
    print()

def example_7_simple_inference():
    """مثال 7: استنتاج بسيط بدون قدرات متقدمة"""
    print("="*50)
    print("مثال 7: استنتاج بسيط")
    print("="*50)
    
    config = CosmosAdvancedConfig(dim=256, n_layers=2)
    model = CosmosAdvancedModel(config)
    model.eval()
    
    # مدخل بسيط
    input_ids = torch.randint(0, config.vocab_size, (2, 5))  # batch_size=2, seq_len=5
    
    with torch.no_grad():
        # استنتاج بسيط بدون قدرات متقدمة
        logits, diagnostics = model(
            input_ids,
            use_reasoning=False,
            use_memory=False,
            use_learning=False,
            use_safety=False,
            use_evaluation=False
        )
    
    print(f"شكل الإدخال: {input_ids.shape}")
    print(f"شكل الإخراج: {logits.shape}")
    print(f"عدد المعلمات: {model.total_params:,}")
    print(f"معلمات قابلة للتدريب: {model.trainable_params:,}")
    print()

if __name__ == "__main__":
    print("✨ أمثلة استخدام نموذج Cosmos المتقدم ✨\n")
    
    # تشغيل جميع الأمثلة
    try:
        example_1_basic_usage()
        example_2_creative_mode()
        example_3_safe_mode()
        example_4_custom_config()
        example_5_save_and_load()
        example_6_learning_from_examples()
        example_7_simple_inference()
        
        print("✅ تمت جميع الأمثلة بنجاح!")
        
    except Exception as e:
        print(f"❌ خطأ في تشغيل الأمثلة: {e}")
        import traceback
        traceback.print_exc()