# 🚀 Cosmos Advanced - دليل البدء السريع

## تثبيت سريع (5 دقائق)

### 1. المتطلبات

```bash
pip install torch transformers
```

### 2. أول استخدام

```python
import torch
from config_system import CosmosAdvancedConfig
from cosmos_model_advanced import CosmosAdvancedModel

# إنشاء النموذج
config = CosmosAdvancedConfig(
    dim=512,      # نموذج صغير للتجربة
    n_layers=4
)
model = CosmosAdvancedModel(config)

# التمرير
input_ids = torch.randint(0, config.vocab_size, (1, 10))
logits, diagnostics = model(
    input_ids,
    return_diagnostics=True
)

print("✅ النموذج يعمل!")
```

## 🎯 3 استخدامات شائعة

### 1️⃣ للإبداع (كتابة، أفكار)

```python
config = CosmosAdvancedConfig()
config.get_preset("creative")
model = CosmosAdvancedModel(config)
```

### 2️⃣ للتحليل (بحث، دراسة)

```python
config = CosmosAdvancedConfig()
config.get_preset("analytical")
model = CosmosAdvancedModel(config)
```

### 3️⃣ للأمان (تطبيقات حساسة)

```python
config = CosmosAdvancedConfig()
config.get_preset("safe")
model = CosmosAdvancedModel(config)
```

## 💡 نصائح سريعة

1. **للسرعة**: فعّل فقط القدرات التي تحتاجها
```python
logits = model(
    input_ids,
    use_reasoning=True,   # فقط ما تحتاج
    use_memory=False,
    use_safety=False,
    use_evaluation=False
)
```

2. **للذاكرة**: استخدم gradient checkpointing
```python
config.gradient_checkpointing = True
```

3. **للتخصيص**: استخدم الإعدادات المسبقة
```python
config.get_preset("creative")  # أسرع من التخصيص اليدوي
```

## 💾 حفظ وتحميل

```python
# حفظ
model.save_pretrained("./my_model")

# تحميل
model = CosmosAdvancedModel.from_pretrained("./my_model")
```

## 🔧 حل المشاكل

### خطأ: Out of Memory
```python
# الحل:
config.gradient_checkpointing = True
config.dim = 256  # نموذج أصغر
config.n_layers = 2
```

### بطيء جداً
```python
# الحل:
config.get_preset("performance")
# أو فعّل قدرات أقل
```

## 📚 الخطوة التالية

1. اقرأ [README.md](README.md) للتفاصيل الكاملة
2. شغّل [example_usage.py](example_usage.py) للأمثلة
3. خصّص النموؤج لاحتياجاتك

---
**جاهز للانطلاق في 5 دقائق! 🚀**
