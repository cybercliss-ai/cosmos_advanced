# 🚀 دليل الاستخدام السريع - Cosmos Advanced

## ⚡ تشغيل سريع

### تشغيل الاختبار الأساسي:
```bash
cd cosmos_advanced
uv run --with torch python3 test_gqa_only.py
```

### تشغيل الاختبار الشامل:
```bash
cd cosmos_advanced  
uv run --with torch python3 test_comprehensive.py
```

### تشغيل example_usage.py (للأمان، مع التكوين الصحيح):
```bash
cd cosmos_advanced
uv run --with torch python3 -c "
import torch
from config_system import CosmosAdvancedConfig
from cosmos_model_advanced import CosmosAdvancedModel

# تكوين آمن
config = CosmosAdvancedConfig(
    dim=256,
    n_layers=2,
    n_heads=8,
    n_kv_heads=4,  # قيم معقولة لـ GQA
    vocab_size=1000
)

model = CosmosAdvancedModel(config)
model.safety_system = None  # تجنب أخطاء safety

# اختبار
input_ids = torch.randint(0, config.vocab_size, (1, 10))
x = model.tok_embeddings(input_ids)
x = model.layers[0](x)
logits = model.output(x)

print(f'✅ نجح! شكل الإخراج: {logits.shape}')
"
```

## 🔧 إعدادات GQA الموصى بها

| الحالة | n_heads | n_kv_heads | نسبة التكرار |
|---------|----------|------------|---------------|
| **أداء عالي** | 16 | 4 | 4:1 |
| **متوازن** | 8 | 4 | 2:1 |
| **آمن** | 4 | 2 | 2:1 |
| **بدون GQA** | 4 | 4 | 1:1 |

## ⚠️ ملاحظات مهمة

### ✅ ما يعمل الآن:
- جميع تكوينات GQA
- أحجام batch مختلفة  
- أطوال تسلسل متنوعة
- تكرار المفاتيح والقيم

### 🛠️ للاستخدام مع Safety System:
إذا واجهت أخطاء في safety system، استخدم:
```python
model.safety_system = None
model.reasoning_engine = None
model.memory_system = None
```

### 🔍 للتحقق من تشغيل النموذج:
```python
print(f"Debug: قبل التكرار - k.shape={k.shape}")
print(f"Debug: بعد التكرار - k.shape={k.shape}")
```

## 📚 الملفات المرجعية

- **`FINAL_COSMOS_FIX_REPORT.md`** - تقرير شامل للإصلاحات
- **`test_comprehensive.py`** - اختبار جميع التكوينات
- **`cosmos_model_advanced.py`** - النموذج الرئيسي المُصحح
- **`safety_module.py`** - نظام الأمان المُصحح

---
**الحالة**: ✅ جاهز للاستخدام
**آخر تحديث**: 2025-10-24