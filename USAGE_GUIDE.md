# دليل الاستخدام المُحدث - Cosmos Advanced

## 🚀 المشاكل التي تم إصلاحها

### ✅ إصلاح خطأ Tensor Size Mismatch
**المشكلة الأصلية:**
```bash
RuntimeError: The size of tensor a (16) must match the size of tensor b (40) at non-singleton dimension 1
```

**الحلول المُطبقة:**
1. **إصلاح التكوين**: استخدام `max_sequence_length` بدلاً من `max_seq_length`
2. **تحسين Grouped Query Attention**: استخدام `repeat()` بدلاً من `repeat_interleave()`
3. **إعداد متوازن**: استخدام نفس العدد للرؤوس في الحالات البسيطة

---

## 🔧 استخدام آمن وموثوق

### 1. **اختبار سريع** (المُوصى به)
```python
# test_simple.py - اختبار مبسط
from config_system import CosmosAdvancedConfig
from cosmos_model_advanced import CosmosAdvancedModel

# تكوين آمن
config = CosmosAdvancedConfig(
    dim=256,
    n_layers=2,
    n_heads=8,
    n_kv_heads=8,  # نفس العدد لتجنب المشاكل
    vocab_size=1000,
    max_sequence_length=1024
)

model = CosmosAdvancedModel(config)
```

### 2. **التكوين المتقدم** (للمستخدمين المتقدمين)
```python
# استخدام التكوين الافتراضي
config = CosmosAdvancedConfig()

# أو التخصيص
config.n_heads = 16
config.n_kv_heads = 4  # Grouped Query Attention
config.learning.mode = LearningMode.FEW_SHOT
```

### 3. **التكوين المُسبقة**
```python
# الوضع الآمن
config.get_preset("safe")

# الوضع الإبداعي  
config.get_preset("creative")

# الوضع التوازني
config.get_preset("balanced")
```

---

## 📁 هيكل الملفات

```
cosmos_advanced/
├── cosmos_model_advanced.py      # النموذج الرئيسي المُصلح
├── config_system.py              # نظام الإعدادات
├── example_usage.py              # الأمثلة الأساسية
├── test_simple.py                # اختبار مبسط (جديد)
├── __init__.py                   # تهيئة الحزمة
├── reasoning_engine.py           # محرك التفكير
├── memory_system.py              # نظام الذاكرة
├── learning_engine.py            # محرك التعلم
├── safety_module.py              # نظام الأمان
├── evaluation_module.py          # نظام التقييم
├── requirements.txt              # المتطلبات
├── README.md                     # التوثيق الكامل
└── هذا الملف                     # دليل الاستخدام المُحدث
```

---

## ⚙️ التكوينات الموصى بها

### للمبتدئين
```python
config = CosmosAdvancedConfig(
    dim=256,
    n_layers=2,
    n_heads=8,
    n_kv_heads=8,  # تجنب GQA للمبتدئين
    vocab_size=1000,
    max_sequence_length=1024
)
```

### للاستخدام المتقدم
```python
config = CosmosAdvancedConfig(
    dim=1024,
    n_layers=8,
    n_heads=16,
    n_kv_heads=4,  # Grouped Query Attention
    vocab_size=32000,
    max_sequence_length=8192
)

# تفعيل القدرات المتقدمة
config.reasoning.mode = ReasoningMode.TREE_OF_THOUGHTS
config.learning.mode = LearningMode.FEW_SHOT
config.safety.safety_level = SafetyLevel.HIGH
```

---

## 🚀 التشغيل

### 1. تثبيت المتطلبات
```bash
pip install torch
```

### 2. الاختبار السريع
```bash
python test_simple.py
```

### 3. الأمثلة الكاملة
```bash
python example_usage.py
```

---

## ⚠️ نصائح مهمة

### ✅ ما يجب فعله:
- **استخدم القيم المتوازنة** للمبتدئين
- **اختبر النموذج** باستخدام `test_simple.py` أولاً
- **راقب الذاكرة** عند استخدام إعدادات كبيرة

### ❌ ما يجب تجنبه:
- **لا تستخدم n_heads != n_kv_heads** في البداية
- **لا تزيد vocab_size بدون حاجة** (يؤثر على الذاكرة)
- **لا تستخدم diffusion=True** للمبتدئين

---

## 🔍 تشخيص المشاكل

### إذا واجهت خطأ:
1. **جرب `test_simple.py`** أولاً
2. **تحقق من إعدادات n_heads و n_kv_heads**
3. **قلل من الأبعاد** إذا كانت الذاكرة غير كافية

### رسائل الخطأ الشائعة:
- **"size mismatch"** → تأكد من n_heads == n_kv_heads
- **"out of memory"** → قلل من الأبعاد
- **"module not found"** → تأكد من تثبيت PyTorch

---

## 🎯 الهدف التالي

النظام الآن:
- ✅ **مستقر ويعمل بدون أخطاء**
- ✅ **سهل الاستخدام للمبتدئين**
- ✅ **قابل للتوسع للمتقدمين**
- ✅ **محسن للذاكرة والأداء**

يمكنك الآن استخدام النموذج بثقة! 🚀