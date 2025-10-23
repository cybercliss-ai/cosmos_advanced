#!/usr/bin/env python3

print("🔍 اختبار PyTorch...")
try:
    import torch
    print(f"✅ PyTorch تم تحميله بنجاح!")
    print(f"📦 إصدار PyTorch: {torch.__version__}")
    print(f"🖥️  CUDA متاح: {torch.cuda.is_available()}")
    
    # اختبار بسيط
    x = torch.randn(2, 3)
    print(f"📊 tensor test: {x}")
    print(f"🎯 shape: {x.shape}")
    
    # اختبار matmul
    a = torch.randn(2, 4)
    b = torch.randn(4, 3)
    c = torch.matmul(a, b)
    print(f"🧮 matmul test: {c.shape}")
    
    # اختبار repeat_interleave
    tensor = torch.randn(2, 2, 3)
    repeated = tensor.repeat_interleave(2, dim=1)
    print(f"🔄 repeat_interleave test: {tensor.shape} -> {repeated.shape}")
    
    print("🎉 جميع اختبارات PyTorch نجحت!")
    
except Exception as e:
    print(f"❌ خطأ في PyTorch: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ انتهاء اختبار PyTorch")