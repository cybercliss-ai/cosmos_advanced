# cosmos_model_advanced.py - نموذج Cosmos المتقدم الشامل
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# استيراد جميع الوحدات - دعم للعب كسكريبت ومودول
try:
    # محاولة الاستيراد النسبي (عند الاستخدام كمودول)
    from .config_system import CosmosAdvancedConfig
    from .reasoning_engine import ReasoningEngine
    from .memory_system import AdaptiveMemorySystem
    from .learning_engine import AdaptiveLearningEngine
    from .safety_module import ComprehensiveSafetySystem
    from .evaluation_module import SelfEvaluationSystem
except ImportError:
    # الاستيراد المطلق (عند التشغيل المباشر)
    from config_system import (
        CosmosAdvancedConfig,
        ReasoningMode,
        LearningMode,
        SafetyLevel,
        VerbosityLevel
    )
    from reasoning_engine import ReasoningEngine
    from memory_system import AdaptiveMemorySystem
    from learning_engine import AdaptiveLearningEngine
    from safety_module import ComprehensiveSafetySystem
    from evaluation_module import SelfEvaluationSystem

class RotaryPositionalEmbedding(nn.Module):
    """تطبيق محسن للترميز الدوار للوضعية"""
    def __init__(self, dim, max_seq_length=4096, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # بناء جداول cos و sin مسبقاً للأطوال الشائعة
        self.max_seq_length = max_seq_length
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, seq_len, device):
        """بناء جداول cache للـ cos و sin"""
        if self.cos_cached is not None and self.cos_cached.shape[-2] >= seq_len:
            return
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.cos_cached = emb.cos()[:, None, :]
        self.sin_cached = emb.sin()[:, None, :]

    def forward(self, x, seq_len=None):
        """
        تطبيق الترميز الدوار
        Args:
            x: tensor of shape [*, seq_len, dim] or [*, n_heads, seq_len, head_dim]
            seq_len: length of sequence
        """
        if seq_len is None:
            seq_len = x.size(-2)
        
        self._build_cache(seq_len, x.device)
        
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        return x * cos + self.rotate_half(x) * sin

    def rotate_half(self, x):
        """تدوير نصف البيانات"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class MultiQueryAttention(nn.Module):
    """الانتباه متعدد الاستعلامات المحسّن"""
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout
        
        self.q_proj = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_sequence_length, config.rope_theta, device=None)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # إسقاط queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # تطبيق RoPE - الحفاظ على الترتيب الصحيح للأبعاد
        # RoPE يتوقع أن يكون البُعد الثاني هو seq_len
        q = q.transpose(1, 2)  # [batch, seq_len, n_heads, head_dim]
        k = k.transpose(1, 2)  # [batch, seq_len, n_kv_heads, head_dim]
        
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        # إعادة ترتيب الأبعاد للانتباه
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, n_kv_heads, seq_len, head_dim]
        
        # Grouped-query attention - نسخ المفاتيح والقيم لمطابقة عدد الرؤوس
        if self.n_heads != self.n_kv_heads:
            # حساب عدد التكرار المطلوب
            repeat_times = self.n_heads // self.n_kv_heads
            print(f"Debug: n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, repeat_times={repeat_times}")
            print(f"Debug: قبل التكرار - k.shape={k.shape}, v.shape={v.shape}")
            
            # إعادة ترتيب الأبعاد للتكرار
            # من [batch, n_kv_heads, seq_len, head_dim] إلى [batch, seq_len, n_kv_heads, head_dim]
            k = k.transpose(1, 2)  # [batch, seq_len, n_kv_heads, head_dim]
            v = v.transpose(1, 2)  # [batch, seq_len, n_kv_heads, head_dim]
            
            # تكرار n_kv_heads إلى n_heads
            # من [batch, seq_len, n_kv_heads, head_dim] إلى [batch, seq_len, n_heads, head_dim]
            k = k.repeat(1, 1, repeat_times, 1)  # تكرار البُعد الثالث (n_heads)
            v = v.repeat(1, 1, repeat_times, 1)  # تكرار البُعد الثالث (n_heads)
            
            # إعادة ترتيب الأبعاد للعودة إلى الشكل المطلوب
            # من [batch, seq_len, n_heads, head_dim] إلى [batch, n_heads, seq_len, head_dim]
            k = k.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
            v = v.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
            
            print(f"Debug: بعد التكرار - k.shape={k.shape}, v.shape={v.shape}")
        
        # حساب الانتباه
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # تطبيق القناع
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # إضافة بُعد للرؤوس
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # تطبيق softmax
        attn = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn = self.attn_dropout(attn)
        
        # حساب الإخراج
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(out)

class RMSNorm(nn.Module):
    """طبقة طبيعية راجعة إلى الجذر (RMSNorm)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class FeedForward(nn.Module):
    """شبكة تغذية أمامية محسنة"""
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.activation_dropout)
        
    def forward(self, x):
        # تطبيق SiLU (Swish) بدلاً من ReLU
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    """كتلة تحويل محسنة"""
    def __init__(self, config):
        super().__init__()
        self.attention = MultiQueryAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = RMSNorm(config.dim, eps=config.norm_eps)
        self.norm2 = RMSNorm(config.dim, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class CosmosAdvancedModel(nn.Module):
    """
    نموذج Cosmos المتقدم مع جميع الإمكانيات:
    - التفكير والاستدلال المتقدم
    - الذاكرة التكيفية
    - التعلم الذاتي
    - الأمان والأخلاقيات
    - التقييم الذاتي
    """
    
    def __init__(self, config: CosmosAdvancedConfig):
        super().__init__()
        self.config = config
        
        # مكونات النموذج الأساسية
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # طبقات التحويل الأساسية
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        # مشاركة الأوزان بين embedding و output
        self.output.weight = self.tok_embeddings.weight
        
        # القدرات المتقدمة
        self.reasoning_engine = ReasoningEngine(config)
        self.memory_system = AdaptiveMemorySystem(config)
        self.learning_engine = AdaptiveLearningEngine(config)
        self.safety_system = ComprehensiveSafetySystem(config)
        self.evaluation_system = SelfEvaluationSystem(config)
        
        # شبكة دمج القدرات
        self.capability_fusion = nn.Sequential(
            nn.Linear(config.dim * 5, config.dim * 2),
            nn.LayerNorm(config.dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.dim * 2, config.dim)
        )
        
        self.apply(self._init_weights)
        
        # إحصائيات النموذج
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        use_reasoning: bool = True,
        use_memory: bool = True,
        use_learning: bool = False,
        use_safety: bool = True,
        use_evaluation: bool = True,
        examples: Optional[List] = None,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        تمرير متقدم مع جميع القدرات
        
        Args:
            input_ids: معرفات الإدخال
            use_reasoning: استخدام محرك التفكير
            use_memory: استخدام نظام الذاكرة
            use_learning: استخدام محرك التعلم
            use_safety: استخدام نظام الأمان
            use_evaluation: استخدام نظام التقييم
            examples: أمثلة للتعلم
            return_diagnostics: إرجاع معلومات التشخيص
        """
        diagnostics = {} if return_diagnostics else None
        
        # 1. الترميز الأساسي
        x = self.tok_embeddings(input_ids)
        x = self.dropout(x)
        
        # 2. التمرير عبر طبقات التحويل
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # أخذ التمثيل الأخير للمعالجة المتقدمة
        representation = x[:, -1, :]  # آخر توكن
        
        # 3. تطبيق القدرات المتقدمة
        capabilities = [representation]
        
        # التفكير والاستدلال
        if use_reasoning:
            reasoning_output, reasoning_info = self.reasoning_engine(representation)
            capabilities.append(reasoning_output)
            if diagnostics is not None:
                diagnostics['reasoning'] = reasoning_info
        else:
            capabilities.append(representation)
        
        # الذاكرة
        if use_memory:
            memory_output, memory_info = self.memory_system(representation)
            capabilities.append(memory_output)
            if diagnostics is not None:
                diagnostics['memory'] = memory_info
        else:
            capabilities.append(representation)
        
        # التعلم الذاتي
        if use_learning and examples:
            learning_output, learning_info = self.learning_engine(representation, examples=examples)
            capabilities.append(learning_output)
            if diagnostics is not None:
                diagnostics['learning'] = learning_info
        else:
            capabilities.append(representation)
        
        # الأمان
        if use_safety:
            safety_output, is_safe, safety_report = self.safety_system(representation)
            capabilities.append(safety_output)
            if diagnostics is not None:
                diagnostics['safety'] = safety_report
                diagnostics['is_safe'] = is_safe
            
            # إيقاف الإخراج إذا كان غير آمن
            if not is_safe and self.config.safety.safety_level.value in ['high', 'strict']:
                # إرجاع رسالة آمنة
                safe_output = torch.zeros_like(x)
                if diagnostics is not None:
                    diagnostics['output_blocked'] = True
                return self.output(safe_output), diagnostics
        else:
            capabilities.append(representation)
        
        # دمج جميع القدرات
        fused_capabilities = torch.cat(capabilities, dim=-1)
        enhanced_repr = self.capability_fusion(fused_capabilities)
        
        # تحديث التمثيل الأخير
        x_enhanced = x.clone()
        x_enhanced[:, -1, :] = enhanced_repr
        
        # 4. التقييم الذاتي
        if use_evaluation:
            # نحتاج query representation - نستخدم الأولي
            query_repr = x[:, 0, :]
            evaluated_output, eval_report = self.evaluation_system(
                enhanced_repr,
                query_repr
            )
            x_enhanced[:, -1, :] = evaluated_output
            
            if diagnostics is not None:
                diagnostics['evaluation'] = eval_report
        
        # 5. الإخراج النهائي
        logits = self.output(x_enhanced)
        
        if diagnostics is not None:
            diagnostics['model_info'] = {
                'total_params': self.total_params,
                'trainable_params': self.trainable_params,
                'config': self.config.to_dict()
            }
        
        return logits, diagnostics
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        use_reasoning: bool = True,
        use_memory: bool = True,
        use_safety: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """توليد نص متقدم"""
        # استخدام إعدادات التوليد من التكوين
        if temperature is None:
            temperature = self.config.generation.temperature
        if top_k is None:
            top_k = self.config.generation.top_k
        if top_p is None:
            top_p = self.config.generation.top_p
        
        generated_tokens = input_ids
        generation_diagnostics = []
        
        for step in range(max_new_tokens):
            # الحصول على الإخراج
            logits, diagnostics = self(
                generated_tokens,
                use_reasoning=use_reasoning,
                use_memory=use_memory,
                use_safety=use_safety,
                return_diagnostics=True
            )
            
            # إذا تم حظر الإخراج لأسباب أمنية
            if diagnostics and diagnostics.get('output_blocked', False):
                break
            
            # أخذ آخر logits
            next_token_logits = logits[:, -1, :]
            
            # تطبيق الحرارة
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Top-K sampling
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-P (nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # العينة
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # إضافة التوكن الجديد
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            
            if diagnostics:
                generation_diagnostics.append(diagnostics)
            
            # التحقق من stop sequences
            # (يتطلب tokenizer للتحقق من stop sequences)
        
        final_diagnostics = {
            'num_tokens_generated': generated_tokens.size(1) - input_ids.size(1),
            'generation_steps': generation_diagnostics
        }
        
        return generated_tokens, final_diagnostics
    
    def save_pretrained(self, save_directory: str):
        """حفظ النموذج مع جميع التكوينات"""
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        
        # حفظ التكوين
        self.config.save(os.path.join(save_directory, "config.json"))
        
        # حفظ النموذج
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # حفظ معلومات النموذج
        model_info = {
            "model_type": "CosmosAdvanced",
            "version": "3.0.0",
            "total_parameters": self.total_params,
            "trainable_parameters": self.trainable_params,
            "capabilities": [
                "advanced_reasoning",
                "adaptive_memory",
                "self_learning",
                "safety_guardrails",
                "self_evaluation",
                "multimodal_ready"
            ]
        }
        
        with open(os.path.join(save_directory, "model_info.json"), 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Model saved to {save_directory}")
        print(f"   Total parameters: {self.total_params:,}")
        print(f"   Trainable parameters: {self.trainable_params:,}")
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """تحميل النموذج من ملف"""
        config = CosmosAdvancedConfig.load(os.path.join(model_path, "config.json"))
        model = cls(config)
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu'))
        print(f"✅ Model loaded from {model_path}")
        return model