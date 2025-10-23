# learning_engine.py - محرك التعلم الذاتي
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import math

class InContextLearner(nn.Module):
    """التعلم من السياق (ICL)"""
    def __init__(self, dim: int):
        super().__init__()
        self.example_encoder = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        self.pattern_extractor = nn.MultiheadAttention(dim, num_heads=8)
        self.adaptation_layer = nn.Linear(dim, dim)
        
    def forward(self, query: torch.Tensor, examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """التعلم من أمثلة في السياق"""
        if not examples:
            return query
        
        # ترميز الأمثلة (input, output pairs)
        encoded_examples = []
        for input_ex, output_ex in examples:
            combined = torch.cat([input_ex, output_ex], dim=-1)
            encoded = self.example_encoder(combined)
            encoded_examples.append(encoded)
        
        # استخراج الأنماط من الأمثلة
        examples_tensor = torch.stack(encoded_examples)
        pattern, _ = self.pattern_extractor(
            query.unsqueeze(0),
            examples_tensor.unsqueeze(1),
            examples_tensor.unsqueeze(1)
        )
        
        # تطبيق النمط المستخرج
        adapted = self.adaptation_layer(pattern.squeeze(0))
        return query + adapted

class FewShotLearner(nn.Module):
    """التعلم من أمثلة قليلة (Few-Shot)"""
    def __init__(self, dim: int, num_shots: int = 5):
        super().__init__()
        self.num_shots = num_shots
        
        # شبكة نموذج أولي (Prototypical Network)
        self.prototype_encoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim)
        )
        
        self.distance_metric = nn.Bilinear(dim, dim, 1)
        self.meta_learner = nn.Linear(dim, dim)
        
    def forward(self, query: torch.Tensor, support_set: List[torch.Tensor]) -> torch.Tensor:
        """التعلم من مجموعة دعم صغيرة"""
        if not support_set:
            return query
        
        # إنشاء النماذج الأولية
        prototypes = []
        for example in support_set[:self.num_shots]:
            prototype = self.prototype_encoder(example)
            prototypes.append(prototype)
        
        # حساب المسافة من جميع النماذج الأولية
        query_encoded = self.prototype_encoder(query)
        
        if prototypes:
            distances = []
            for proto in prototypes:
                # Ensure both tensors have compatible dimensions for bilinear layer
                if proto.dim() == 1:  # proto is 1D, add batch dimension
                    # Expand proto to match query_encoded's batch dimension
                    proto_expanded = proto.unsqueeze(0).expand(query_encoded.size(0), -1)
                else:
                    proto_expanded = proto
                
                dist = self.distance_metric(query_encoded, proto_expanded)
                distances.append(dist)
            
            # الانتباه المبني على المسافة
            distances = torch.stack(distances)
            weights = F.softmax(-distances, dim=0)  # أقرب = وزن أكبر
            
            # دمج النماذج الأولية
            weighted_proto = sum(w * p for w, p in zip(weights, prototypes))
            
            # تطبيق التعلم الوصفي
            adapted = self.meta_learner(weighted_proto)
            return query + adapted
        
        return query

class MetaLearner(nn.Module):
    """التعلم الوصفي - التعلم عن كيفية التعلم"""
    def __init__(self, dim: int):
        super().__init__()
        # MAML-inspired architecture
        self.fast_weights_generator = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim * dim)  # توليد أوزان سريعة
        )
        
        self.task_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor, task_examples: List[torch.Tensor]) -> torch.Tensor:
        """تكييف سريع مع مهمة جديدة"""
        if not task_examples:
            return x
        
        # ترميز المهمة من الأمثلة
        task_repr = torch.stack(task_examples).mean(dim=0)
        task_encoded = self.task_encoder(task_repr)
        
        # توليد أوزان سريعة خاصة بالمهمة
        fast_weights_flat = self.fast_weights_generator(task_encoded)
        fast_weights = fast_weights_flat.view(self.task_encoder[0].in_features, -1)
        
        # تطبيق التحويل الخاص بالمهمة
        adapted = F.linear(x, fast_weights[:, :x.size(-1)])
        
        return adapted

class ContinualLearner(nn.Module):
    """التعلم المستمر - التعلم بدون نسيان"""
    def __init__(self, dim: int, num_tasks: int = 10):
        super().__init__()
        self.dim = dim
        self.num_tasks = num_tasks
        
        # Elastic Weight Consolidation (EWC) inspired
        self.task_specific_layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_tasks)
        ])
        
        self.importance_weights = nn.ParameterList([
            nn.Parameter(torch.ones(dim)) for _ in range(num_tasks)
        ])
        
        self.task_selector = nn.Linear(dim, num_tasks)
        self.shared_representation = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """التعلم مع الاحتفاظ بالمعرفة السابقة"""
        # تمثيل مشترك
        shared = self.shared_representation(x)
        
        if task_id is not None and task_id < self.num_tasks:
            # استخدام طبقة خاصة بالمهمة
            task_specific = self.task_specific_layers[task_id](shared)
            importance = self.importance_weights[task_id]
            output = shared + importance.unsqueeze(0) * task_specific
        else:
            # اختيار تلقائي للمهمة
            task_scores = F.softmax(self.task_selector(shared), dim=-1)
            output = shared.clone()
            
            for i, layer in enumerate(self.task_specific_layers):
                weight = task_scores[..., i].unsqueeze(-1)
                importance = self.importance_weights[i]
                task_out = layer(shared)
                output += weight * importance.unsqueeze(0) * task_out
        
        return output

class TransferLearner(nn.Module):
    """نقل التعلم بين المجالات"""
    def __init__(self, dim: int):
        super().__init__()
        self.source_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        
        self.target_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        
        # Domain adaptation layer
        self.domain_adapter = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, source_knowledge: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        """نقل المعرفة من مجال المصدر إلى الهدف"""
        source_encoded = self.source_encoder(source_knowledge)
        target_encoded = self.target_encoder(target_input)
        
        # تكييف المجال
        combined = torch.cat([source_encoded, target_encoded], dim=-1)
        adapted = self.domain_adapter(combined)
        
        return target_encoded + adapted

class ActiveLearner(nn.Module):
    """التعلم النشط - اختيار الأمثلة الأكثر فائدة"""
    def __init__(self, dim: int):
        super().__init__()
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.informativeness_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
    def select_examples(self, candidates: List[torch.Tensor], num_select: int = 5) -> List[int]:
        """اختيار الأمثلة الأكثر فائدة للتعلم"""
        scores = []
        
        for candidate in candidates:
            uncertainty = self.uncertainty_estimator(candidate).item()
            informativeness = self.informativeness_scorer(candidate).item()
            
            # دمج عدم اليقين والمعلوماتية
            score = 0.5 * uncertainty + 0.5 * informativeness
            scores.append(score)
        
        # اختيار الأمثلة ذات الدرجات الأعلى
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted_indices[:num_select]

class AdaptiveLearningEngine(nn.Module):
    """محرك التعلم التكيفي الشامل"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.dim
        
        # أنواع التعلم المختلفة
        self.in_context = InContextLearner(dim)
        self.few_shot = FewShotLearner(dim, config.learning.few_shot_examples)
        self.meta_learner = MetaLearner(dim)
        self.continual = ContinualLearner(dim)
        self.transfer = TransferLearner(dim)
        self.active = ActiveLearner(dim)
        
        # شبكة اختيار استراتيجية التعلم
        self.strategy_selector = nn.Linear(dim, 5)
        
        # ذاكرة الأمثلة
        self.example_buffer = []
        self.max_examples = 100
        
    def forward(
        self, 
        x: torch.Tensor, 
        mode: Optional[str] = None,
        examples: Optional[List] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """تطبيق التعلم التكيفي"""
        learning_info = {}
        
        if mode is None:
            mode = self.config.learning.mode.value
        
        if mode == "in_context" and examples:
            output = self.in_context(x, examples)
            learning_info['mode'] = 'In-Context Learning'
            learning_info['num_examples'] = len(examples)
            
        elif mode == "few_shot" and examples:
            support_set = [ex[0] if isinstance(ex, tuple) else ex for ex in examples]
            output = self.few_shot(x, support_set)
            learning_info['mode'] = 'Few-Shot Learning'
            learning_info['support_size'] = len(support_set)
            
        elif mode == "meta_learning" and examples:
            task_examples = [ex[0] if isinstance(ex, tuple) else ex for ex in examples]
            output = self.meta_learner(x, task_examples)
            learning_info['mode'] = 'Meta-Learning'
            
        elif mode == "continual_learning":
            output = self.continual(x)
            learning_info['mode'] = 'Continual Learning'
            
        else:
            output = x
            learning_info['mode'] = 'Direct'
        
        # تخزين في مخزن الأمثلة للتعلم المستقبلي
        if self.config.learning.adaptation_speed > 0:
            self.example_buffer.append(x.detach())
            if len(self.example_buffer) > self.max_examples:
                self.example_buffer.pop(0)
            learning_info['buffer_size'] = len(self.example_buffer)
        
        return output, learning_info
    
    def adapt_to_feedback(self, feedback_signal: torch.Tensor):
        """التكيف بناءً على تغذية راجعة"""
        if self.example_buffer:
            # استخدام التغذية الراجعة لتحديث استراتيجية التعلم
            pass
