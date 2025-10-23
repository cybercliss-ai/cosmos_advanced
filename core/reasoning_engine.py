# reasoning_engine.py - محرك التفكير والاستدلال المتقدم
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math

@dataclass
class ThoughtNode:
    """عقدة تفكير في شجرة الأفكار"""
    content: str
    score: float
    depth: int
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class ChainOfThoughtModule(nn.Module):
    """وحدة التفكير المتسلسل (CoT)"""
    def __init__(self, hidden_dim: int, num_steps: int = 5):
        super().__init__()
        self.num_steps = num_steps
        self.step_encoder = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_steps)
        ])
        self.step_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_steps)
        ])
        self.thought_aggregator = nn.Linear(hidden_dim * num_steps, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """تنفيذ التفكير المتسلسل"""
        thoughts = []
        current = x
        
        for i in range(self.num_steps):
            # كل خطوة تفكير
            current = self.step_encoder[i](current)
            current = torch.relu(current)
            current = self.step_norms[i](current)
            thoughts.append(current)
        
        # دمج جميع الأفكار
        all_thoughts = torch.cat(thoughts, dim=-1)
        final_thought = self.thought_aggregator(all_thoughts)
        
        return final_thought, thoughts

class TreeOfThoughtsModule(nn.Module):
    """وحدة شجرة الأفكار (ToT)"""
    def __init__(self, hidden_dim: int, num_branches: int = 3, max_depth: int = 3):
        super().__init__()
        self.num_branches = num_branches
        self.max_depth = max_depth
        self.branch_generators = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_branches)
        ])
        self.evaluator = nn.Linear(hidden_dim, 1)  # لتقييم جودة كل فكرة
        self.selector = nn.Linear(hidden_dim, num_branches)  # لاختيار أفضل الفروع
        
    def forward(self, x: torch.Tensor, depth: int = 0) -> Tuple[torch.Tensor, List[ThoughtNode]]:
        """تنفيذ شجرة الأفكار"""
        if depth >= self.max_depth:
            return x, []
        
        # توليد فروع متعددة
        branches = []
        scores = []
        
        for i in range(self.num_branches):
            branch = self.branch_generators[i](x)
            branch = torch.relu(branch)
            score = torch.sigmoid(self.evaluator(branch))
            
            branches.append(branch)
            scores.append(score)
        
        # اختيار أفضل الفروع
        branch_scores = torch.cat(scores, dim=-1)
        branch_weights = torch.softmax(branch_scores, dim=-1)
        
        # دمج الفروع بناءً على أوزانها
        weighted_branches = sum(w.unsqueeze(-1) * b for w, b in zip(branch_weights.unbind(-1), branches))
        
        return weighted_branches, branches

class SelfConsistencyModule(nn.Module):
    """وحدة الاتساق الذاتي"""
    def __init__(self, hidden_dim: int, num_samples: int = 3):
        super().__init__()
        self.num_samples = num_samples
        self.samplers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_samples)
        ])
        self.consistency_checker = nn.Linear(hidden_dim * num_samples, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """توليد عدة حلول والتحقق من اتساقها"""
        samples = []
        
        for i in range(self.num_samples):
            sample = self.samplers[i](x)
            sample = torch.tanh(sample)
            samples.append(sample)
        
        # حساب الاتساق بين العينات
        all_samples = torch.cat(samples, dim=-1)
        consistency_score = torch.sigmoid(self.consistency_checker(all_samples))
        
        # اختيار العينة الأكثر تمثيلاً
        avg_sample = torch.stack(samples).mean(dim=0)
        
        return avg_sample, consistency_score.item()

class ReflexionModule(nn.Module):
    """وحدة التفكير التأملي (Reflexion)"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_evaluator = nn.Linear(hidden_dim, hidden_dim)
        self.error_detector = nn.Linear(hidden_dim, 1)
        self.corrector = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reflection_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, previous_output: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """تنفيذ التفكير التأملي والتصحيح الذاتي"""
        # تقييم الحالة الحالية
        evaluation = self.self_evaluator(x)
        error_score = torch.sigmoid(self.error_detector(evaluation))
        
        reflection_info = {
            'error_detected': error_score.item() > 0.5,
            'confidence': 1.0 - error_score.item()
        }
        
        # إذا كان هناك خطأ محتمل، نطبق التصحيح
        if error_score.item() > 0.5 and previous_output is not None:
            combined = torch.cat([x, previous_output], dim=-1)
            corrected = self.corrector(combined)
            corrected = self.reflection_norm(corrected)
            reflection_info['correction_applied'] = True
            return corrected, reflection_info
        
        reflection_info['correction_applied'] = False
        return x, reflection_info

class AnalogicalReasoningModule(nn.Module):
    """وحدة الاستدلال القياسي"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.analogy_mapper = nn.Linear(hidden_dim, hidden_dim)
        self.similarity_scorer = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.transfer_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """إيجاد التشابهات ونقل المعرفة"""
        # تعيين المصدر والهدف إلى مساحة مشتركة
        mapped_source = self.analogy_mapper(source)
        mapped_target = self.analogy_mapper(target)
        
        # حساب التشابه
        similarity = torch.sigmoid(self.similarity_scorer(mapped_source, mapped_target))
        
        # نقل المعرفة من المصدر إلى الهدف
        combined = torch.cat([mapped_source, mapped_target], dim=-1)
        transferred = self.transfer_network(combined)
        
        return transferred, similarity.item()

class ReasoningEngine(nn.Module):
    """محرك التفكير الشامل"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.dim
        
        # تهيئة جميع وحدات التفكير
        self.cot = ChainOfThoughtModule(hidden_dim, config.reasoning.reasoning_steps)
        self.tot = TreeOfThoughtsModule(
            hidden_dim, 
            config.reasoning.multi_path_exploration,
            config.reasoning.thinking_depth
        )
        self.self_consistency = SelfConsistencyModule(
            hidden_dim,
            config.reasoning.consistency_checks
        )
        self.reflexion = ReflexionModule(hidden_dim)
        self.analogical = AnalogicalReasoningModule(hidden_dim)
        
        # وحدة اختيار وضع التفكير
        self.mode_selector = nn.Linear(hidden_dim, 5)  # 5 modes
        
    def forward(
        self, 
        x: torch.Tensor, 
        mode: Optional[str] = None,
        previous_output: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """تنفيذ التفكير باستخدام الوضع المحدد"""
        reasoning_info = {}
        
        if mode is None:
            mode = self.config.reasoning.mode.value
        
        if mode == "chain_of_thought":
            output, thoughts = self.cot(x)
            reasoning_info['num_thoughts'] = len(thoughts)
            reasoning_info['mode'] = 'CoT'
            
        elif mode == "tree_of_thoughts":
            output, branches = self.tot(x)
            reasoning_info['num_branches'] = len(branches)
            reasoning_info['mode'] = 'ToT'
            
        elif mode == "self_consistency":
            output, consistency = self.self_consistency(x)
            reasoning_info['consistency_score'] = consistency
            reasoning_info['mode'] = 'Self-Consistency'
            
        elif mode == "reflexion":
            output, reflection_info = self.reflexion(x, previous_output)
            reasoning_info.update(reflection_info)
            reasoning_info['mode'] = 'Reflexion'
            
        else:
            output = x
            reasoning_info['mode'] = 'Direct'
        
        # إذا كان التفكير الموسع مفعلاً
        if self.config.reasoning.extended_thinking:
            # تطبيق المزيد من التحويلات
            output, _ = self.cot(output)
            reasoning_info['extended'] = True
        
        return output, reasoning_info
