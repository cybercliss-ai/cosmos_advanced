# memory_system.py - نظام الذاكرة المتقدم
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import math

class MemoryEntry:
    """إدخال في الذاكرة"""
    def __init__(self, key: torch.Tensor, value: torch.Tensor, importance: float = 1.0, timestamp: int = 0):
        self.key = key
        self.value = value
        self.importance = importance
        self.timestamp = timestamp
        self.access_count = 0

class WorkingMemory(nn.Module):
    """الذاكرة العاملة - للمعلومات الحالية"""
    def __init__(self, dim: int, capacity: int = 2048):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        
        # شبكات الاسترجاع
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        
    def store(self, x: torch.Tensor, importance: float = 1.0):
        """تخزين معلومة جديدة"""
        key = self.key_proj(x)
        value = self.value_proj(x)
        entry = MemoryEntry(key, value, importance, len(self.memories))
        self.memories.append(entry)
        
    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """استرجاع أقرب الذكريات"""
        if len(self.memories) == 0:
            return torch.zeros_like(query)
        
        q = self.query_proj(query)
        
        # حساب التشابه مع جميع الذكريات
        similarities = []
        for entry in self.memories:
            sim = F.cosine_similarity(q, entry.key, dim=-1)
            # دمج التشابه مع الأهمية
            score = sim * entry.importance
            similarities.append((score, entry))
        
        # اختيار أفضل k ذكريات
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_entries = similarities[:top_k]
        
        # دمج الذكريات المسترجعة
        if top_entries:
            values = torch.stack([entry.value for _, entry in top_entries])
            scores = torch.stack([score for score, _ in top_entries])
            weights = F.softmax(scores, dim=0).unsqueeze(-1)
            retrieved = (values * weights).sum(dim=0)
            
            # تحديث عدد الوصول
            for _, entry in top_entries:
                entry.access_count += 1
                
            return retrieved
        
        return torch.zeros_like(query)
    
    def clear(self):
        """مسح الذاكرة"""
        self.memories.clear()

class EpisodicMemory(nn.Module):
    """الذاكرة الحلقية - للأحداث المحددة"""
    def __init__(self, dim: int, num_episodes: int = 100):
        super().__init__()
        self.dim = dim
        self.num_episodes = num_episodes
        self.episodes = []
        
        self.episode_encoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        self.context_encoder = nn.Linear(dim, dim)
        
    def store_episode(self, states: List[torch.Tensor], context: torch.Tensor):
        """تخزين حلقة كاملة"""
        # دمج جميع الحالات في الحلقة
        episode_repr = torch.stack(states).mean(dim=0)
        encoded_episode = self.episode_encoder(episode_repr)
        encoded_context = self.context_encoder(context)
        
        episode = {
            'representation': encoded_episode,
            'context': encoded_context,
            'states': states,
            'timestamp': len(self.episodes)
        }
        
        self.episodes.append(episode)
        
        # الاحتفاظ بعدد محدود من الحلقات
        if len(self.episodes) > self.num_episodes:
            self.episodes.pop(0)
    
    def retrieve_similar_episodes(self, query: torch.Tensor, top_k: int = 3) -> List[Dict]:
        """استرجاع الحلقات المشابهة"""
        if not self.episodes:
            return []
        
        similarities = []
        for episode in self.episodes:
            sim = F.cosine_similarity(query, episode['representation'], dim=-1)
            similarities.append((sim.item(), episode))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in similarities[:top_k]]

class SemanticMemory(nn.Module):
    """الذاكرة الدلالية - للحقائق والمفاهيم"""
    def __init__(self, dim: int, num_concepts: int = 1000):
        super().__init__()
        self.dim = dim
        self.num_concepts = num_concepts
        
        # مصفوفة المفاهيم
        self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, dim))
        self.concept_attention = nn.MultiheadAttention(dim, num_heads=8)
        
        self.knowledge_encoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """استرجاع المعرفة الدلالية"""
        # الانتباه للمفاهيم ذات الصلة
        attended, _ = self.concept_attention(
            x.unsqueeze(0),
            self.concept_embeddings.unsqueeze(1),
            self.concept_embeddings.unsqueeze(1)
        )
        
        # دمج مع المدخل الأصلي
        combined = x + attended.squeeze(0)
        knowledge = self.knowledge_encoder(combined)
        
        return knowledge

class ProceduralMemory(nn.Module):
    """الذاكرة الإجرائية - لكيفية القيام بالأشياء"""
    def __init__(self, dim: int, num_skills: int = 50):
        super().__init__()
        self.dim = dim
        self.num_skills = num_skills
        
        # مكتبة المهارات
        self.skill_library = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            ) for _ in range(num_skills)
        ])
        
        self.skill_selector = nn.Linear(dim, num_skills)
        
    def forward(self, x: torch.Tensor, skill_id: Optional[int] = None) -> torch.Tensor:
        """تطبيق مهارة محددة"""
        if skill_id is not None:
            # استخدام مهارة محددة
            return self.skill_library[skill_id](x)
        else:
            # اختيار المهارة الأنسب تلقائياً
            skill_scores = F.softmax(self.skill_selector(x), dim=-1)
            
            # تطبيق مزيج من المهارات
            output = torch.zeros_like(x)
            for i, skill in enumerate(self.skill_library):
                weight = skill_scores[..., i].unsqueeze(-1)
                output += weight * skill(x)
            
            return output

class MemoryConsolidation(nn.Module):
    """توحيد الذاكرة - نقل من قصيرة المدى إلى طويلة المدى"""
    def __init__(self, dim: int):
        super().__init__()
        self.importance_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.consolidation_network = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, memories: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[float]]:
        """توحيد الذكريات وتحديد أهميتها"""
        consolidated = []
        importances = []
        
        for memory in memories:
            # حساب الأهمية
            importance = self.importance_scorer(memory).item()
            
            # توحيد الذاكرة
            if importance > 0.5:  # عتبة الأهمية
                consolidated_memory = self.consolidation_network(memory)
                consolidated.append(consolidated_memory)
                importances.append(importance)
        
        return consolidated, importances

class AdaptiveMemorySystem(nn.Module):
    """نظام الذاكرة التكيفي الشامل"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.dim
        
        # أنواع الذاكرة المختلفة
        self.working_memory = WorkingMemory(dim, config.memory.working_memory_size)
        self.episodic_memory = EpisodicMemory(dim, num_episodes=100)
        self.semantic_memory = SemanticMemory(dim, num_concepts=1000)
        self.procedural_memory = ProceduralMemory(dim, num_skills=50)
        self.consolidation = MemoryConsolidation(dim)
        
        # شبكة دمج الذاكرة
        self.memory_fusion = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
        self.current_episode_states = []
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """معالجة شاملة للذاكرة"""
        memory_info = {}
        
        # استرجاع من أنواع الذاكرة المختلفة
        working_mem = self.working_memory.retrieve(x, top_k=5)
        semantic_mem = self.semantic_memory(x)
        procedural_mem = self.procedural_memory(x)
        
        # استرجاع الحلقات المشابهة
        similar_episodes = self.episodic_memory.retrieve_similar_episodes(x, top_k=3)
        if similar_episodes:
            episodic_mem = similar_episodes[0]['representation']
            memory_info['similar_episodes_found'] = len(similar_episodes)
        else:
            episodic_mem = torch.zeros_like(x)
            memory_info['similar_episodes_found'] = 0
        
        # دمج جميع أنواع الذاكرة
        all_memories = torch.cat([
            working_mem,
            episodic_mem,
            semantic_mem,
            procedural_mem
        ], dim=-1)
        
        fused_memory = self.memory_fusion(all_memories)
        
        # تخزين في الذاكرة العاملة
        if self.config.memory.working_memory_size > 0:
            self.working_memory.store(x, importance=1.0)
        
        # تتبع الحلقة الحالية
        if self.config.memory.episodic_memory:
            self.current_episode_states.append(x.detach())
            memory_info['episode_length'] = len(self.current_episode_states)
        
        return fused_memory, memory_info
    
    def end_episode(self, context: torch.Tensor):
        """إنهاء الحلقة الحالية وتخزينها"""
        if self.current_episode_states and self.config.memory.episodic_memory:
            self.episodic_memory.store_episode(self.current_episode_states, context)
            self.current_episode_states = []
    
    def consolidate_memories(self):
        """توحيد الذكريات من قصيرة المدى إلى طويلة المدى"""
        if self.working_memory.memories:
            memories = [entry.value for entry in self.working_memory.memories]
            consolidated, importances = self.consolidation(memories)
            return len(consolidated)
        return 0
