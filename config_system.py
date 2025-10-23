# config_system.py - نظام الإعدادات الشامل
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum

class ReasoningMode(Enum):
    """أنماط التفكير والاستدلال"""
    CHAIN_OF_THOUGHT = "chain_of_thought"  # CoT
    TREE_OF_THOUGHTS = "tree_of_thoughts"  # ToT
    GRAPH_OF_THOUGHTS = "graph_of_thoughts"  # GoT
    SELF_CONSISTENCY = "self_consistency"
    REFLEXION = "reflexion"
    STEP_BACK = "step_back"
    ANALOGICAL = "analogical"
    ABDUCTIVE = "abductive"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"

class LearningMode(Enum):
    """أنماط التعلم الذاتي"""
    IN_CONTEXT = "in_context"  # ICL
    FEW_SHOT = "few_shot"  # 1-10 examples
    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"
    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"
    CONTINUAL_LEARNING = "continual_learning"
    ACTIVE_LEARNING = "active_learning"
    SELF_SUPERVISED = "self_supervised"

class SafetyLevel(Enum):
    """مستويات الأمان"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    STRICT = "strict"

class VerbosityLevel(Enum):
    """مستويات التفصيل في التواصل"""
    MINIMAL = "minimal"
    CONCISE = "concise"
    BALANCED = "balanced"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

@dataclass
class ReasoningConfig:
    """إعدادات التفكير والاستدلال"""
    mode: ReasoningMode = ReasoningMode.CHAIN_OF_THOUGHT
    thinking_depth: int = 3  # 1-10
    reasoning_steps: int = 5
    extended_thinking: bool = False  # o1-style
    internal_monologue: bool = True
    scratchpad_enabled: bool = True
    self_reflection: bool = True
    multi_path_exploration: int = 3  # for ToT
    consistency_checks: int = 3  # for self-consistency

@dataclass
class PlanningConfig:
    """إعدادات التخطيط والتنفيذ"""
    task_decomposition: bool = True
    max_decomposition_depth: int = 5
    contingency_planning: bool = True
    parallel_execution: bool = True
    error_recovery: bool = True
    adaptive_replanning: bool = True
    progress_monitoring: bool = True
    resource_allocation: bool = True

@dataclass
class CreativityConfig:
    """إعدادات الإبداع والابتكار"""
    creativity_level: float = 0.7  # 0-1
    novelty_seeking: float = 0.6
    risk_taking: float = 0.5
    unconventional_thinking: bool = True
    divergent_thinking: bool = True
    convergent_thinking: bool = True
    lateral_thinking: bool = True
    idea_generation_rate: int = 5
    conceptual_blending: bool = True

@dataclass
class LearningConfig:
    """إعدادات التعلم الذاتي"""
    mode: LearningMode = LearningMode.IN_CONTEXT
    learning_rate: float = 0.001
    adaptation_speed: float = 0.5  # 0-1
    memory_consolidation: bool = True
    knowledge_integration: bool = True
    forgetting_rate: float = 0.1  # 0-1
    exploration_exploitation_ratio: float = 0.3  # 0=exploit, 1=explore
    meta_learning_enabled: bool = True
    transfer_learning_enabled: bool = True
    few_shot_examples: int = 5

@dataclass
class SearchConfig:
    """إعدادات البحث والاسترجاع"""
    search_depth: int = 10
    relevance_threshold: float = 0.7
    recency_bias: float = 0.3
    source_diversity: bool = True
    citation_mode: bool = True
    semantic_search: bool = True
    multi_hop_retrieval: bool = True
    fact_verification: bool = True
    source_validation: bool = True

@dataclass
class GenerationConfig:
    """إعدادات التوليد والإنشاء"""
    temperature: float = 0.7  # 0-2
    top_p: float = 0.9  # 0-1
    top_k: int = 50
    frequency_penalty: float = 0.0  # -2 to 2
    presence_penalty: float = 0.0  # -2 to 2
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    diversity_penalty: float = 0.0
    max_tokens: int = 2048
    min_tokens: int = 10
    stop_sequences: List[str] = field(default_factory=list)
    beam_search: bool = False
    num_beams: int = 1

@dataclass
class MemoryConfig:
    """إعدادات الذاكرة والسياق"""
    context_window_size: int = 8192
    working_memory_size: int = 2048
    short_term_memory_size: int = 4096
    long_term_memory_enabled: bool = True
    episodic_memory: bool = True
    semantic_memory: bool = True
    procedural_memory: bool = True
    memory_retention_period: int = 7  # days
    compression_ratio: float = 0.5
    priority_based_retention: bool = True
    retrieval_speed: str = "fast"  # fast, balanced, accurate

@dataclass
class EvaluationConfig:
    """إعدادات التقييم الذاتي"""
    self_reflection: bool = True
    self_correction: bool = True
    confidence_estimation: bool = True
    uncertainty_quantification: bool = True
    answer_verification: bool = True
    quality_scoring: bool = True
    bias_detection: bool = True
    hallucination_detection: bool = True
    verification_depth: int = 2  # 1-5
    confidence_threshold: float = 0.8
    double_checking: bool = True

@dataclass
class SafetyConfig:
    """إعدادات الأمان والأخلاقيات"""
    safety_level: SafetyLevel = SafetyLevel.HIGH
    content_filtering: bool = True
    toxicity_detection: bool = True
    bias_mitigation: bool = True
    fairness_constraints: bool = True
    privacy_protection: bool = True
    adversarial_defense: bool = True
    jailbreak_prevention: bool = True
    harm_prevention_threshold: float = 0.8
    ethical_guidelines: bool = True

@dataclass
class CommunicationConfig:
    """إعدادات التواصل والتفاعل"""
    verbosity: VerbosityLevel = VerbosityLevel.BALANCED
    formality_level: str = "professional"  # casual, professional, formal
    tone: str = "friendly"  # friendly, neutral, formal
    explanation_depth: int = 3  # 1-5
    clarification_seeking: bool = True
    proactive_suggestions: bool = True
    adaptive_communication: bool = True
    empathetic_responses: bool = True
    multi_turn_dialogue: bool = True
    question_frequency: str = "balanced"  # low, balanced, high

@dataclass
class PersonaConfig:
    """إعدادات التخصيص والشخصية"""
    persona_type: str = "assistant"  # teacher, expert, assistant, creative
    expertise_domain: List[str] = field(default_factory=list)
    communication_style: str = "adaptive"
    cultural_adaptation: bool = True
    age_appropriate: bool = True
    personality_traits: List[str] = field(default_factory=lambda: ["helpful", "curious"])
    custom_instructions: str = ""
    language_preference: str = "auto"

@dataclass
class PerformanceConfig:
    """إعدادات الأداء والكفاءة"""
    response_speed: str = "balanced"  # fast, balanced, accurate
    latency_tolerance: float = 1.0  # seconds
    batch_processing: bool = True
    streaming_mode: bool = True
    caching_enabled: bool = True
    compression_level: int = 1  # 0-9
    parallelization_degree: int = 4
    memory_management: str = "auto"  # auto, conservative, aggressive
    energy_efficiency: bool = True

@dataclass
class IntegrationConfig:
    """إعدادات التكامل والتوسع"""
    rag_enabled: bool = True
    knowledge_graph: bool = False
    external_database: bool = False
    plugin_system: bool = True
    webhook_integration: bool = False
    api_rate_limit: int = 100  # requests per minute
    service_timeout: int = 30  # seconds
    fallback_strategies: bool = True

@dataclass
class AnalysisConfig:
    """إعدادات التحليل والبيانات"""
    data_analysis: bool = True
    statistical_reasoning: bool = True
    pattern_recognition: bool = True
    anomaly_detection: bool = True
    trend_analysis: bool = True
    predictive_modeling: bool = False
    causal_inference: bool = True
    analysis_depth: int = 3  # 1-5
    statistical_significance: float = 0.05
    visualization_enabled: bool = True

@dataclass
class LanguageConfig:
    """إعدادات اللغة والترجمة"""
    primary_language: str = "auto"
    multilingual: bool = True
    translation_quality: str = "balanced"  # fast, balanced, accurate
    cultural_localization: bool = True
    formality_in_translation: str = "preserve"
    dialect_adaptation: bool = False
    code_switching: bool = True

@dataclass
class EducationConfig:
    """إعدادات التعليم والتدريب"""
    teaching_style: str = "socratic"  # socratic, direct, scaffolding
    difficulty_level: str = "adaptive"  # beginner, intermediate, advanced, adaptive
    hints_frequency: str = "on_request"  # never, on_request, proactive
    worked_examples: bool = True
    practice_problems: bool = True
    feedback_loops: bool = True
    assessment_mode: bool = False

@dataclass
class MonitoringConfig:
    """إعدادات المراقبة والتتبع"""
    usage_analytics: bool = True
    performance_metrics: bool = True
    error_logging: bool = True
    user_feedback_collection: bool = True
    quality_monitoring: bool = True
    logging_level: str = "info"  # disabled, basic, info, detailed, debug
    telemetry_enabled: bool = True
    privacy_mode: bool = False

@dataclass
class CosmosAdvancedConfig:
    """التكوين الشامل لنموذج Cosmos المتقدم"""
    # Model Architecture
    dim: int = 1024
    n_layers: int = 8
    head_dim: int = 128
    hidden_dim: int = 4096
    n_heads: int = 16
    n_kv_heads: int = 4
    norm_eps: float = 1e-5
    vocab_size: int = 32000
    rope_theta: float = 10000.0
    window_size: int = 4096
    dropout: float = 0.1
    activation_dropout: float = 0.0
    attention_dropout: float = 0.0
    gradient_checkpointing: bool = True
    max_sequence_length: int = 8192
    lora_rank: Optional[int] = 4
    diffusion: bool = True
    latent_dim: int = 512
    
    # Advanced Capabilities
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    creativity: CreativityConfig = field(default_factory=CreativityConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    education: EducationConfig = field(default_factory=EducationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل التكوين إلى قاموس"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                # Convert nested dataclass
                result[key] = value
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    def save(self, filepath: str):
        """حفظ التكوين إلى ملف JSON"""
        config_dict = self.to_dict()
        # Convert enums in nested configs
        for key, value in config_dict.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(v, 'value'):
                        value[k] = v.value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str):
        """تحميل التكوين من ملف JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested configs
        reasoning = ReasoningConfig(**config_dict.pop('reasoning', {}))
        planning = PlanningConfig(**config_dict.pop('planning', {}))
        creativity = CreativityConfig(**config_dict.pop('creativity', {}))
        learning = LearningConfig(**config_dict.pop('learning', {}))
        search = SearchConfig(**config_dict.pop('search', {}))
        generation = GenerationConfig(**config_dict.pop('generation', {}))
        memory = MemoryConfig(**config_dict.pop('memory', {}))
        evaluation = EvaluationConfig(**config_dict.pop('evaluation', {}))
        safety = SafetyConfig(**config_dict.pop('safety', {}))
        communication = CommunicationConfig(**config_dict.pop('communication', {}))
        persona = PersonaConfig(**config_dict.pop('persona', {}))
        performance = PerformanceConfig(**config_dict.pop('performance', {}))
        integration = IntegrationConfig(**config_dict.pop('integration', {}))
        analysis = AnalysisConfig(**config_dict.pop('analysis', {}))
        language = LanguageConfig(**config_dict.pop('language', {}))
        education = EducationConfig(**config_dict.pop('education', {}))
        monitoring = MonitoringConfig(**config_dict.pop('monitoring', {}))
        
        return cls(
            reasoning=reasoning,
            planning=planning,
            creativity=creativity,
            learning=learning,
            search=search,
            generation=generation,
            memory=memory,
            evaluation=evaluation,
            safety=safety,
            communication=communication,
            persona=persona,
            performance=performance,
            integration=integration,
            analysis=analysis,
            language=language,
            education=education,
            monitoring=monitoring,
            **config_dict
        )
    
    def get_preset(self, preset_name: str):
        """تطبيق إعدادات مسبقة"""
        presets = {
            "creative": self._creative_preset,
            "analytical": self._analytical_preset,
            "educational": self._educational_preset,
            "safe": self._safe_preset,
            "performance": self._performance_preset,
        }
        if preset_name in presets:
            presets[preset_name]()
    
    def _creative_preset(self):
        self.creativity.creativity_level = 0.9
        self.creativity.novelty_seeking = 0.8
        self.generation.temperature = 1.2
        self.reasoning.mode = ReasoningMode.TREE_OF_THOUGHTS
    
    def _analytical_preset(self):
        self.reasoning.mode = ReasoningMode.CHAIN_OF_THOUGHT
        self.analysis.analysis_depth = 5
        self.evaluation.verification_depth = 3
        self.generation.temperature = 0.3
    
    def _educational_preset(self):
        self.education.teaching_style = "socratic"
        self.communication.explanation_depth = 5
        self.communication.verbosity = VerbosityLevel.DETAILED
    
    def _safe_preset(self):
        self.safety.safety_level = SafetyLevel.STRICT
        self.safety.content_filtering = True
        self.safety.harm_prevention_threshold = 0.95
    
    def _performance_preset(self):
        self.performance.response_speed = "fast"
        self.performance.caching_enabled = True
        self.performance.parallelization_degree = 8
