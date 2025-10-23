# __init__.py - ملف التهيئة لحزمة Cosmos Advanced

"""
Cosmos Advanced AI Model
========================

نموذج ذكاء اصطناعي شامل مع قدرات متقدمة في:
- التفكير والاستدلال
- الذاكرة التكيفية
- التعلم الذاتي
- الأمان والأخلاقيات
- التقييم الذاتي

مثال الاستخدام:
-----------
>>> from cosmos_advanced import CosmosAdvancedModel, CosmosAdvancedConfig
>>> config = CosmosAdvancedConfig()
>>> model = CosmosAdvancedModel(config)
>>> # استخدام النموذج...
"""

__version__ = "3.0.0"
__author__ = "Cosmos AI Team"
__license__ = "MIT"

# استيراد المكونات الرئيسية
from .core.cosmos_model_advanced import CosmosAdvancedModel
from .core.config_system import (
    CosmosAdvancedConfig,
    ReasoningConfig,
    PlanningConfig,
    CreativityConfig,
    LearningConfig,
    SearchConfig,
    GenerationConfig,
    MemoryConfig,
    EvaluationConfig,
    SafetyConfig,
    CommunicationConfig,
    PersonaConfig,
    PerformanceConfig,
    IntegrationConfig,
    AnalysisConfig,
    LanguageConfig,
    EducationConfig,
    MonitoringConfig,
    ReasoningMode,
    LearningMode,
    SafetyLevel,
    VerbosityLevel
)

from .reasoning_engine import ReasoningEngine
from .memory_system import AdaptiveMemorySystem
from .learning_engine import AdaptiveLearningEngine
from .safety_module import ComprehensiveSafetySystem
from .evaluation_module import SelfEvaluationSystem

# تحديد ما يتم تصديره عند استخدام from cosmos_advanced import *
__all__ = [
    # النموذج الرئيسي
    'CosmosAdvancedModel',
    
    # التكوين
    'CosmosAdvancedConfig',
    'ReasoningConfig',
    'PlanningConfig',
    'CreativityConfig',
    'LearningConfig',
    'SearchConfig',
    'GenerationConfig',
    'MemoryConfig',
    'EvaluationConfig',
    'SafetyConfig',
    'CommunicationConfig',
    'PersonaConfig',
    'PerformanceConfig',
    'IntegrationConfig',
    'AnalysisConfig',
    'LanguageConfig',
    'EducationConfig',
    'MonitoringConfig',
    
    # الأنماط
    'ReasoningMode',
    'LearningMode',
    'SafetyLevel',
    'VerbosityLevel',
    
    # المحركات
    'ReasoningEngine',
    'AdaptiveMemorySystem',
    'AdaptiveLearningEngine',
    'ComprehensiveSafetySystem',
    'SelfEvaluationSystem',
]

# معلومات عن الحزمة
def get_version():
    """إرجاع رقم الإصدار"""
    return __version__

def get_info():
    """إرجاع معلومات عن الحزمة"""
    return {
        'name': 'Cosmos Advanced AI Model',
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'capabilities': [
            'Advanced Reasoning (10+ modes)',
            'Adaptive Memory System',
            'Self-Learning Engine',
            'Comprehensive Safety',
            'Self-Evaluation',
            'Multimodal Ready'
        ]
    }

print(f"✨ Cosmos Advanced AI Model v{__version__} loaded successfully!")
