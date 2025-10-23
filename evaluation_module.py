# evaluation_module.py - وحدة التقييم الذاتي
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math

class ConfidenceEstimator(nn.Module):
    """تقدير الثقة في الإجابة"""
    def __init__(self, dim: int):
        super().__init__()
        self.confidence_network = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # تقدير عدم اليقين
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Softplus()  # دائماً إيجابي
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, float]:
        """تقدير الثقة وعدم اليقين"""
        confidence = self.confidence_network(x).item()
        uncertainty = self.uncertainty_estimator(x).item()
        
        return {
            'confidence': confidence,
            'uncertainty': uncertainty,
            'reliability': confidence * (1 - uncertainty),
            'confidence_level': self._categorize_confidence(confidence)
        }
    
    def _categorize_confidence(self, confidence: float) -> str:
        if confidence >= 0.9:
            return 'very_high'
        elif confidence >= 0.7:
            return 'high'
        elif confidence >= 0.5:
            return 'medium'
        elif confidence >= 0.3:
            return 'low'
        else:
            return 'very_low'

class QualityScorer(nn.Module):
    """تقييم جودة الإخراج"""
    def __init__(self, dim: int):
        super().__init__()
        self.quality_dimensions = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 8)  # 8 quality dimensions
        )
        
        self.dimensions = [
            'relevance', 'coherence', 'accuracy', 'completeness',
            'clarity', 'conciseness', 'informativeness', 'helpfulness'
        ]
        
    def forward(self, x: torch.Tensor, query: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """تقييم متعدد الأبعاد للجودة"""
        # إذا كان هناك استعلام، دمجه
        if query is not None:
            x_eval = torch.cat([x, query], dim=-1)
            x_eval = nn.Linear(x_eval.size(-1), self.quality_dimensions[0].in_features).to(x.device)(x_eval)
        else:
            x_eval = x
        
        scores = torch.sigmoid(self.quality_dimensions(x_eval))
        
        quality_report = {}
        for i, dimension in enumerate(self.dimensions):
            quality_report[dimension] = scores[i].item()
        
        quality_report['overall_quality'] = scores.mean().item()
        quality_report['quality_grade'] = self._grade_quality(quality_report['overall_quality'])
        quality_report['strengths'] = [self.dimensions[i] for i in scores.topk(3).indices.tolist()]
        quality_report['weaknesses'] = [self.dimensions[i] for i in scores.topk(3, largest=False).indices.tolist()]
        
        return quality_report
    
    def _grade_quality(self, score: float) -> str:
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.6:
            return 'C'
        elif score >= 0.5:
            return 'D'
        else:
            return 'F'

class SelfCorrectionModule(nn.Module):
    """وحدة التصحيح الذاتي"""
    def __init__(self, dim: int):
        super().__init__()
        self.error_detector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 5)  # 5 types of errors
        )
        
        self.error_types = [
            'logical_error', 'factual_error', 'consistency_error',
            'relevance_error', 'formatting_error'
        ]
        
        self.corrector = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x: torch.Tensor, original_query: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """كشف وتصحيح الأخطاء"""
        error_scores = torch.sigmoid(self.error_detector(x))
        
        error_report = {}
        for i, error_type in enumerate(self.error_types):
            error_report[error_type] = error_scores[i].item()
        
        max_error = error_scores.max().item()
        error_report['max_error_type'] = self.error_types[error_scores.argmax().item()]
        error_report['needs_correction'] = max_error > 0.5
        
        if error_report['needs_correction']:
            # تطبيق التصحيح
            if original_query is not None:
                combined = torch.cat([x, original_query], dim=-1)
            else:
                combined = torch.cat([x, x], dim=-1)
            
            corrected = self.corrector(combined)
            error_report['correction_applied'] = True
            return corrected, error_report
        
        error_report['correction_applied'] = False
        return x, error_report

class AnswerVerifier(nn.Module):
    """التحقق من الإجابات"""
    def __init__(self, dim: int):
        super().__init__()
        self.verification_network = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        self.consistency_checker = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        answer: torch.Tensor, 
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """التحقق من صحة الإجابة"""
        # التحقق من التوافق مع الاستعلام
        combined = torch.cat([answer, query], dim=-1)
        verification_score = self.verification_network(combined).item()
        
        result = {
            'verification_score': verification_score,
            'answer_matches_query': verification_score > 0.7
        }
        
        # إذا كان هناك سياق، التحقق من الاتساق
        if context is not None:
            full_context = torch.cat([answer, query, context], dim=-1)
            consistency_score = self.consistency_checker(full_context).item()
            result['consistency_score'] = consistency_score
            result['is_consistent'] = consistency_score > 0.6
        
        result['overall_verified'] = result['answer_matches_query']
        if 'is_consistent' in result:
            result['overall_verified'] = result['overall_verified'] and result['is_consistent']
        
        return result

class FactualityChecker(nn.Module):
    """فاحص الحقائق"""
    def __init__(self, dim: int):
        super().__init__()
        self.factual_scorer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        self.claim_extractor = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """فحص الحقائق في المحتوى"""
        factuality_score = self.factual_scorer(x).item()
        
        return {
            'factuality_score': factuality_score,
            'likely_factual': factuality_score > 0.7,
            'confidence_in_facts': factuality_score,
            'verification_needed': factuality_score < 0.5
        }

class CompletenessChecker(nn.Module):
    """فاحص الاكتمال"""
    def __init__(self, dim: int):
        super().__init__()
        self.completeness_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, answer: torch.Tensor, query: torch.Tensor) -> Dict[str, float]:
        """تقييم اكتمال الإجابة"""
        combined = torch.cat([answer, query], dim=-1)
        completeness = self.completeness_scorer(combined).item()
        
        return {
            'completeness_score': completeness,
            'is_complete': completeness > 0.75,
            'missing_information': 1.0 - completeness
        }

class SelfEvaluationSystem(nn.Module):
    """نظام التقييم الذاتي الشامل"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.dim
        
        # جميع وحدات التقييم
        self.confidence_estimator = ConfidenceEstimator(dim)
        self.quality_scorer = QualityScorer(dim)
        self.self_correction = SelfCorrectionModule(dim)
        self.answer_verifier = AnswerVerifier(dim)
        self.factuality_checker = FactualityChecker(dim)
        self.completeness_checker = CompletenessChecker(dim)
        
        # عداد التكرار
        self.iteration_count = 0
        self.max_iterations = config.evaluation.verification_depth
        
    def forward(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """تقييم شامل مع تصحيح تكراري"""
        evaluation_report = {
            'iterations': [],
            'improvements_made': 0
        }
        
        current_output = x
        
        for iteration in range(self.max_iterations):
            iter_report = {'iteration': iteration + 1}
            
            # 1. تقدير الثقة
            if self.config.evaluation.confidence_estimation:
                confidence = self.confidence_estimator(current_output)
                iter_report['confidence'] = confidence
            
            # 2. تقييم الجودة
            if self.config.evaluation.quality_scoring:
                quality = self.quality_scorer(current_output, query)
                iter_report['quality'] = quality
            
            # 3. فحص الحقائق
            factuality = self.factuality_checker(current_output)
            iter_report['factuality'] = factuality
            
            # 4. فحص الاكتمال
            completeness = self.completeness_checker(current_output, query)
            iter_report['completeness'] = completeness
            
            # 5. التحقق من الإجابة
            if self.config.evaluation.answer_verification:
                verification = self.answer_verifier(current_output, query, context)
                iter_report['verification'] = verification
            
            # 6. التصحيح الذاتي إذا لزم الأمر
            if self.config.evaluation.self_correction:
                corrected, correction_report = self.self_correction(current_output, query)
                iter_report['correction'] = correction_report
                
                if correction_report['correction_applied']:
                    current_output = corrected
                    evaluation_report['improvements_made'] += 1
                    iter_report['output_updated'] = True
                else:
                    iter_report['output_updated'] = False
                    # إذا لم يتم التصحيح، يمكن التوقف
                    if iteration > 0:
                        break
            
            evaluation_report['iterations'].append(iter_report)
            
            # التوقف إذا وصلنا لجودة عالية
            if 'quality' in iter_report and iter_report['quality']['overall_quality'] > 0.9:
                break
        
        # التقييم النهائي
        final_confidence = self.confidence_estimator(current_output)
        final_quality = self.quality_scorer(current_output, query)
        
        evaluation_report['final_assessment'] = {
            'confidence': final_confidence,
            'quality': final_quality,
            'total_iterations': len(evaluation_report['iterations']),
            'improvements_made': evaluation_report['improvements_made'],
            'ready_for_output': (
                final_confidence['confidence'] >= self.config.evaluation.confidence_threshold and
                final_quality['overall_quality'] >= 0.6
            )
        }
        
        return current_output, evaluation_report
