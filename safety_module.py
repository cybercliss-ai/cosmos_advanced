# safety_module.py - وحدة الأمان والأخلاقيات
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import re

class ContentFilter(nn.Module):
    """تصفية المحتوى الضار"""
    def __init__(self, dim: int):
        super().__init__()
        self.harmful_detector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # فئات المحتوى الضار
        self.category_classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 7)  # 7 categories
        )
        
        self.harm_categories = [
            'violence', 'hate_speech', 'sexual_content', 
            'self_harm', 'illegal_activity', 'misinformation',
            'privacy_violation'
        ]
        
    def forward(self, x: torch.Tensor) -> Tuple[bool, Dict[str, float]]:
        """فحص المحتوى للأمان"""
        # الكشف العام عن الضرر
        harmful_output = self.harmful_detector(x)
        if harmful_output.dim() > 0:
            harmfulness = harmful_output.mean().item()
        else:
            harmfulness = harmful_output.item()
        
        # تصنيف نوع الضرر
        category_scores = F.softmax(self.category_classifier(x), dim=-1)
        
        # التأكد من أن category_scores له shape صحيح
        if category_scores.dim() > 1:
            category_scores = category_scores.mean(dim=0)  # averaging over batch
        
        harm_breakdown = {}
        for i, category in enumerate(self.harm_categories):
            if i < len(category_scores):
                harm_breakdown[category] = category_scores[i].item()
            else:
                harm_breakdown[category] = 0.0
        
        is_safe = harmfulness < 0.5
        
        return is_safe, {
            'overall_harmfulness': harmfulness,
            'categories': harm_breakdown,
            'max_category': self.harm_categories[category_scores.argmax().item()]
        }

class ToxicityDetector(nn.Module):
    """كشف السمية في النص"""
    def __init__(self, dim: int):
        super().__init__()
        self.toxicity_scorer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 6)  # 6 toxicity types
        )
        
        self.toxicity_types = [
            'toxic', 'severe_toxic', 'obscene',
            'threat', 'insult', 'identity_hate'
        ]
        
    def forward(self, x: torch.Tensor) -> Dict[str, float]:
        """تحليل مستويات السمية"""
        scores = torch.sigmoid(self.toxicity_scorer(x))
        
        # التأكد من أن scores له shape صحيح
        if scores.dim() > 1:
            scores = scores.mean(dim=0)  # averaging over batch
        
        toxicity_scores = {}
        for i, tox_type in enumerate(self.toxicity_types):
            if i < len(scores):
                toxicity_scores[tox_type] = scores[i].item()
            else:
                toxicity_scores[tox_type] = 0.0
        
        toxicity_scores['overall'] = scores.max().item()
        toxicity_scores['is_toxic'] = toxicity_scores['overall'] > 0.5
        
        return toxicity_scores

class BiasDetector(nn.Module):
    """كشف التحيز في المحتوى"""
    def __init__(self, dim: int):
        super().__init__()
        self.bias_analyzer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 5)  # 5 bias types
        )
        
        self.bias_types = [
            'gender_bias', 'racial_bias', 'age_bias',
            'religious_bias', 'political_bias'
        ]
        
        # مخفف التحيز
        self.debiaser = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """كشف وتخفيف التحيز"""
        bias_scores = torch.sigmoid(self.bias_analyzer(x))
        
        # التأكد من أن bias_scores له shape صحيح
        if bias_scores.dim() > 1:
            bias_scores = bias_scores.mean(dim=0)  # averaging over batch
        
        bias_report = {}
        for i, bias_type in enumerate(self.bias_types):
            if i < len(bias_scores):
                bias_report[bias_type] = bias_scores[i].item()
            else:
                bias_report[bias_type] = 0.0
        
        bias_report['overall_bias'] = bias_scores.mean().item()
        bias_report['max_bias_type'] = self.bias_types[bias_scores.argmax().item()]
        
        # تخفيف التحيز إذا كان مرتفعاً
        if bias_report['overall_bias'] > 0.5:
            debiased_x = self.debiaser(x)
            bias_report['debiasing_applied'] = True
            return debiased_x, bias_report
        
        bias_report['debiasing_applied'] = False
        return x, bias_report

class HallucinationDetector(nn.Module):
    """كشف الهلوسة (المعلومات غير الحقيقية)"""
    def __init__(self, dim: int):
        super().__init__()
        self.confidence_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.factuality_checker = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        self.consistency_checker = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """فحص احتمالية الهلوسة"""
        confidence_output = self.confidence_estimator(x)
        if confidence_output.dim() > 0:
            confidence = confidence_output.mean().item()
        else:
            confidence = confidence_output.item()
            
        factuality_output = self.factuality_checker(x)
        if factuality_output.dim() > 0:
            factuality = factuality_output.mean().item()
        else:
            factuality = factuality_output.item()
        
        result = {
            'confidence': confidence,
            'factuality': factuality,
            'likely_hallucination': (confidence > 0.7 and factuality < 0.3)
        }
        
        # فحص الاتساق مع السياق
        if context is not None:
            combined = torch.cat([x, context], dim=-1)
            consistency_output = self.consistency_checker(combined)
            if consistency_output.dim() > 0:
                consistency = consistency_output.mean().item()
            else:
                consistency = consistency_output.item()
            result['consistency_with_context'] = consistency
            result['likely_hallucination'] = result['likely_hallucination'] or (consistency < 0.4)
        
        return result

class PrivacyProtector(nn.Module):
    """حماية الخصوصية"""
    def __init__(self, dim: int):
        super().__init__()
        self.pii_detector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 10)  # 10 types of PII
        )
        
        self.pii_types = [
            'name', 'email', 'phone', 'address', 'ssn',
            'credit_card', 'ip_address', 'medical_info',
            'financial_info', 'credentials'
        ]
        
        self.anonymizer = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """كشف وحماية المعلومات الشخصية"""
        pii_scores = torch.sigmoid(self.pii_detector(x))
        
        # التأكد من أن pii_scores له shape صحيح
        if pii_scores.dim() > 1:
            pii_scores = pii_scores.mean(dim=0)  # averaging over batch
        
        pii_detected = {}
        for i, pii_type in enumerate(self.pii_types):
            if i < len(pii_scores):
                pii_detected[pii_type] = pii_scores[i].item()
            else:
                pii_detected[pii_type] = 0.0
        
        max_pii_score = pii_scores.max().item()
        
        report = {
            'pii_types_detected': pii_detected,
            'contains_pii': max_pii_score > 0.6,
            'max_pii_type': self.pii_types[pii_scores.argmax().item()],
            'anonymization_applied': False
        }
        
        # تطبيق التيك إذا تم اكتشاف PII
        if report['contains_pii']:
            anonymized_x = self.anonymizer(x)
            report['anonymization_applied'] = True
            return anonymized_x, report
        
        return x, report

class AdversarialDefense(nn.Module):
    """الدفاع ضد الهجمات الخصمية"""
    def __init__(self, dim: int):
        super().__init__()
        self.attack_detector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        self.perturbation_cleaner = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """كشف وتنظيف الهجمات الخصمية"""
        attack_score = self.attack_detector(x).item()
        
        report = {
            'attack_probability': attack_score,
            'likely_adversarial': attack_score > 0.7,
            'defense_applied': False
        }
        
        if report['likely_adversarial']:
            cleaned_x = self.perturbation_cleaner(x)
            # دمج مع المدخل الأصلي
            output = 0.7 * cleaned_x + 0.3 * x
            report['defense_applied'] = True
            return output, report
        
        return x, report

class JailbreakPrevention(nn.Module):
    """منع محاولات الاختراق"""
    def __init__(self, dim: int):
        super().__init__()
        self.jailbreak_detector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 5)  # 5 jailbreak patterns
        )
        
        self.jailbreak_patterns = [
            'role_play_attack', 'instruction_override',
            'context_confusion', 'encoding_bypass',
            'prompt_injection'
        ]
        
    def forward(self, x: torch.Tensor) -> Tuple[bool, Dict[str, float]]:
        """كشف محاولات الاختراق"""
        pattern_scores = torch.sigmoid(self.jailbreak_detector(x))
        
        # التأكد من أن pattern_scores هو 1D tensor
        if pattern_scores.dim() > 1:
            pattern_scores = pattern_scores.squeeze()
        
        jailbreak_analysis = {}
        for i, pattern in enumerate(self.jailbreak_patterns):
            if i < pattern_scores.shape[0]:  # التأكد من وجود العنصر
                score = pattern_scores[i]
                jailbreak_analysis[pattern] = score.mean().item() if score.numel() > 1 else score.item()
            else:
                jailbreak_analysis[pattern] = 0.0  # قيمة افتراضية
        
        max_score = pattern_scores.max().item()
        jailbreak_analysis['overall_jailbreak_risk'] = max_score
        
        # التأكد من أن argmax_index ضمن الحدود
        if pattern_scores.shape[0] > 0:
            detected_idx = pattern_scores.argmax().item()
            detected_idx = min(detected_idx, len(self.jailbreak_patterns) - 1)
            jailbreak_analysis['detected_pattern'] = self.jailbreak_patterns[detected_idx]
        else:
            jailbreak_analysis['detected_pattern'] = "unknown"
        
        jailbreak_analysis['is_jailbreak_attempt'] = max_score > 0.65
        
        is_safe = not jailbreak_analysis['is_jailbreak_attempt']
        
        return is_safe, jailbreak_analysis

class EthicalGuardian(nn.Module):
    """الحارس الأخلاقي - لضمان الامتثال للمعايير الأخلاقية"""
    def __init__(self, dim: int):
        super().__init__()
        self.ethical_principles = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 6)  # 6 ethical principles
        )
        
        self.principles = [
            'beneficence',  # النفع
            'non_maleficence',  # عدم الضرر
            'autonomy',  # الاستقلالية
            'justice',  # العدالة
            'transparency',  # الشفافية
            'accountability'  # المساءلة
        ]
        
    def forward(self, x: torch.Tensor) -> Dict[str, float]:
        """تقييم مدى الامتثال للمبادئ الأخلاقية"""
        scores = torch.sigmoid(self.ethical_principles(x))
        
        # التأكد من أن scores هو 1D tensor
        if scores.dim() > 1:
            scores = scores.squeeze()
        
        ethical_assessment = {}
        for i, principle in enumerate(self.principles):
            if i < scores.shape[0]:  # التأكد من وجود العنصر
                score = scores[i]
                ethical_assessment[principle] = score.mean().item() if score.numel() > 1 else score.item()
            else:
                ethical_assessment[principle] = 0.5  # قيمة افتراضية
        
        ethical_assessment['overall_ethical_score'] = scores.mean().item()
        ethical_assessment['is_ethical'] = ethical_assessment['overall_ethical_score'] > 0.6
        
        if scores.shape[0] > 0:
            weakest_idx = scores.argmin().item()
            weakest_idx = min(weakest_idx, len(self.principles) - 1)
            ethical_assessment['weakest_principle'] = self.principles[weakest_idx]
        else:
            ethical_assessment['weakest_principle'] = "unknown"
        
        return ethical_assessment

class ComprehensiveSafetySystem(nn.Module):
    """نظام الأمان الشامل"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.dim
        
        # جميع وحدات الأمان
        self.content_filter = ContentFilter(dim)
        self.toxicity_detector = ToxicityDetector(dim)
        self.bias_detector = BiasDetector(dim)
        self.hallucination_detector = HallucinationDetector(dim)
        self.privacy_protector = PrivacyProtector(dim)
        self.adversarial_defense = AdversarialDefense(dim)
        self.jailbreak_prevention = JailbreakPrevention(dim)
        self.ethical_guardian = EthicalGuardian(dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
        """فحص شامل للأمان"""
        safety_report = {
            'checks_performed': [],
            'warnings': [],
            'interventions': []
        }
        
        output = x
        overall_safe = True
        
        # 1. تصفية المحتوى
        if self.config.safety.content_filtering:
            is_safe, content_report = self.content_filter(output)
            safety_report['content_filter'] = content_report
            safety_report['checks_performed'].append('content_filtering')
            if not is_safe:
                overall_safe = False
                safety_report['warnings'].append('Harmful content detected')
        
        # 2. كشف السمية
        if self.config.safety.toxicity_detection:
            toxicity_report = self.toxicity_detector(output)
            safety_report['toxicity'] = toxicity_report
            safety_report['checks_performed'].append('toxicity_detection')
            if toxicity_report['is_toxic']:
                overall_safe = False
                safety_report['warnings'].append('Toxic content detected')
        
        # 3. كشف وتخفيف التحيز
        if self.config.safety.bias_mitigation:
            output, bias_report = self.bias_detector(output)
            safety_report['bias'] = bias_report
            safety_report['checks_performed'].append('bias_detection')
            if bias_report['debiasing_applied']:
                safety_report['interventions'].append('Bias mitigation applied')
        
        # 4. كشف الهلوسة
        hallucination_report = self.hallucination_detector(output, context)
        safety_report['hallucination'] = hallucination_report
        safety_report['checks_performed'].append('hallucination_detection')
        if hallucination_report['likely_hallucination']:
            safety_report['warnings'].append('Possible hallucination detected')
        
        # 5. حماية الخصوصية
        if self.config.safety.privacy_protection:
            output, privacy_report = self.privacy_protector(output)
            safety_report['privacy'] = privacy_report
            safety_report['checks_performed'].append('privacy_protection')
            if privacy_report['anonymization_applied']:
                safety_report['interventions'].append('Privacy protection applied')
        
        # 6. الدفاع ضد الهجمات
        if self.config.safety.adversarial_defense:
            output, defense_report = self.adversarial_defense(output)
            safety_report['adversarial'] = defense_report
            safety_report['checks_performed'].append('adversarial_defense')
            if defense_report['defense_applied']:
                safety_report['interventions'].append('Adversarial defense applied')
        
        # 7. منع الاختراق
        if self.config.safety.jailbreak_prevention:
            is_safe, jailbreak_report = self.jailbreak_prevention(output)
            safety_report['jailbreak'] = jailbreak_report
            safety_report['checks_performed'].append('jailbreak_prevention')
            if not is_safe:
                overall_safe = False
                safety_report['warnings'].append('Jailbreak attempt detected')
        
        # 8. التقييم الأخلاقي
        if self.config.safety.ethical_guidelines:
            ethical_report = self.ethical_guardian(output)
            safety_report['ethics'] = ethical_report
            safety_report['checks_performed'].append('ethical_evaluation')
            if not ethical_report['is_ethical']:
                safety_report['warnings'].append(f"Ethical concern: {ethical_report['weakest_principle']}")
        
        # التقييم النهائي
        safety_report['overall_safe'] = overall_safe
        safety_report['safety_level'] = self.config.safety.safety_level.value
        safety_report['num_warnings'] = len(safety_report['warnings'])
        safety_report['num_interventions'] = len(safety_report['interventions'])
        
        return output, overall_safe, safety_report
