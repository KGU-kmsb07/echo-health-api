import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "benefits")
MODEL_PATH = os.path.join(MODEL_DIR, "benefit_text_mining_model.pkl")
MODEL_VERSION = "benefit-text-mining-rules-v2"


def clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


@dataclass
class BenefitTextMiningModel:
    version: str = MODEL_VERSION
    health_keywords: list[str] = field(default_factory=lambda: [
        "건강", "금연", "건강검진", "운동", "비만", "고혈압", "당뇨", "보건소", "정신건강"
    ])
    health_context_words: list[str] = field(default_factory=lambda: [
        "검진", "진료", "의료", "보건", "질병", "예방", "상담", "운동", "정신",
        "금연", "비만", "고혈압", "당뇨", "혈압", "혈당", "치료", "관리"
    ])
    national_markers: list[str] = field(default_factory=lambda: [
        "전국", "중앙부처", "보건복지부", "국민건강보험공단", "질병관리청", "한국건강관리협회"
    ])
    national_agency_markers: list[str] = field(default_factory=lambda: [
        "보건복지부", "질병관리청", "국민건강보험공단", "건강보험심사평가원",
        "고용노동부", "여성가족부", "식품의약품안전처", "병무청", "행정안전부",
        "교육부", "국가보훈부", "환경부", "문화체육관광부", "농림축산식품부",
        "해양수산부", "중소벤처기업부", "경찰청", "소방청"
    ])
    topic_words: dict[str, list[str]] = field(default_factory=lambda: {
        "금연": ["금연", "흡연"],
        "검진": ["건강검진", "검진"],
        "운동": ["운동", "신체활동", "체력"],
        "비만": ["비만", "체중"],
        "고혈압": ["고혈압", "혈압"],
        "당뇨": ["당뇨", "혈당"],
        "정신건강": ["정신건강", "마음건강", "정신질환", "자살예방"],
        "치아": ["치아", "구강"],
    })
    condition_words: dict[str, list[str]] = field(default_factory=lambda: {
        "흡연자": ["흡연자", "금연치료", "금연클리닉", "금연상담"],
        "당뇨": ["당뇨병 환자", "당뇨환자", "당뇨병 투약자"],
        "고혈압": ["고혈압 환자", "고혈압"],
        "저소득": ["저소득", "수급자", "차상위", "의료급여"],
        "근로자": ["근로자", "직장인", "사업장", "자영업자"],
        "군인": ["군인"],
        "장애": ["장애인", "근로장애인", "장애아"],
        "보훈": ["국가유공자", "보훈", "참전", "무공수훈", "애국지사"],
        "질환자": ["환자", "질환", "치매", "암", "석면"],
        "한부모": ["한부모"],
    })
    hard_required_conditions: set[str] = field(default_factory=lambda: {"군인", "장애", "보훈", "한부모"})

    def benefit_text(self, benefit: dict[str, Any]) -> str:
        return " ".join(
            clean_text(benefit.get(key))
            for key in [
                "title",
                "provider",
                "summary",
                "target",
                "applicationMethod",
                "deadline",
                "region",
                "supportType",
                "serviceField",
                "userType",
            ]
        )

    def is_national(self, benefit: dict[str, Any]) -> bool:
        text = f"{benefit.get('region', '')} {benefit.get('provider', '')}"
        return any(marker in text for marker in self.national_markers)

    def is_national_agency(self, benefit: dict[str, Any]) -> bool:
        text = f"{benefit.get('region', '')} {benefit.get('provider', '')}"
        return any(marker in text for marker in self.national_agency_markers)

    def is_local_government(self, benefit: dict[str, Any]) -> bool:
        text = f"{benefit.get('region', '')} {benefit.get('provider', '')}"
        local_tokens = [
            "특별시", "광역시", "특별자치시", "특별자치도", "도 ", "시 ", "군 ", "구 ",
            "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
            "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
        ]
        return any(token in text for token in local_tokens)

    def is_health_related(self, benefit: dict[str, Any]) -> bool:
        focused_text = " ".join([
            clean_text(benefit.get("title")),
            clean_text(benefit.get("summary")),
            clean_text(benefit.get("target")),
            clean_text(benefit.get("serviceField")),
            clean_text(benefit.get("userType")),
            " ".join(benefit.get("tags", [])),
        ])

        strong_words = ["금연", "건강검진", "비만", "고혈압", "당뇨", "보건소", "정신건강"]
        if any(keyword in focused_text for keyword in strong_words):
            return True
        if "건강" in focused_text and any(word in focused_text for word in self.health_context_words):
            return True
        if "운동" in focused_text and any(word in focused_text for word in ["건강", "비만", "체력", "신체활동", "재활", "보건"]):
            return True
        return False

    def extract_age_constraints(self, text: str) -> dict[str, Any]:
        ranges: list[tuple[int, int]] = []
        exact: list[int] = []
        min_age: int | None = None
        max_age: int | None = None

        for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*세\s*이상(?:에서)?\s*(?:만\s*)?(\d{1,3})\s*세\s*(이하|미만|까지)", text):
            start, end = int(match.group(1)), int(match.group(2))
            if match.group(3) == "미만":
                end -= 1
            ranges.append((start, end))

        for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*(?:세)?\s*[~\-–]\s*(?:만\s*)?(\d{1,3})\s*세", text):
            ranges.append((int(match.group(1)), int(match.group(2))))

        if not ranges:
            for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*세\s*이상", text):
                value = int(match.group(1))
                min_age = value if min_age is None else max(min_age, value)

            for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*세\s*(이하|미만)", text):
                value = int(match.group(1)) - (1 if match.group(2) == "미만" else 0)
                max_age = value if max_age is None else min(max_age, value)

        for match in re.finditer(r"(?<!\d)(\d{1,3})\s*세(?!\s*(?:이상|이하|미만|초과|부터|까지|~|\-|–))", text):
            value = int(match.group(1))
            if 0 < value <= 120:
                exact.append(value)

        for match in re.finditer(r"(?<!\d)(\d{1,2})0대", text):
            decade = int(match.group(1)) * 10
            ranges.append((decade, decade + 9))

        return {"ranges": ranges, "exact": exact, "min": min_age, "max": max_age}

    def age_group_terms(self, age: int) -> list[str]:
        if age < 13:
            return ["아동", "어린이", "영유아"]
        if age < 19:
            return ["청소년", "학생"]
        if age < 35:
            return ["청년", "대학생", "사회초년생"]
        if age < 50:
            return ["중장년", "성인"]
        if age < 65:
            return ["중장년", "장년", "성인"]
        return ["노인", "어르신", "고령", "시니어", "65세"]

    def age_constraint_match(self, text: str, age: int | None) -> tuple[bool, int, list[str]]:
        if not age:
            return True, 0, []

        if age >= 19 and any(word in text for word in ["초등학교", "중학교", "고등학교", "청소년", "학교 밖"]):
            return False, -100, ["성인에게 맞지 않는 청소년 대상"]
        if age < 19 and any(word in text for word in ["청년", "대학생", "중장년", "장년", "노인", "어르신", "고령", "시니어"]):
            return False, -100, ["미성년자에게 맞지 않는 생애주기"]

        constraints = self.extract_age_constraints(text)
        ranges = constraints["ranges"]
        exact = constraints["exact"]
        min_age = constraints["min"]
        max_age = constraints["max"]

        if ranges:
            if any(start <= age <= end for start, end in ranges):
                return True, 35, [f"{age}세 연령 조건 일치"]
            return False, -100, ["연령 조건 불일치"]

        if exact:
            if age in exact:
                return True, 35, [f"{age}세 전용 조건 일치"]
            return False, -100, ["연령 조건 불일치"]

        if min_age is not None and age < min_age:
            return False, -100, ["최소 연령 조건 불일치"]
        if max_age is not None and age > max_age:
            return False, -100, ["최대 연령 조건 불일치"]
        if min_age is not None or max_age is not None:
            return True, 30, [f"{age}세 연령 조건 일치"]

        age_terms = ["아동", "어린이", "영유아", "청소년", "청년", "대학생", "중장년", "장년", "노인", "어르신", "고령", "시니어"]
        mentioned_terms = [term for term in age_terms if term in text]
        if mentioned_terms:
            if any(term in text for term in self.age_group_terms(age)):
                return True, 25, [f"{self.age_group_terms(age)[0]} 생애주기 일치"]
            return False, -100, ["생애주기 조건 불일치"]

        return True, 8, ["연령 제한 없음"]

    def extract_traits(self, text: str) -> dict[str, Any]:
        topics = [name for name, words in self.topic_words.items() if any(word in text for word in words)]
        required = [name for name, words in self.condition_words.items() if any(word in text for word in words)]

        if any(word in text for word in ["임산부", "산모", "임신", "출산", "신생아", "영유아"]):
            life_stage = "maternity_or_infant"
        elif any(word in text for word in ["초등학교", "중학교", "고등학교", "청소년", "학교 밖"]):
            life_stage = "youth"
        elif any(word in text for word in ["청년", "대학생", "사회초년생"]):
            life_stage = "young_adult"
        elif any(word in text for word in ["노인", "어르신", "고령", "시니어"]):
            life_stage = "senior"
        elif any(word in text for word in ["전 국민", "전국민", "누구나", "일반 국민"]):
            life_stage = "all"
        else:
            life_stage = "general"

        gender = "all"
        if any(word in text for word in ["임산부", "산모", "임신", "출산", "여성", "여자"]):
            gender = "female"
        elif any(word in text for word in ["남성", "남자", "전립선"]):
            gender = "male"

        return {"topics": topics, "required": required, "lifeStage": life_stage, "gender": gender}

    def score_region(self, benefit: dict[str, Any], region: str = "", sub_region: str = "") -> tuple[int, list[str]]:
        if region == "국가기관":
            if self.is_national_agency(benefit):
                return 60, ["국가기관 혜택"]
            return -100, ["국가기관 아님"]

        if not region and not sub_region:
            if self.is_national_agency(benefit):
                return 15, ["국가기관 혜택"]
            return 0, ["전국 기준"]

        if self.is_national_agency(benefit):
            return 25, ["국가기관 혜택"]
        if self.is_national(benefit):
            return 20, ["전국 혜택"]

        text = self.benefit_text(benefit)
        score = 0
        reasons: list[str] = []
        if region and region in text:
            score += 35
            reasons.append(f"{region} 지역 일치")
        if sub_region and sub_region in text:
            score += 45
            reasons.append(f"{sub_region} 세부지역 일치")
        if score == 0:
            score -= 30
            reasons.append("지역 불일치")
        return score, reasons

    def relevance_score(self, benefit: dict[str, Any], keyword: str = "") -> int:
        words = [word for word in re.split(r"\s+", keyword.strip()) if word] or self.health_keywords
        title = benefit.get("title", "")
        text = self.benefit_text(benefit)
        score = 0
        for word in words:
            if word in title:
                score += 5
            if word in benefit.get("summary", ""):
                score += 3
            if word in text:
                score += 1
        return score

    def score_benefit(self, benefit: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        text = self.benefit_text(benefit)
        traits = self.extract_traits(text)
        age_ok, age_score, age_reasons = self.age_constraint_match(text, profile.get("age"))
        region_score, region_reasons = self.score_region(benefit, profile.get("region", ""), profile.get("sub_region", ""))
        reasons = list(age_reasons)
        warnings: list[str] = []
        exclusions: list[str] = []

        if not age_ok:
            exclusions.append("연령 조건 불일치")

        gender = clean_text(profile.get("gender", ""))
        gender_score = 0
        if gender.startswith("남") and traits["gender"] == "female":
            exclusions.append("성별 조건 불일치")
        elif gender.startswith("여") and traits["gender"] == "male":
            exclusions.append("성별 조건 불일치")
        elif traits["gender"] in ("male", "female"):
            gender_score += 20
            reasons.append("성별 조건 일치")

        age = profile.get("age")
        life_score = 0
        if age:
            if age >= 19 and traits["lifeStage"] in ("youth", "maternity_or_infant"):
                exclusions.append("생애주기 조건 불일치")
            elif age < 19 and traits["lifeStage"] in ("young_adult", "senior"):
                exclusions.append("생애주기 조건 불일치")
            elif age < 35 and traits["lifeStage"] == "young_adult":
                life_score += 20
                reasons.append("청년 대상")
            elif traits["lifeStage"] in ("all", "general"):
                life_score += 8

        risk_words = set(profile.get("risks") or [])
        topic_score = 0
        if traits["topics"]:
            topic_score += min(24, len(traits["topics"]) * 6)
            reasons.extend([f"{topic} 관련" for topic in traits["topics"][:2]])
        if "hypertension" in risk_words and "고혈압" in traits["topics"]:
            topic_score += 25
            reasons.append("고혈압 위험 맞춤")
        if "diabetes" in risk_words and "당뇨" in traits["topics"]:
            topic_score += 25
            reasons.append("당뇨 위험 맞춤")
        if "obesity" in risk_words and ("비만" in traits["topics"] or "운동" in traits["topics"]):
            topic_score += 20
            reasons.append("체중 관리 맞춤")
        if "smoking" in risk_words or "흡연" in clean_text(profile.get("smoking", "")):
            if "금연" in traits["topics"]:
                topic_score += 30
                reasons.append("흡연/금연 맞춤")

        generic_penalty = 0
        known_unlocks = set()
        if "흡연" in clean_text(profile.get("smoking", "")):
            known_unlocks.add("흡연자")
        if "hypertension" in risk_words:
            known_unlocks.add("고혈압")
        if "diabetes" in risk_words:
            known_unlocks.add("당뇨")

        keyword = clean_text(profile.get("keyword", ""))
        for condition in traits["required"]:
            if condition in known_unlocks:
                reasons.append(f"{condition} 조건 일치")
                continue
            if condition in self.hard_required_conditions and condition not in keyword:
                exclusions.append(f"{condition} 전용")
            else:
                generic_penalty -= 12
                warnings.append(f"조건 확인 필요: {condition}")

        personal_score = age_score + gender_score + life_score + topic_score + generic_penalty
        keyword_score = self.relevance_score(benefit, keyword) if keyword else 0
        total_score = personal_score * 10 + region_score + keyword_score
        return {
            "eligible": not exclusions,
            "personalScore": personal_score,
            "regionScore": region_score,
            "keywordScore": keyword_score,
            "totalScore": total_score,
            "matchReasons": list(dict.fromkeys(reasons + region_reasons))[:5],
            "conditionWarnings": list(dict.fromkeys(warnings))[:4],
            "exclusions": exclusions,
            "topics": traits["topics"],
            "requiredConditions": traits["required"],
        }


def build_default_model() -> BenefitTextMiningModel:
    return BenefitTextMiningModel()


def save_model(path: str = MODEL_PATH) -> BenefitTextMiningModel:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = build_default_model()
    with open(path, "wb") as file:
        pickle.dump(model, file)
    return model


def load_model(path: str = MODEL_PATH) -> BenefitTextMiningModel:
    if not os.path.exists(path):
        return save_model(path)
    with open(path, "rb") as file:
        model = pickle.load(file)
    if getattr(model, "version", "") != MODEL_VERSION:
        return save_model(path)
    return model


_MODEL: BenefitTextMiningModel | None = None


def get_benefit_text_mining_model() -> BenefitTextMiningModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


# Explanation:
# - This file defines the rule-based text-mining "model" used by the government health benefit matcher.
# - The model is intentionally saved as a PKL file under models/benefits so the API loads a stable artifact,
#   similar to the disease PKL models already used elsewhere in the backend.
# - It is not a statistical ML model. It is a deterministic Korean text-mining model made of regex patterns
#   and curated keyword dictionaries because Gov24 benefit eligibility text is mostly unstructured prose.
# - The age extractor handles exact age ("56세"), ranges ("20~64세"), and prose ranges
#   ("9세 이상 18세 이하"). This prevents obvious mismatches such as showing a 56-year-old-only benefit
#   to a 23-year-old user.
# - The trait extractor mines health topics, life-stage words, gender-only wording, and special conditions
#   such as smoker, low-income, worker, military, disability, and veteran-related requirements.
# - Scoring is split into personalScore and regionScore by design. personalScore is weighted first because
#   the product goal is user personalization first; regionScore is then used as the second sorting layer.
# - "국가기관" is treated as a separate non-local scope, not the same as "전국". When selected, only
#   central ministries, agencies, and national public bodies such as 보건복지부, 병무청, 질병관리청,
#   국민건강보험공단, and 건강보험심사평가원 pass the region layer; local 시/도/군/구 programs are excluded.
# - matchReasons and conditionWarnings are returned to the frontend so cards can explain why an item was
#   recommended and which user-specific eligibility conditions still need confirmation.
