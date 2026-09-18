"""L3 工具 - 安全检测后端抽象层 + 本地默认实现。"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    """检测结果。"""

    risk_level: str
    findings: list[dict] = field(default_factory=list)
    message: str = ""


_PII_PATTERNS: list[tuple[str, str]] = [
    ("phone", r"1[3-9]\d{9}"),
    ("id_card", r"\d{17}[\dXx]"),
    ("email", r"[a-zA-Z0-9.+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    ("bank_card", r"\b\d{16,19}\b"),
    ("ip_address", r"(?<!\d)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?!\d)"),
]

_VIOLATION_WORDS: list[str] = [
    "笨蛋", "蠢货", "白痴", "废物", "滚蛋", "去死",
    "fuck", "shit", "damn", "bitch", "asshole",
    "脑残", "弱智", "变态", "神经病",
    "色情", "黄色", "裸体", "裸照",
    "赌博", "毒品", "大麻", "可卡因",
    "诈骗", "洗钱", "贿赂",
    "反动", "颠覆", "暴乱",
    "黑客", "攻击", "入侵", "病毒", "木马",
    "枪支", "弹药", "爆炸", "炸弹",
]

_SQL_INJECTION_PATTERNS: list[str] = [
    r"'\s*OR\s*'1'\s*=\s*'1",
    r"UNION\s+SELECT",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM",
    r"INSERT\s+INTO",
    r"--\s*$",
    r";\s*DROP",
]

_PROMPT_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(previous|above|all)\s+instructions",
    r"disregard\s+(previous|all)\s+instructions",
    r"system\s*:",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"forget\s+(everything|all\s+rules)",
    r"override\s+(system|safety)\s+prompt",
]

_SENSITIVE_FILE_PATTERNS: list[str] = [
    r"\.env$",
    r"\.env\.",
    r"\.ssh[/]",
    r"\.aws[/]",
    r"\.git[/]",
    r"\.npmrc$",
    r"\.pypirc$",
    r"id_rsa",
    r"id_dsa",
    r"id_ed25519",
    r"\.pem$",
    r"\.key$",
    r"\.pfx$",
    r"\.keystore$",
    r"credentials",
    r"secrets?\.(yml|yaml|json|toml|ini|conf)",
]

_SECRET_PATTERNS: list[tuple[str, str]] = [
    ("aws_access_key", r"AKIA[0-9A-Z]{16}"),
    ("private_key_header", r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----"),
    ("password_assignment", r"(?i)(password|passwd|pwd)\s*[=:]\s*\S+"),
    ("token_assignment", r"(?i)(token|api_key|apikey|secret_key)\s*[=:]\s*\S+"),
    ("google_api_key", r"AIza[0-9A-Za-z\-_]{35}"),
    ("github_token", r"gh[pousr]_[A-Za-z0-9]{36}"),
]


def _mask(value: str, keep_front: int = 2, keep_end: int = 2) -> str:
    """脱敏处理：保留前 N 和后 N 个字符，中间用 * 替代。"""
    if len(value) <= keep_front + keep_end:
        return "*" * len(value)
    return f"{value[:keep_front]}{'*' * (len(value) - keep_front - keep_end)}{value[-keep_end:]}"


class SafetyBackend(ABC):
    """检测后端抽象基类。"""

    @abstractmethod
    def detect_pii(self, text: str) -> DetectionResult:
        ...

    @abstractmethod
    def detect_violation(self, text: str) -> DetectionResult:
        ...

    @abstractmethod
    def validate_input(self, text: str) -> DetectionResult:
        ...

    def scan_file(self, filepath: str, content: str) -> DetectionResult:
        ...


class LocalSafetyBackend(SafetyBackend):
    """本地正则 + 关键词检测，默认后端。"""

    def detect_pii(self, text: str) -> DetectionResult:
        findings: list[dict] = []
        occupied: list[tuple[int, int]] = []

        for pii_type, pattern in _PII_PATTERNS:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                if any(s <= start < e or s < end <= e for s, e in occupied):
                    continue
                occupied.append((start, end))
                raw = match.group()
                masked = _mask(raw)
                findings.append({"type": pii_type, "value": masked, "raw": raw})

        if not findings:
            return DetectionResult(risk_level="safe", findings=[], message="未检测到 PII 信息")

        return DetectionResult(
            risk_level="warning",
            findings=findings,
            message=f"检测到 {len(findings)} 个 PII 信息",
        )

    def detect_violation(self, text: str) -> DetectionResult:
        text_lower = text.lower()
        hits: list[dict] = []
        for word in _VIOLATION_WORDS:
            if word.lower() in text_lower:
                hits.append({"type": "violation_word", "value": word})

        if not hits:
            return DetectionResult(risk_level="safe", findings=[], message="未检测到违规内容")

        return DetectionResult(
            risk_level="danger",
            findings=hits,
            message=f"检测到 {len(hits)} 个违规词",
        )

    def validate_input(self, text: str) -> DetectionResult:
        findings: list[dict] = []

        if not text or not text.strip():
            return DetectionResult(risk_level="warning", findings=[{"type": "empty", "value": ""}], message="输入为空")

        if len(text) > 10000:
            findings.append({"type": "too_long", "value": f"长度 {len(text)}"})

        control_chars = re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text)
        if control_chars:
            findings.append({"type": "control_char", "value": f"{len(control_chars)} 个控制字符"})

        for pattern in _SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                findings.append({"type": "sql_injection", "value": pattern})
                break

        for pattern in _PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                findings.append({"type": "prompt_injection", "value": pattern})
                break

        if not findings:
            return DetectionResult(risk_level="safe", findings=[], message="输入规范，无异常")

        risk = "danger" if any(f["type"] in ("sql_injection", "control_char") for f in findings) else "warning"
        return DetectionResult(risk_level=risk, findings=findings, message=f"检测到 {len(findings)} 个问题")

    def scan_file(self, filepath: str, content: str) -> DetectionResult:
        findings: list[dict] = []

        for pattern in _SENSITIVE_FILE_PATTERNS:
            if re.search(pattern, filepath, re.IGNORECASE):
                findings.append({"type": "sensitive_path", "value": filepath})
                break

        for secret_type, pattern in _SECRET_PATTERNS:
            for match in re.finditer(pattern, content):
                findings.append({"type": secret_type, "value": _mask(match.group())})

        pii_result = self.detect_pii(content)
        if pii_result.findings:
            findings.extend(pii_result.findings)

        violation_result = self.detect_violation(content)
        if violation_result.findings:
            findings.extend(violation_result.findings)

        if not findings:
            return DetectionResult(risk_level="safe", findings=[], message="文件未检测到敏感信息")

        has_danger = any(f["type"] in ("sensitive_path", "private_key_header", "password_assignment", "token_assignment") for f in findings)
        risk = "danger" if has_danger else "warning"
        return DetectionResult(
            risk_level=risk,
            findings=findings,
            message=f"检测到 {len(findings)} 个敏感项",
        )


_default_backend: SafetyBackend = LocalSafetyBackend()


def get_backend() -> SafetyBackend:
    return _default_backend


def set_backend(backend: SafetyBackend) -> None:
    global _default_backend
    _default_backend = backend
