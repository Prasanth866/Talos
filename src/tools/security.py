from __future__ import annotations

import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


_AWS_KEY_PATTERN = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_API_KEY_PATTERN = re.compile(
    r"\b(?:sk|sk-proj|sk-ant|ghp|gho|ghu|ghs|glpat)-[a-zA-Z0-9_\-]{20,}\b"
)
_PEM_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    r"[\s\S]*?"
    r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
)
_GENERIC_BEARER_PATTERN = re.compile(
    r"(?i)\b(?:bearer|token|secret|password|apikey)\s*[:=]\s*['\"]?"
    r"([a-zA-Z0-9_\-\.]{24,})['\"]?"
)


class SecretsScanner:
    """Detects and redacts credentials, access tokens, and private keys."""

    @classmethod
    def scan_and_redact(cls, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Scans text for secrets and returns (redacted_text, detected_threats)."""
        if not text:
            return text, []

        redacted = text
        threats: list[dict[str, Any]] = []

        aws_matches = _AWS_KEY_PATTERN.findall(redacted)
        if aws_matches:
            for match in set(aws_matches):
                threats.append(
                    {
                        "threat_type": "AWS_ACCESS_KEY",
                        "preview": f"{match[:4]}...{match[-4:]}",
                    }
                )
                redacted = redacted.replace(match, "[REDACTED_SECRET: AWS_ACCESS_KEY]")

        api_matches = _API_KEY_PATTERN.findall(redacted)
        if api_matches:
            for match in set(api_matches):
                threats.append(
                    {
                        "threat_type": "API_KEY",
                        "preview": f"{match[:7]}...",
                    }
                )
                redacted = redacted.replace(match, "[REDACTED_SECRET: API_KEY]")

        pem_matches = _PEM_PRIVATE_KEY_PATTERN.findall(redacted)
        if pem_matches:
            for match in set(pem_matches):
                threats.append(
                    {
                        "threat_type": "PRIVATE_KEY",
                        "preview": "-----BEGIN PRIVATE KEY-----...",
                    }
                )
                redacted = redacted.replace(match, "[REDACTED_SECRET: PRIVATE_KEY]")

        if threats:
            logger.warning(
                "security.secrets_intercepted",
                threat_count=len(threats),
                threat_types=[t["threat_type"] for t in threats],
            )

        return redacted, threats
