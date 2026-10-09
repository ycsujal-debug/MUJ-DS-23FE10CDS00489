import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent
CONFIG_YAML_PATH = CONFIG_DIR / "config.yaml"


class NormalizerConfig(BaseModel):
    max_repeated_chars: int = 2
    shout_case_threshold: float = 0.35
    leetspeak_dict: Dict[str, str] = Field(default_factory=dict)


class NGramConfig(BaseModel):
    default_n: int = 2
    default_smoothing: str = "laplace"
    laplace_alpha: float = 1.0
    low_perplexity_threshold: float = 18.0
    high_perplexity_threshold: float = 400.0
    variance_threshold: float = 4000.0


class POSProfilerConfig(BaseModel):
    promotional_modifier_ratio_threshold: float = 0.65
    bot_modifier_ratio_threshold: float = 0.12
    low_lexical_diversity_threshold: float = 0.45
    superlative_density_threshold: float = 0.08


class ClassicalNLPConfig(BaseModel):
    normalizer: NormalizerConfig = Field(default_factory=NormalizerConfig)
    ngrams: NGramConfig = Field(default_factory=NGramConfig)
    pos_profiler: POSProfilerConfig = Field(default_factory=POSProfilerConfig)


class LLMConfig(BaseModel):
    default_provider: str = "gemini"
    gemini_model: str = "gemini-1.5-flash"
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_retries: int = 3
    timeout_seconds: int = 30
    backoff_factor: float = 2.0


class AppConfig(BaseModel):
    title: str = "ReviewRadar: E-Commerce Fake Review & Bot Detector"
    description: str = "Forensic NLP Pipeline combining Classical Statistics and LLM Analysis"
    version: str = "1.0.0"


class Settings(BaseSettings):
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    default_llm_provider: str = Field(default="gemini", alias="DEFAULT_LLM_PROVIDER")
    
    classical_nlp: ClassicalNLPConfig = Field(default_factory=ClassicalNLPConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    app: AppConfig = Field(default_factory=AppConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


def load_settings(yaml_path: Optional[Path] = None) -> Settings:
    target_path = yaml_path or CONFIG_YAML_PATH
    yaml_data: Dict[str, Any] = {}
    
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    provider = os.getenv("DEFAULT_LLM_PROVIDER", yaml_data.get("llm", {}).get("default_provider", "gemini"))

    classical_data = yaml_data.get("classical_nlp", {})
    llm_data = yaml_data.get("llm", {})
    app_data = yaml_data.get("app", {})

    settings = Settings(
        GEMINI_API_KEY=gemini_key,
        OPENAI_API_KEY=openai_key,
        DEFAULT_LLM_PROVIDER=provider,
        classical_nlp=ClassicalNLPConfig(
            normalizer=NormalizerConfig(**classical_data.get("normalizer", {})),
            ngrams=NGramConfig(**classical_data.get("ngrams", {})),
            pos_profiler=POSProfilerConfig(**classical_data.get("pos_profiler", {}))
        ),
        llm=LLMConfig(**llm_data),
        app=AppConfig(**app_data)
    )
    return settings


settings = load_settings()
