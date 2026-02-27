import asyncio
import base64
import io
import time
import random
import re
import wave
import librosa
from pathlib import Path
from typing import Any, Dict, Callable, Union

import requests

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_schema import GenerateVoiceoverInput
from open_storyline.nodes.node_state import NodeState
from open_storyline.utils.register import NODE_REGISTRY


# IndexTTS default voice index
_DEFAULT_VOICE_INDEX = "zh_female_intellectual"
_DEFAULT_INDEXTTS_BASE_URL = "http://39.102.122.9:8049"

# Retry / timeout constants
_MAX_RETRIES = 3
_CONNECT_TIMEOUT = 5        # seconds
_READ_TIMEOUT = 300          # seconds (TTS generation can be slow)
_MAX_TEXT_LEN = 500          # max characters per TTS API call


@NODE_REGISTRY.register()
class GenerateVoiceoverNode(BaseNode):
    meta = NodeMeta(
        name="generate_voiceover",
        description="Generate voice-over based on the script",
        node_id="generate_voiceover",
        node_kind="tts",
        require_prior_kind=["group_clips", "generate_script"],
        default_require_prior_kind=["group_clips", "generate_script"],
    )

    input_schema = GenerateVoiceoverInput

    # provider -> handler method name
    _PROVIDER_HANDLERS: Dict[str, str] = {
        "indextts": "_tts_indextts_sync",
    }

    _DEFAULT_PROVIDER = "indextts"

    MILLISECONDS_PER_SECOND = 1000.0

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        node_state.node_summary.info_for_user("Voiceover not generated")
        return {"voiceover": []}

    async def process(self, node_state: NodeState, inputs: Dict[str, Any], **params) -> Any:
        # 1) Get script
        group_scripts = (inputs.get("generate_script") or {}).get("group_scripts") or []
        if not isinstance(group_scripts, list) or not group_scripts:
            node_state.node_summary.info_for_user("No script found for voiceover generation (group_scripts is empty)")
            return {"voiceover": []}

        # 2) Provider selection — always use indextts
        provider_name = (inputs.get("provider") or "").strip() or self._DEFAULT_PROVIDER
        handler = self._get_provider_handler(provider_name)

        # 3) Resolve voice index from inputs (UI selection) or fall back to default
        voice_index = (inputs.get("voice_index") or inputs.get("index") or "").strip()
        if not voice_index:
            voice_index = _DEFAULT_VOICE_INDEX

        # 4) Resolve base_url
        base_url = (inputs.get("base_url") or "").strip()
        if not base_url:
            provider_cfg = self._get_provider_cfg(provider_name)
            base_url = (provider_cfg.get("base_url") or "").strip() or _DEFAULT_INDEXTTS_BASE_URL

        node_state.node_summary.info_for_user(f"TTS: IndexTTS, voice: {voice_index}")

        # 5) Health check before batch generation
        healthy = await asyncio.to_thread(self._check_health, base_url)
        if not healthy:
            raise RuntimeError(
                f"IndexTTS service at {base_url} is not reachable. "
                f"Please check if the service is running."
            )

        # 6) Prepare output directory
        artifact_id = node_state.artifact_id
        session_id = node_state.session_id
        if not artifact_id or not session_id:
            raise ValueError("缺失 artifact_id / session_id，无法生成配音输出目录")

        output_dir = self.server_cache_dir / str(session_id) / str(artifact_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 7) Generate segment by segment
        ts_ms = int(time.time() * 1000)
        total = len(group_scripts)
        voiceover: list[dict[str, Any]] = []

        for i, group in enumerate(group_scripts, start=1):
            group_id = (group or {}).get("group_id", "")
            raw_text = (group or {}).get("raw_text", "")

            if not group_id:
                raise ValueError(f"Missing group_id: {group}")
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError(f"raw_text is empty for group_id={group_id}, cannot generate speech.")

            voiceover_id = f"voiceover_{i:04d}"
            wav_path = output_dir / f"{voiceover_id}_{ts_ms}.wav"

            await asyncio.to_thread(
                handler,
                text=raw_text,
                wav_path=wav_path,
                base_url=base_url,
                voice_index=voice_index,
            )

            duration = self._wav_duration_ms(wav_path)
            voiceover.append(
                {
                    "voiceover_id": voiceover_id,
                    "group_id": group_id,
                    "path": str(wav_path),
                    "duration": duration,
                }
            )

            node_state.node_summary.info_for_user(
                f"Generated {voiceover_id} ({i}/{total})",
                preview_urls=[str(wav_path)],
            )

        node_state.node_summary.info_for_user(f"Generated {len(voiceover)} voiceover segments in total")
        return {"voiceover": voiceover}

    # ---------------------------------------------------------------------
    # Provider dispatch / config helpers
    # ---------------------------------------------------------------------

    def _get_provider_handler(self, provider_name: str) -> Callable[..., None]:
        if provider_name is None or provider_name == "":
            provider_name = self._DEFAULT_PROVIDER
        method_name = self._PROVIDER_HANDLERS.get(provider_name)
        if not method_name:
            raise ValueError(f"Unsupported TTS provider: {provider_name}, currently supported: {list(self._PROVIDER_HANDLERS.keys())}")
        handler = getattr(self, method_name, None)
        if not callable(handler):
            raise ValueError(f"Handler for provider={provider_name} not implemented: {method_name}")
        return handler

    def _get_provider_cfg(self, provider_name: str) -> Dict[str, Any]:
        providers = getattr(self.server_cfg.generate_voiceover, "providers", None) or {}
        cfg = providers.get(provider_name)
        if not isinstance(cfg, dict):
            return {"base_url": _DEFAULT_INDEXTTS_BASE_URL}
        return cfg

    def _check_health(self, base_url: str) -> bool:
        """Check if IndexTTS service is reachable before batch generation."""
        try:
            r = requests.get(
                f"{base_url.rstrip('/')}/api/v1/health",
                timeout=(_CONNECT_TIMEOUT, 10),
            )
            return r.ok
        except Exception:
            return False

    def _wav_duration_ms(self, wav_path: Union[str, Path]) -> int:
        p = str(wav_path)
        duration_s = librosa.get_duration(path=p)
        return int(round(duration_s * self.MILLISECONDS_PER_SECOND))

    # ---------------------------------------------------------------------
    # Text splitting for long segments
    # ---------------------------------------------------------------------

    @staticmethod
    def _split_long_text(text: str, max_len: int = _MAX_TEXT_LEN) -> list[str]:
        """
        Split long text into chunks at sentence boundaries (。！？.!?)
        to avoid IndexTTS quality degradation on overly long inputs.
        """
        text = text.strip()
        if len(text) <= max_len:
            return [text]

        # Split at sentence-ending punctuation, keep the delimiter
        parts = re.split(r'(?<=[。！？.!?\n])', text)
        chunks: list[str] = []
        current = ""

        for part in parts:
            if not part:
                continue
            if len(current) + len(part) <= max_len:
                current += part
            else:
                if current:
                    chunks.append(current.strip())
                current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]

    # ---------------------------------------------------------------------
    # IndexTTS provider implementation
    # ---------------------------------------------------------------------

    def _tts_indextts_sync(
        self,
        *,
        text: str,
        wav_path: Path,
        base_url: str,
        voice_index: str,
    ) -> None:
        """
        Call IndexTTS API to generate voiceover.
        - Splits long text into chunks and concatenates audio.
        - Retries transient failures with exponential backoff + jitter.
        
        API: POST /api/v1/tts  (form-data)
        Response: {"success": true, "audio_base64": "...", "sample_rate": 24000}
        """
        chunks = self._split_long_text(text)
        audio_parts: list[bytes] = []

        for chunk in chunks:
            audio_bytes = self._call_indextts_api(
                text=chunk,
                base_url=base_url,
                voice_index=voice_index,
            )
            audio_parts.append(audio_bytes)

        # Concatenate WAV parts using wave stdlib (handles variable header sizes)
        if len(audio_parts) == 1:
            wav_path.write_bytes(audio_parts[0])
        else:
            combined = self._concat_wav_parts(audio_parts)
            wav_path.write_bytes(combined)

    def _call_indextts_api(
        self,
        *,
        text: str,
        base_url: str,
        voice_index: str,
    ) -> bytes:
        """Single API call with retry + exponential backoff."""
        api_url = base_url.rstrip("/") + "/api/v1/tts"

        data = {
            "input_text": text,
            "index": voice_index,
            "beam_size": 1,
            "sample_rate": 24000,
        }

        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.post(
                    api_url,
                    data=data,
                    timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                )
                resp.raise_for_status()

                resp_json = resp.json()
                if not isinstance(resp_json, dict):
                    raise RuntimeError(f"IndexTTS: invalid response: {resp.text[:200]}")

                if not resp_json.get("success"):
                    error_msg = resp_json.get("error") or resp_json.get("message") or "Unknown error"
                    raise RuntimeError(f"IndexTTS TTS failed: {error_msg}")

                audio_b64 = resp_json.get("audio_base64")
                if not audio_b64:
                    raise RuntimeError("IndexTTS: no audio_base64 in response")

                try:
                    return base64.b64decode(audio_b64)
                except Exception as e:
                    raise RuntimeError(
                        f"IndexTTS: base64 decode failed: {e}, "
                        f"data[:64]={str(audio_b64)[:64]}"
                    )

            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                continue

        raise RuntimeError(
            f"IndexTTS API failed after {_MAX_RETRIES} retries: {last_error}"
        )

    @staticmethod
    def _concat_wav_parts(audio_parts: list[bytes]) -> bytes:
        """Concatenate multiple WAV byte-strings using the wave stdlib.
        Correctly handles variable-length WAV headers (LIST, INFO, fact chunks).
        """
        output = io.BytesIO()
        with wave.open(io.BytesIO(audio_parts[0]), 'rb') as first:
            params = first.getparams()
            with wave.open(output, 'wb') as out:
                out.setparams(params)
                for part in audio_parts:
                    with wave.open(io.BytesIO(part), 'rb') as w:
                        out.writeframes(w.readframes(w.getnframes()))
        return output.getvalue()