"""
TemplateStore — 模板持久化

JSON 文件存储于 .storyline/templates/ 目录，
每个模板一个文件：{template_id}.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from open_storyline.pipeline.edit_template import EditTemplate, PRESET_TEMPLATES
from open_storyline.utils.logging import get_logger

logger = get_logger(__name__)


class TemplateStore:
    """模板 CRUD + JSON 持久化"""

    def __init__(self, templates_dir: str = ".storyline/templates"):
        self._dir = Path(templates_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # 内存缓存
        self._cache: Dict[str, EditTemplate] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # 懒加载
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # 1) 先加载预设
        for preset in PRESET_TEMPLATES:
            self._cache[preset.template_id] = preset.model_copy(deep=True)
        # 2) 加载用户自定义（磁盘文件）
        try:
            for f in self._dir.iterdir():
                if f.suffix != ".json":
                    continue
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    tpl = EditTemplate(**data)
                    # 用户模板覆盖同名预设（支持用户修改预设的 fork）
                    self._cache[tpl.template_id] = tpl
                except Exception as e:
                    logger.warning(f"Failed to load template {f}: {e}")
        except FileNotFoundError:
            pass
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_all(self) -> List[EditTemplate]:
        """返回全部模板（预设 + 用户自定义），按 created_at 排序"""
        self._ensure_loaded()
        templates = list(self._cache.values())
        templates.sort(key=lambda t: (not t.is_preset, t.created_at))
        return templates

    def get(self, template_id: str) -> Optional[EditTemplate]:
        self._ensure_loaded()
        return self._cache.get(template_id)

    def save(self, template: EditTemplate) -> EditTemplate:
        """创建或更新模板，并持久化到磁盘"""
        self._ensure_loaded()
        template.updated_at = time.time()
        if template.template_id not in self._cache:
            template.created_at = time.time()
        self._cache[template.template_id] = template
        # 持久化
        self._write_to_disk(template)
        return template

    def delete(self, template_id: str) -> bool:
        """删除模板（预设不可删除）"""
        self._ensure_loaded()
        tpl = self._cache.get(template_id)
        if not tpl:
            return False
        if tpl.is_preset:
            raise ValueError(f"Cannot delete preset template: {template_id}")
        del self._cache[template_id]
        # 删除磁盘文件
        fp = self._dir / f"{template_id}.json"
        if fp.exists():
            fp.unlink()
        return True

    # ------------------------------------------------------------------
    # 磁盘 IO
    # ------------------------------------------------------------------
    def _write_to_disk(self, template: EditTemplate) -> None:
        fp = self._dir / f"{template.template_id}.json"
        fp.write_text(
            template.model_dump_json(indent=2),
            encoding="utf-8",
        )
