"""
EditTemplate — 编辑模板数据模型

定义剪辑流水线的配置模板，包含各节点的参数预设，
支持全自动 / 半自动两种执行模式。
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class NodeConfig(BaseModel):
    """单个节点的配置"""
    node_id: str = Field(..., description="节点 ID，如 'filter_clips', 'select_BGM'")
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: 正常执行; skip: 跳过; default: 使用默认参数"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="节点参数，对应各 *Input schema 字段，如 user_request, filter_include 等"
    )
    confirm_required: bool = Field(
        default=False,
        description="半自动模式下，该节点是否需要用户确认"
    )


class EditTemplate(BaseModel):
    """编辑模板"""
    template_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = Field(..., description="模板显示名，如 '旅行Vlog'")
    description: str = Field(default="", description="模板描述")
    nodes: List[NodeConfig] = Field(default_factory=list, description="各节点配置列表")
    auto_mode: Literal["full_auto", "semi_auto"] = Field(
        default="full_auto",
        description="full_auto: 全自动; semi_auto: 半自动（关键节点需确认）"
    )
    semi_auto_timeout_sec: int = Field(
        default=10,
        ge=3,
        le=60,
        description="半自动模式下等待用户确认的超时秒数"
    )
    is_preset: bool = Field(default=False, description="是否为内置预设模板（不可删除）")
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# 预设节点 ID 列表（DAG 拓扑顺序）
# ---------------------------------------------------------------------------
DEFAULT_PIPELINE_ORDER: List[str] = [
    "search_media",
    "load_media",
    "split_shots",
    "understand_clips",
    "filter_clips",
    "group_clips",
    "script_template_rec",
    "generate_script",
    "recommend_effects",
    "generate_voiceover",
    "select_BGM",
    "plan_timeline",
    "render_video",
]


# ---------------------------------------------------------------------------
# 内置预设模板
# ---------------------------------------------------------------------------

def _preset_travel_vlog() -> EditTemplate:
    return EditTemplate(
        template_id="preset_travel_vlog",
        name="旅行 Vlog",
        description="适合旅行类素材，自动生成配音和轻快 BGM",
        is_preset=True,
        auto_mode="full_auto",
        nodes=[
            NodeConfig(node_id="search_media", mode="skip"),
            NodeConfig(node_id="load_media", mode="auto"),
            NodeConfig(node_id="split_shots", mode="auto"),
            NodeConfig(node_id="understand_clips", mode="auto"),
            NodeConfig(node_id="filter_clips", mode="auto", params={
                "user_request": "保留风景优美、具有旅行氛围的镜头"
            }),
            NodeConfig(node_id="group_clips", mode="auto", params={
                "user_request": "按照旅行时间线组织，先总览再细节"
            }),
            NodeConfig(node_id="generate_script", mode="auto", params={
                "user_request": "轻松活泼的旅行 Vlog 文案风格"
            }),
            NodeConfig(node_id="generate_voiceover", mode="auto"),
            NodeConfig(node_id="select_BGM", mode="auto", params={
                "filter_include": {"mood": ["Chill", "Happy"], "scene": ["Travel", "Vlog"]}
            }),
            NodeConfig(node_id="plan_timeline", mode="auto"),
            NodeConfig(node_id="render_video", mode="auto"),
        ],
    )


def _preset_food_short() -> EditTemplate:
    return EditTemplate(
        template_id="preset_food_short",
        name="美食短片",
        description="适合美食/餐饮素材，突出食物质感",
        is_preset=True,
        auto_mode="full_auto",
        nodes=[
            NodeConfig(node_id="search_media", mode="skip"),
            NodeConfig(node_id="load_media", mode="auto"),
            NodeConfig(node_id="split_shots", mode="auto"),
            NodeConfig(node_id="understand_clips", mode="auto"),
            NodeConfig(node_id="filter_clips", mode="auto", params={
                "user_request": "保留食物特写和烹饪过程的镜头"
            }),
            NodeConfig(node_id="group_clips", mode="auto", params={
                "user_request": "按照烹饪流程组织，从食材到成品"
            }),
            NodeConfig(node_id="generate_script", mode="auto", params={
                "user_request": "精简的美食解说文案，突出食材和口感"
            }),
            NodeConfig(node_id="generate_voiceover", mode="auto"),
            NodeConfig(node_id="select_BGM", mode="auto", params={
                "filter_include": {"mood": ["Chill", "Happy"], "scene": ["Food", "Cafe"]}
            }),
            NodeConfig(node_id="plan_timeline", mode="auto"),
            NodeConfig(node_id="render_video", mode="auto"),
        ],
    )


def _preset_quick_cut() -> EditTemplate:
    return EditTemplate(
        template_id="preset_quick_cut",
        name="快速剪辑",
        description="最简流程，跳过筛选和配音，快速出片",
        is_preset=True,
        auto_mode="full_auto",
        nodes=[
            NodeConfig(node_id="search_media", mode="skip"),
            NodeConfig(node_id="load_media", mode="auto"),
            NodeConfig(node_id="split_shots", mode="auto"),
            NodeConfig(node_id="understand_clips", mode="skip"),
            NodeConfig(node_id="filter_clips", mode="skip"),
            NodeConfig(node_id="group_clips", mode="auto"),
            NodeConfig(node_id="generate_script", mode="skip"),
            NodeConfig(node_id="generate_voiceover", mode="skip"),
            NodeConfig(node_id="select_BGM", mode="auto"),
            NodeConfig(node_id="plan_timeline", mode="auto"),
            NodeConfig(node_id="render_video", mode="auto"),
        ],
    )


def _preset_semi_auto() -> EditTemplate:
    return EditTemplate(
        template_id="preset_semi_auto",
        name="半自动剪辑",
        description="关键节点需要确认（筛选、文案、配音），超时自动使用默认值",
        is_preset=True,
        auto_mode="semi_auto",
        semi_auto_timeout_sec=10,
        nodes=[
            NodeConfig(node_id="search_media", mode="skip"),
            NodeConfig(node_id="load_media", mode="auto"),
            NodeConfig(node_id="split_shots", mode="auto"),
            NodeConfig(node_id="understand_clips", mode="auto"),
            NodeConfig(node_id="filter_clips", mode="auto", confirm_required=True),
            NodeConfig(node_id="group_clips", mode="auto"),
            NodeConfig(node_id="generate_script", mode="auto", confirm_required=True),
            NodeConfig(node_id="generate_voiceover", mode="auto", confirm_required=True),
            NodeConfig(node_id="select_BGM", mode="auto"),
            NodeConfig(node_id="plan_timeline", mode="auto"),
            NodeConfig(node_id="render_video", mode="auto"),
        ],
    )


PRESET_TEMPLATES: List[EditTemplate] = [
    _preset_travel_vlog(),
    _preset_food_short(),
    _preset_quick_cut(),
    _preset_semi_auto(),
]
