"""
PipelineExecutor — 后端流水线执行器

按 DAG 拓扑顺序执行节点，支持：
- 全自动模式：所有节点连续执行
- 半自动模式：关键节点等待用户确认，超时回退默认值
- 进度回调：每个节点开始/完成/跳过时触发
"""
from __future__ import annotations

import asyncio
import traceback
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Literal, Optional

from open_storyline.nodes.node_manager import NodeManager
from open_storyline.pipeline.edit_template import (
    DEFAULT_PIPELINE_ORDER,
    EditTemplate,
    NodeConfig,
)
from open_storyline.storage.agent_memory import ArtifactStore
from open_storyline.storage.file import FileCompressor
from open_storyline.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ProgressCallback = Callable[
    [str, str, float, str],  # (node_id, status, progress_0_1, message)
    Coroutine[Any, Any, None],
]
ConfirmCallback = Callable[
    [str, Dict[str, Any], int],            # (node_id, params, timeout_sec)
    Coroutine[Any, Any, Dict[str, Any]],   # returns confirmed params
]


class PipelineError(Exception):
    """流水线执行过程中的错误"""

    def __init__(self, node_id: str, message: str, cause: Optional[Exception] = None):
        self.node_id = node_id
        super().__init__(f"[{node_id}] {message}")
        self.__cause__ = cause


class PipelineExecutor:
    """
    核心执行器：复用 NodeManager 的依赖检查 + ToolInterceptor 的级联执行模式，
    但绕过 LLM Agent，直接按模板配置驱动。
    """

    def __init__(
        self,
        node_manager: NodeManager,
        store: ArtifactStore,
        session_id: str,
        runtime: Any,          # agent runtime context (ClientContext)
    ):
        self.node_manager = node_manager
        self.store = store
        self.session_id = session_id
        self.runtime = runtime

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    async def run(
        self,
        template: EditTemplate,
        *,
        on_progress: Optional[ProgressCallback] = None,
        on_confirm: Optional[ConfirmCallback] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        按模板配置执行完整流水线。

        Returns:
            {"status": "done", "results": {node_id: result_summary, ...}}
        """
        # 1) 构建执行计划
        plan = self._build_execution_plan(template)
        results: Dict[str, Any] = {}
        total = len(plan)

        logger.info(
            f"[Pipeline] Starting pipeline '{template.name}' "
            f"({template.auto_mode}) with {total} nodes"
        )

        for idx, node_cfg in enumerate(plan):
            node_id = node_cfg.node_id

            # 检查取消
            if cancel_event and cancel_event.is_set():
                logger.info(f"[Pipeline] Cancelled before {node_id}")
                if on_progress:
                    await on_progress(node_id, "cancelled", idx / total, "用户取消")
                return {"status": "cancelled", "results": results}

            # 跳过
            if node_cfg.mode == "skip":
                logger.info(f"[Pipeline] Skipping {node_id}")
                if on_progress:
                    await on_progress(node_id, "skipped", (idx + 1) / total, "已跳过")
                results[node_id] = {"status": "skipped"}
                continue

            # 半自动确认
            params = dict(node_cfg.params)
            if (
                template.auto_mode == "semi_auto"
                and node_cfg.confirm_required
                and on_confirm is not None
            ):
                if on_progress:
                    await on_progress(
                        node_id, "waiting_confirm",
                        idx / total,
                        f"等待确认 ({template.semi_auto_timeout_sec}s)"
                    )
                params = await self._confirm_or_timeout(
                    node_id, params,
                    template.semi_auto_timeout_sec,
                    on_confirm,
                )

            # 执行节点
            if on_progress:
                await on_progress(node_id, "running", idx / total, f"正在执行 {node_id}")

            try:
                result = await self._execute_node(node_id, node_cfg.mode, params)
                results[node_id] = {
                    "status": "done",
                    "summary": result.get("summary", ""),
                    "is_error": result.get("isError", False),
                }
                status = "error" if result.get("isError") else "done"
                if on_progress:
                    await on_progress(
                        node_id, status,
                        (idx + 1) / total,
                        result.get("summary", "完成") if not result.get("isError") else str(result.get("summary", "执行出错")),
                    )
                if result.get("isError"):
                    logger.error(f"[Pipeline] {node_id} returned error: {result.get('summary')}")
                    # 非致命错误继续执行（除 plan_timeline / render_video）
                    if node_id in ("plan_timeline", "render_video"):
                        return {"status": "error", "failed_node": node_id, "results": results}
            except Exception as exc:
                logger.error(f"[Pipeline] {node_id} raised exception: {exc}")
                logger.debug(traceback.format_exc())
                results[node_id] = {"status": "error", "error": str(exc)}
                if on_progress:
                    await on_progress(node_id, "error", (idx + 1) / total, str(exc))
                if node_id in ("load_media", "plan_timeline", "render_video"):
                    return {"status": "error", "failed_node": node_id, "results": results}

        logger.info("[Pipeline] Pipeline completed successfully")
        return {"status": "done", "results": results}

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_execution_plan(self, template: EditTemplate) -> List[NodeConfig]:
        """
        按 DAG 拓扑顺序排列模板中的节点配置。
        未在模板中指定的节点按 DEFAULT_PIPELINE_ORDER 的默认行为处理。
        """
        # 模板节点 -> dict
        node_cfg_map: Dict[str, NodeConfig] = {
            nc.node_id: nc for nc in template.nodes
        }

        plan: List[NodeConfig] = []
        for node_id in DEFAULT_PIPELINE_ORDER:
            if node_id in node_cfg_map:
                plan.append(node_cfg_map[node_id])
            else:
                # 未在模板中指定 -> 跳过可选节点，保留固定节点
                if node_id in ("load_media", "plan_timeline", "render_video"):
                    plan.append(NodeConfig(node_id=node_id, mode="auto"))
                else:
                    plan.append(NodeConfig(node_id=node_id, mode="skip"))

        return plan

    async def _execute_node(
        self,
        node_id: str,
        mode: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        执行单个节点，复用现有 ToolInterceptor 的依赖解析逻辑。

        通过直接调用 NodeManager 的 tool.arun() 来执行，
        因为 ToolInterceptor 的 inject_media_content_before 会
        自动处理依赖注入。
        """
        tool = self.node_manager.get_tool(node_id)
        if not tool:
            raise PipelineError(node_id, f"Tool '{node_id}' not found in NodeManager")

        # 构造工具调用参数
        tool_args = {
            "artifact_id": self.store.generate_artifact_id(node_id),
            "mode": mode,
        }
        tool_args.update(params)

        from langchain_core.messages import ToolCall
        tool_call = ToolCall(
            args=tool_args,
            tool_call_type="auto" if mode == "auto" else "default",
            runtime=self.runtime,
        )

        result = await tool.arun(tool_call)

        # result 可以是 Command (from interceptor) or dict
        if hasattr(result, "update"):
            # Command 对象，提取 ToolMessage content
            messages = result.update.get("messages", [])
            if messages:
                content = messages[0].content
                if isinstance(content, dict):
                    return {
                        "summary": content.get("summary", {}).get("node_summary", ""),
                        "isError": content.get("isError", False),
                    }
            return {"summary": "", "isError": False}
        elif isinstance(result, dict):
            return result
        else:
            return {"summary": str(result), "isError": False}

    async def _confirm_or_timeout(
        self,
        node_id: str,
        params: Dict[str, Any],
        timeout_sec: int,
        on_confirm: ConfirmCallback,
    ) -> Dict[str, Any]:
        """半自动：请求确认，超时走默认值"""
        try:
            confirmed = await asyncio.wait_for(
                on_confirm(node_id, params, timeout_sec),
                timeout=timeout_sec,
            )
            logger.info(f"[Pipeline] {node_id} confirmed by user")
            return confirmed if isinstance(confirmed, dict) else params
        except asyncio.TimeoutError:
            logger.info(
                f"[Pipeline] {node_id} confirmation timed out "
                f"after {timeout_sec}s, using defaults"
            )
            return params
