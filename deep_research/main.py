"""
Deep Research System - Main Entry Point

This is the main entry point for running the Deep Research pipeline.
It supports both command-line and programmatic usage.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.state import GraphState, create_initial_state
from core.utils import (
    setup_logging,
    get_logger,
    print_banner,
    print_step,
    print_state_summary,
    format_state_summary,
    write_file,
    ensure_dir,
)
from core.citation_manager import CitationManager
from config import get_config
from workflow import compile_workflow, visualize_workflow


logger = get_logger(__name__)


def run_research(
    query: str,
    output_dir: Optional[str] = None,
    use_mock: bool = False,
    verbose: bool = False,
    show_state: bool = False,
    save_intermediate: bool = False,
) -> GraphState:
    """
    Run the Deep Research pipeline for a given query.
    
    Args:
        query: The research question to investigate
        output_dir: Directory to save outputs (optional)
        use_mock: Whether to use mock tools for testing
        verbose: Whether to print verbose output
        show_state: 是否在每步后打印当前状态摘要（子任务、数据条数、草稿预览等）
        save_intermediate: 是否在每步后把中间状态保存到 JSON 文件
        
    Returns:
        Final GraphState with results
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    # 设置 Mock 模式环境变量（供 execution agent 读取）
    import os
    if use_mock:
        os.environ["DEEP_RESEARCH_USE_MOCK"] = "true"
    else:
        os.environ.pop("DEEP_RESEARCH_USE_MOCK", None)
    
    print_banner("Deep Research System")
    print(f"\n研究问题: {query}\n")
    if use_mock:
        print("[Mock 模式] 使用模拟数据，不调用真实 API\n")
    
    # Reset citation manager for fresh run
    CitationManager.reset()
    
    # Create initial state
    initial_state = create_initial_state(query)
    
    # Compile workflow
    print_step("编译工作流", "running")
    workflow = compile_workflow(use_checkpointer=True)
    print_step("编译工作流", "done")
    
    # Run the workflow
    print_step("开始研究流程", "running")
    
    config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
    
    try:
        import json
        step_num = 0
        
        # Stream the workflow execution
        for event in workflow.stream(initial_state.model_dump(), config):
            for node_name, state_update in event.items():
                step_num += 1
                
                # 进度：步骤 N - 节点名
                print_step(f"步骤 {step_num}: {node_name}", "done")
                
                if verbose:
                    print(f"  [本步更新] workflow_status={state_update.get('workflow_status', '-')}")
                    if "sub_tasks" in state_update:
                        print(f"  子任务数: {len(state_update['sub_tasks'])}")
                    if "task_plan" in state_update:
                        print(f"  执行计划: {state_update['task_plan']}")
                    if "raw_data" in state_update:
                        print(f"  原始数据: {len(state_update['raw_data'])} 条")
                    if "selected_data" in state_update:
                        print(f"  筛选数据: {len(state_update['selected_data'])} 条")
                    if "draft_report" in state_update and state_update["draft_report"]:
                        print(f"  报告草稿: {len(state_update['draft_report'])} 字")
                    if "review_feedback" in state_update and state_update["review_feedback"]:
                        fb = state_update["review_feedback"]
                        approved = fb.get("is_approved") if isinstance(fb, dict) else getattr(fb, "is_approved", None)
                        print(f"  审查结果: 通过={approved}")
                
                # 每步后：获取完整当前状态并可选打印/保存
                current_values = workflow.get_state(config).values
                if not current_values:
                    current_values = state_update
                
                if show_state or save_intermediate:
                    # 合并成“当前完整状态”（stream 返回的是增量，get_state 是合并后的）
                    full_state = dict(current_values) if current_values else dict(state_update)
                    
                    if show_state:
                        print_state_summary(full_state, title=f"步骤 {step_num} 完成后 · {node_name}")
                    
                    if save_intermediate:
                        out_path = ensure_dir(output_dir or "./intermediate")
                        safe_state = _state_to_serializable(full_state)
                        fname = out_path / f"state_step_{step_num:02d}_{node_name}.json"
                        write_file(fname, json.dumps(safe_state, ensure_ascii=False, indent=2))
                        print(f"  [已保存中间状态] {fname}")
        
        # Get final state
        final_state_dict = workflow.get_state(config).values
        final_state = GraphState(**final_state_dict)
        
        print_step("研究流程完成", "done")
        
        # Save outputs if requested
        if output_dir:
            save_outputs(final_state, output_dir)
        
        return final_state
        
    except Exception as e:
        logger.error("Research pipeline failed", error=str(e))
        print_step(f"研究流程失败: {str(e)}", "error")
        raise


def _state_to_serializable(state: dict) -> dict:
    """把 state 里的 Pydantic 对象转成可 JSON 序列化的 dict。"""
    import json
    from pydantic import BaseModel
    
    out = {}
    for k, v in state.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, BaseModel):
            out[k] = v.model_dump()
        elif isinstance(v, list):
            out[k] = [
                x.model_dump() if isinstance(x, BaseModel) else x
                for x in v
            ]
        elif isinstance(v, dict):
            out[k] = {kk: (vv.model_dump() if isinstance(vv, BaseModel) else vv) for kk, vv in v.items()}
        else:
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)
    return out


async def run_research_async(
    query: str,
    output_dir: Optional[str] = None,
    use_mock: bool = False,
    verbose: bool = False,
    show_state: bool = False,
    save_intermediate: bool = False,
) -> GraphState:
    """
    Async version of the research pipeline.
    
    Args:
        query: The research question to investigate
        output_dir: Directory to save outputs (optional)
        use_mock: Whether to use mock tools for testing
        verbose: Whether to print verbose output
        show_state: 是否在每步后打印当前状态摘要
        save_intermediate: 是否在每步后保存中间状态到 JSON
        
    Returns:
        Final GraphState with results
    """
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    # 设置 Mock 模式环境变量（供 execution agent 读取）
    import os
    if use_mock:
        os.environ["DEEP_RESEARCH_USE_MOCK"] = "true"
    else:
        os.environ.pop("DEEP_RESEARCH_USE_MOCK", None)
    
    print_banner("Deep Research System (Async)")
    print(f"\n研究问题: {query}\n")
    if use_mock:
        print("[Mock 模式] 使用模拟数据，不调用真实 API\n")
    
    CitationManager.reset()
    initial_state = create_initial_state(query)
    
    print_step("编译工作流", "running")
    workflow = compile_workflow(use_checkpointer=True)
    print_step("编译工作流", "done")
    
    print_step("开始异步研究流程", "running")
    
    config = {"configurable": {"thread_id": f"research_async_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
    
    try:
        import json
        step_num = 0
        
        async for event in workflow.astream(initial_state.model_dump(), config):
            for node_name, state_update in event.items():
                step_num += 1
                print_step(f"步骤 {step_num}: {node_name}", "done")
                
                if verbose:
                    print(f"  [本步更新] workflow_status={state_update.get('workflow_status', '-')}")
                
                current_values = workflow.get_state(config).values
                full_state = dict(current_values) if current_values else dict(state_update)
                
                if show_state:
                    print_state_summary(full_state, title=f"步骤 {step_num} 完成后 · {node_name}")
                
                if save_intermediate:
                    out_path = ensure_dir(output_dir or "./intermediate")
                    safe_state = _state_to_serializable(full_state)
                    fname = out_path / f"state_step_{step_num:02d}_{node_name}.json"
                    write_file(fname, json.dumps(safe_state, ensure_ascii=False, indent=2))
                    print(f"  [已保存中间状态] {fname}")
        
        final_state_dict = workflow.get_state(config).values
        final_state = GraphState(**final_state_dict)
        
        print_step("异步研究流程完成", "done")
        
        if output_dir:
            save_outputs(final_state, output_dir)
        
        return final_state
        
    except Exception as e:
        logger.error("Async research pipeline failed", error=str(e))
        print_step(f"异步研究流程失败: {str(e)}", "error")
        raise


def save_outputs(state: GraphState, output_dir: str) -> None:
    """
    Save research outputs to files.
    
    Args:
        state: Final GraphState
        output_dir: Directory to save outputs
    """
    output_path = ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save final report
    if state.final_report:
        report_file = output_path / f"report_{timestamp}.md"
        write_file(report_file, state.final_report)
        print(f"\n报告已保存至: {report_file}")
    elif state.draft_report:
        draft_file = output_path / f"draft_{timestamp}.md"
        write_file(draft_file, state.draft_report)
        print(f"\n草稿已保存至: {draft_file}")
    
    # Save metadata
    metadata = {
        "query": state.original_query,
        "timestamp": timestamp,
        "sub_tasks": [t.model_dump() for t in state.sub_tasks],
        "task_plan": state.task_plan,
        "raw_data_count": len(state.raw_data),
        "selected_data_count": len(state.selected_data),
        "revision_count": state.revision_count,
        "errors": state.errors,
        "workflow_status": state.workflow_status,
    }
    
    import json
    metadata_file = output_path / f"metadata_{timestamp}.json"
    write_file(metadata_file, json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"元数据已保存至: {metadata_file}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Deep Research System - AI-powered research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py "人工智能在医疗领域的应用"
  python main.py "量子计算的发展现状" -o ./output -v
  python main.py --visualize
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="研究问题"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="输出目录",
        default="./output"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出（本步更新字段）"
    )
    
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="每步完成后打印当前状态摘要（子任务、数据条数、草稿预览等）"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="每步完成后将中间状态保存为 JSON 到输出目录（或 ./intermediate）"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用模拟工具（用于测试）"
    )
    
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="使用异步模式"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="显示工作流程图"
    )
    
    args = parser.parse_args()
    
    # Show workflow visualization
    if args.visualize:
        print(visualize_workflow())
        return
    
    # Check for query
    if not args.query:
        parser.print_help()
        print("\n错误: 请提供研究问题")
        sys.exit(1)
    
    try:
        if args.use_async:
            # Run async version
            final_state = asyncio.run(
                run_research_async(
                    query=args.query,
                    output_dir=args.output,
                    use_mock=args.mock,
                    verbose=args.verbose,
                    show_state=args.show_state,
                    save_intermediate=args.save_intermediate,
                )
            )
        else:
            # Run sync version
            final_state = run_research(
                query=args.query,
                output_dir=args.output,
                use_mock=args.mock,
                verbose=args.verbose,
                show_state=args.show_state,
                save_intermediate=args.save_intermediate,
            )
        
        # Print summary
        print("\n" + "=" * 60)
        print("研究完成!")
        print("=" * 60)
        print(f"子任务数: {len(final_state.sub_tasks)}")
        print(f"收集数据: {len(final_state.raw_data)} 条")
        print(f"筛选数据: {len(final_state.selected_data)} 条")
        print(f"修订轮次: {final_state.revision_count}")
        print(f"最终状态: {final_state.workflow_status}")
        
        if final_state.errors:
            print(f"\n警告/错误: {len(final_state.errors)} 条")
            for err in final_state.errors:
                print(f"  - {err}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
