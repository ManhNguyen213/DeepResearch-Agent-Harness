"""Query Planner node: Decomposes complex multi-hop queries into SubQueryTask items."""

from src.observability.tracing import observe, update_span_io
from src.schemas.state import HarnessState, SubQueryTask


class QueryPlanner:
    """Decomposes a user research query into targeted sub-queries informed by Procedural Memory."""

    @observe(
        name="query-planner",
        as_type="chain",
        capture_input=False,
        capture_output=False,
    )
    def plan(self, state: HarnessState) -> dict[str, list[SubQueryTask] | int]:
        """Generate initial or follow-up sub-queries for Isolated Sub-agent Workers."""
        if state.sub_queries:
            next_iter = state.iteration_count + 1
            update_span_io(
                input_data={
                    "user_query": state.user_query,
                    "existing_sub_queries_count": len(state.sub_queries),
                },
                output_data={
                    "iteration_count": next_iter,
                    "sub_queries": [sq.model_dump() for sq in state.sub_queries],
                },
            )
            return {"iteration_count": next_iter}

        default_task = SubQueryTask(
            sub_query_id="sq_1",
            query_text=state.user_query,
            target_entities=[],
            completed=False,
        )
        next_iter = state.iteration_count + 1
        update_span_io(
            input_data={
                "user_query": state.user_query,
                "procedural_rules": state.procedural_rules,
            },
            output_data={
                "iteration_count": next_iter,
                "sub_queries": [default_task.model_dump()],
            },
        )
        return {
            "sub_queries": [default_task],
            "iteration_count": next_iter,
        }
