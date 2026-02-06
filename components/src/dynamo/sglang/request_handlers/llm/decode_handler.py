# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Any, AsyncGenerator, Dict

import sglang as sgl

from dynamo._core import Component, Context
from dynamo.sglang.args import Config, DisaggregationMode
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class DecodeWorkerHandler(BaseWorkerHandler):
    """Handler for decode workers in both aggregated and disaggregated serving modes."""

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher,
        generate_endpoint=None,
    ) -> None:
        """Initialize decode worker handler.

        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Metrics publisher for the worker.
            generate_endpoint: The endpoint handle for discovery registration.
        """
        super().__init__(
            component,
            engine,
            config,
            publisher,
            generate_endpoint,
        )
        if self.serving_mode == DisaggregationMode.DECODE:
            logging.info(
                "Decode worker handler initialized (disaggregated decode mode)"
            )
        else:
            logging.info("Decode worker handler initialized (aggregated mode)")

    def cleanup(self) -> None:
        """Shutdown the engine and cleanup resources."""
        super().cleanup()
        self.engine.shutdown()
        logging.info("Engine shutdown")

    def _build_sampling_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Build sampling params from request format.

        Args:
            request: Request dict in either token-based or OpenAI format.

        Returns:
            Dict of sampling parameters for SGLang engine.
        """
        if self.skip_tokenizer_init:
            # Token-based request format
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})

            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
            }
        else:
            # OpenAI request format
            param_mapping = {
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "max_new_tokens": request.get("max_tokens"),
            }

        return {k: v for k, v in param_mapping.items() if v is not None}

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response in aggregated or disaggregated mode.

        Args:
            request: Request dict with input and sampling parameters.
            context: Context object for cancellation handling.

        Yields:
            Response dicts with token_ids or OpenAI-formatted chunks.

        Raises:
            RuntimeError: If no bootstrap info received from prefill worker.
        """
        logging.debug(f"New Request ID: {context.id()}")
        trace_id = context.trace_id
        sampling_params = self._build_sampling_params(request)
        input_param = self._get_input_param(request)

        if self.serving_mode == DisaggregationMode.DECODE:
            # Check if bootstrap_info is pre-computed in the request (from frontend)
            bootstrap_info = request.get("bootstrap_info")

            if not bootstrap_info:
                raise RuntimeError(
                    "bootstrap_info is required for disaggregated decode but was not provided"
                )

            logging.debug(
                f"Using bootstrap_info: "
                f"host={bootstrap_info['bootstrap_host']}, "
                f"port={bootstrap_info['bootstrap_port']}, "
                f"room={bootstrap_info['bootstrap_room']}"
            )

            trace_header = (
                self._get_trace_header(context) if self.enable_trace else None
            )

            # Extract dp_rank from routing info (set by KV router)
            routing = request.get("routing") or {}
            dp_rank = routing.get("dp_rank")

            decode = await self.engine.async_generate(
                **input_param,
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=bootstrap_info["bootstrap_host"],
                bootstrap_port=bootstrap_info["bootstrap_port"],
                bootstrap_room=bootstrap_info["bootstrap_room"],
                external_trace_header=trace_header,
                rid=trace_id,
                data_parallel_rank=dp_rank,
            )

            if self.skip_tokenizer_init:
                async for out in self._process_token_stream(decode, context):
                    yield out
            else:
                async for out in self._process_text_stream(decode, context):
                    yield out
        else:
            # Extract image URLs for multimodal requests. SGLang's mm_data_processor
            # handles loading/preprocessing, and the scheduler does vision encoding.
            image_data = None
            image_items = request.get("multi_modal_data", {}).get("image_url")
            if image_items:
                image_data = []
                for item in image_items:
                    if isinstance(item, str):
                        image_data.append(item)
                    elif isinstance(item, dict) and "Url" in item:
                        image_data.append(item["Url"])
                image_data = image_data or None

            trace_header = (
                self._get_trace_header(context) if self.enable_trace else None
            )

            # Extract dp_rank from routing info (set by KV router)
            routing = request.get("routing") or {}
            dp_rank = routing.get("dp_rank")

            agg = await self.engine.async_generate(
                **input_param,
                image_data=image_data,
                sampling_params=sampling_params,
                stream=True,
                external_trace_header=trace_header,
                rid=trace_id,
                data_parallel_rank=dp_rank,
            )
            if self.skip_tokenizer_init:
                async for out in self._process_token_stream(agg, context):
                    yield out
            else:
                async for out in self._process_text_stream(agg, context):
                    yield out

    async def _process_token_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process token-based stream output.

        With stream_output=True (enforced by Dynamo), SGLang sends disjoint segments
        containing only new tokens since the last output. We pass these through directly.

        Cancellation is handled inline: context.is_stopped() is checked each iteration
        and abort_request is called directly when detected, avoiding the overhead of a
        background asyncio.Task per request.

        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.

        Yields:
            Dict with token_ids and optional finish_reason.
        """
        sglang_request_id = None

        async for res in stream_source:
            meta_info = res["meta_info"]

            if sglang_request_id is None:
                sglang_request_id = meta_info.get("id")

            # Inline cancellation: abort via engine when context signals stop.
            if context.is_stopped():
                self._abort_request(sglang_request_id)
                break

            finish_reason = meta_info["finish_reason"]

            # With stream_output=True, output_ids contains only new tokens (disjoint)
            output_ids = res.get("output_ids")
            if not output_ids and not finish_reason:
                yield {"finish_reason": "error", "token_ids": []}
                break

            if finish_reason:
                input_tokens = meta_info["prompt_tokens"]
                completion_tokens = meta_info["completion_tokens"]
                cached_tokens = meta_info["cached_tokens"]
                yield {
                    "token_ids": output_ids,
                    "finish_reason": finish_reason["type"],
                    "completion_usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": input_tokens + completion_tokens,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens}
                        if cached_tokens
                        else None,
                    },
                }
            else:
                yield {"token_ids": output_ids}

    async def _process_text_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process text-based stream output in OpenAI format.

        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.

        Yields:
            OpenAI-formatted chat completion chunk dicts.
        """
        count = 0
        created_time = int(time.time())
        model_name = self.config.server_args.served_model_name
        sglang_request_id = None

        async for res in stream_source:
            meta_info = res["meta_info"]

            if sglang_request_id is None:
                sglang_request_id = meta_info.get("id")

            if context.is_stopped():
                self._abort_request(sglang_request_id)
                break

            text = res.get("text", "")
            finish_reason = meta_info["finish_reason"]
            next_count = len(text)
            delta = text[count:]

            yield {
                "id": meta_info["id"],
                "created": created_time,
                "choices": [
                    {
                        "index": res.get("index", 0),
                        "delta": {"role": "assistant", "content": delta},
                        "finish_reason": finish_reason["type"]
                        if finish_reason
                        else None,
                    }
                ],
                "model": model_name,
                "object": "chat.completion.chunk",
            }
            count = next_count
