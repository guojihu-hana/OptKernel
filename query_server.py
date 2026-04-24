import os

TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
SGLANG_KEY = os.environ.get("SGLANG_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
from typing import Any, Optional
from pathlib import Path
import json
import sys
import time
import uuid

_CONCISE_THINKING_ZH = (
    "Think concisely, avoid any repetition, and do not output the entire code during thinking."
)


def _merge_concise_system_instruction(system_prompt: str) -> str:
    base = (system_prompt or "").rstrip()
    if _CONCISE_THINKING_ZH in base:
        return base
    return f"{base}\n{_CONCISE_THINKING_ZH}"


def colorize_finish_reason(reason: Optional[str]) -> str:
    colors = {
        "stop": "\033[92m",  # Green
        "end_turn": "\033[92m",
        "length": "\033[93m",  # Yellow
        "max_tokens": "\033[93m",
        "content_filter": "\033[91m",  # Red
        "stop_sequence": "\033[91m",
        "tool_calls": "\033[94m",  # Blue
        "function_call": "\033[94m",
        "tool_use": "\033[94m",
        "null": "\033[90m",  # Grey
    }
    reset_color = "\033[0m"
    if reason is None:
        return f"\033[90mFinish reason: unknown{reset_color}"
    color = colors.get(reason, "\033[90m")  # Default to grey
    return f"{color}Finish reason: {reason}{reset_color}"


def _qs_ret(result: Any, llm_output_dumped: bool = False) -> tuple[Any, bool]:
    """Pair query_server payload with whether llm_output was written via stream_dump_path (local/vllm)."""
    return (result, llm_output_dumped)


def query_server(
    prompt: str | list[dict],
    system_prompt: str = "You are a helpful assistant",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    max_tokens: int = 128,
    num_completions: int = 1,
    server_port: int = 30000,
    server_address: str = "localhost",
    server_type: str = "local",
    model_name: str = "default",
    is_reasoning_model: bool = True,
    budget_tokens: int = 0,
    reasoning_effort: str = "medium",
    log_path: Optional[str] = None,
    call_type: str = "unknown",
    round_idx: int = -1,
    return_metadata: bool = False,
    stream_dump_path: Optional[str] = None,
    openai_compatible_api_key: str = "",
    repetition_penalty: float = 1.00,
    max_context_length: int = 0,
):
    system_prompt = _merge_concise_system_instruction(system_prompt)
    match server_type:
        case "local":
            from llm_local import GenerationConfig, get_llm

            # Same OpenAI-compatible URL as vllm; use server_address/server_port (defaults: localhost:30000).
            llm = get_llm(
                model_name,
                server_url=f"http://{server_address}:{server_port}/v1",
                api_key=openai_compatible_api_key or "EMPTY",
            )
            model = model_name

        case "vllm":
            from llm_local import GenerationConfig, get_llm

            llm = get_llm(
                model_name,
                server_url=f"http://{server_address}:{server_port}/v1",
                api_key=openai_compatible_api_key or "EMPTY",
            )
            model = model_name

        case "sglang":
            from llm_local import GenerationConfig, get_llm

            llm = get_llm(
                model_name,
                server_url=f"http://{server_address}:{server_port}/v1",
                api_key=openai_compatible_api_key or "EMPTY",
            )
            model = model_name

        case "deepseek":
            from openai import OpenAI
            client = OpenAI(
                api_key=DEEPSEEK_KEY,
                base_url="https://api.deepseek.com",
                timeout=10000000,
                max_retries=3,
            )
            model = model_name

        case "fireworks":
            from openai import OpenAI
            client = OpenAI(
                api_key=FIREWORKS_API_KEY,
                base_url="https://api.fireworks.ai/inference/v1",
                timeout=10000000,
                max_retries=3,
            )
            model = model_name

        case "anthropic":
            import anthropic
            anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
            client_kwargs = {"api_key": ANTHROPIC_KEY}
            if anthropic_base_url:
                client_kwargs["base_url"] = anthropic_base_url
            client = anthropic.Anthropic(**client_kwargs)
            model = model_name

        case "google":
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_KEY)
            model = model_name

        case "together":
            from together import Together
            client = Together(api_key=TOGETHER_KEY)
            model = model_name

        case "sambanova":
            from openai import OpenAI
            client = OpenAI(api_key=SAMBANOVA_API_KEY, base_url="https://api.sambanova.ai/v1")
            model = model_name

        case "openai":
            from openai import OpenAI

            _openai_base = os.environ.get("OPENAI_BASE_URL", "").strip()
            if _openai_base:
                # e.g. vLLM OpenAI-compatible API: http://host:8000/v1
                _key = OPENAI_KEY or os.environ.get("OPENAI_API_KEY") or "EMPTY"
                client = OpenAI(api_key=_key, base_url=_openai_base)
            else:
                client = OpenAI(api_key=OPENAI_KEY)
            model = model_name

        case "file_queue":
            # Offline compute node -> shared filesystem queue -> online dev machine.
            # server_address is treated as queue_dir (preferred), else use env.
            # This returns a single string completion (like other providers).
            queue_dir = (server_address or "").strip() or os.environ.get("KERNELGEN_QUEUE_DIR", "").strip()
            if not queue_dir:
                raise ValueError(
                    "server_type='file_queue' requires --server_address <queue_dir> "
                    "or env KERNELGEN_QUEUE_DIR to be set."
                )
            queue_root = Path(queue_dir).expanduser().resolve()
            req_dir = queue_root / "requests"
            resp_dir = queue_root / "responses"
            err_dir = queue_root / "errors"
            req_dir.mkdir(parents=True, exist_ok=True)
            resp_dir.mkdir(parents=True, exist_ok=True)
            err_dir.mkdir(parents=True, exist_ok=True)

            def _atomic_write_json(path: Path, obj) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(path)

            def _file_queue_call(messages_payload, system_prompt_payload: str) -> str | list[str]:
                req_id = uuid.uuid4().hex
                created_at = time.strftime("%Y-%m-%d %H:%M:%S")
                req = {
                    "id": req_id,
                    "created_at": created_at,
                    "model": model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "system": system_prompt_payload,
                    "messages": messages_payload,
                    "call_type": call_type,
                    "round_idx": round_idx,
                    "is_reasoning_model": is_reasoning_model,
                    "budget_tokens": budget_tokens,
                }
                _atomic_write_json(req_dir / f"{req_id}.json", req)

                timeout_s = float(os.environ.get("KERNELGEN_QUEUE_TIMEOUT_S", "3600"))
                poll_s = float(os.environ.get("KERNELGEN_QUEUE_POLL_S", "0.5"))
                t0 = time.time()
                resp_path = resp_dir / f"{req_id}.json"
                err_path = err_dir / f"{req_id}.json"

                while True:
                    if resp_path.exists():
                        data = json.loads(resp_path.read_text(encoding="utf-8"))
                        content = data.get("content", "")

                        # Optional usage logging if server provides it
                        usage = data.get("usage", None)
                        if usage:
                            try:
                                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) if isinstance(usage, dict) else 0
                                output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0
                                total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
                                usage_str = f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}"
                                print(usage_str, file=sys.stderr, flush=True)
                                if log_path and log_path != "":
                                    import datetime as _dt
                                    file_exists = os.path.exists(log_path)
                                    with open(log_path, "a", encoding="utf-8") as f:
                                        if not file_exists:
                                            f.write("timestamp,round_idx,call_type,input_tokens,output_tokens,total_tokens\n")
                                        timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        f.write(f"{timestamp},{round_idx},{call_type},{input_tokens},{output_tokens},{total_tokens}\n")
                            except Exception:
                                pass

                        if isinstance(content, list):
                            return content
                        return str(content or "")

                    if err_path.exists():
                        data = json.loads(err_path.read_text(encoding="utf-8"))
                        raise RuntimeError(f"file_queue server error: {data.get('error_type')}: {data.get('error')}")

                    if time.time() - t0 > timeout_s:
                        raise TimeoutError(f"file_queue timeout after {timeout_s}s waiting for id={req_id}")
                    time.sleep(poll_s)

            # Defer real execution below (shared messages building logic).

        case _:
            raise NotImplementedError(f"Unsupported server_type: {server_type}")

    def _pack(text: str, finish_reason: Optional[str] = None) -> str | dict:
        if not return_metadata:
            return text
        return {"text": text, "finish_reason": finish_reason}

    # ------------------ Local / vLLM --------------------
    if server_type in {"local", "vllm"}:
        assert isinstance(prompt, str), "Only string prompt supported for local/vllm model"
        if return_metadata:
            from llm_local import (
                llm_streaming_enabled,
                max_continuation_rounds,
                max_token_continue_enabled,
                openai_chat_completion_with_truncation_retry,
            )

            extra_body = None
            if is_reasoning_model:
                think_obj = {"type": "enabled"}
                if int(budget_tokens or 0) > 0:
                    think_obj["budget_tokens"] = int(budget_tokens)
                extra_body = {"thinking": think_obj}

            _dp = (stream_dump_path or "").strip() or None
            max_cont = 0 if not max_token_continue_enabled() else max_continuation_rounds()
            text, finish_reason, dumped_to_file = openai_chat_completion_with_truncation_retry(
                llm.client,
                model=model,
                system_prompt=system_prompt,
                original_user=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=None,
                extra_body=extra_body,
                use_stream=llm_streaming_enabled(),
                dump_path=_dp,
                max_continuations=max_cont,
                round_idx=round_idx,
                repetition_penalty=repetition_penalty,
                max_context_length=int(max_context_length or 0),
            )
            return _qs_ret(_pack(text, finish_reason), dumped_to_file)

        _dump = (stream_dump_path or "").strip() or None
        cfg = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            enable_thinking=bool(is_reasoning_model),
            thinking_budget_tokens=int(budget_tokens or 0),
            stream_dump_path=_dump,
            round_idx=round_idx,
            max_context_length=int(max_context_length or 0),
        )

        output, dumped_to_file = llm.chat(
            system_prompt,
            prompt,
            cfg,
        )
        return _qs_ret(_pack(output, None), dumped_to_file)

    # ------------------ File queue ---------------------
    if server_type == "file_queue":
        # Mirror the "cloud APIs" interface: accept string prompt or messages list.
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = prompt
        # Convert to the anthropic-style expected by the server: system separate, messages without system role
        sys_txt = system_prompt
        msg_payload = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role == "system":
                # prefer explicit system_prompt argument; keep last system if present
                if isinstance(content, str) and content.strip():
                    sys_txt = content
                continue
            msg_payload.append({"role": role, "content": content})
        content = _file_queue_call(msg_payload, sys_txt)
        text = content[0] if isinstance(content, list) and content else str(content or "")
        return _qs_ret(_pack(text, None), False)

    # ------------------ Cloud APIs ---------------------
    outputs = []

    if server_type == "google":
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=generation_config,
        )
        response = model.generate_content(prompt)

        # Usage logging
        usage_metadata = getattr(response, 'usage_metadata', None)
        if usage_metadata:
            input_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
            total_tokens = getattr(usage_metadata, 'total_token_count', 0)
            usage_str = f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}"
            print(usage_str, file=sys.stderr, flush=True)
            if log_path and log_path != "":
                try:
                    import os as _os
                    import datetime as _datetime
                    file_exists = _os.path.exists(log_path)
                    with open(log_path, "a", encoding="utf-8") as f:
                        if not file_exists:
                            f.write("timestamp,round_idx,call_type,input_tokens,output_tokens,total_tokens\n")
                        timestamp = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{timestamp},{round_idx},{call_type},{input_tokens},{output_tokens},{total_tokens}\n")
                except Exception as e:
                    print(f"Warning: Failed to write usage log to {log_path}: {e}", file=sys.stderr, flush=True)

        # Finish reason
        try:
            candidates = getattr(response, 'candidates', [])
            if candidates:
                candidate = candidates[0]
                finish_reason_obj = getattr(candidate, 'finish_reason', None)
                finish_reason = getattr(finish_reason_obj, 'name', str(finish_reason_obj))
                print(colorize_finish_reason(finish_reason), file=sys.stderr, flush=True)
                if finish_reason in {"MAX_TOKENS", "length", "max_tokens"}:
                    print(
                        f"Warning: Output truncated due to max_tokens limit ({max_tokens})",
                        file=sys.stderr,
                        flush=True,
                    )
        except Exception:
            pass

        return _qs_ret(_pack(response.text, finish_reason if 'finish_reason' in locals() else None), False)

    elif server_type == "anthropic":
        assert isinstance(prompt, str)
        if is_reasoning_model:
            response = client.beta.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                thinking={"type": "enabled", "budget_tokens": budget_tokens},
                betas=["output-128k-2025-02-19"],
            )
        else:
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
        # Usage Logging
        if hasattr(response, 'usage'):
            input_tokens = getattr(response.usage, "input_tokens", None)
            output_tokens = getattr(response.usage, "output_tokens", None)
            total_tokens = getattr(response.usage, 'total_tokens', input_tokens + output_tokens
                                   if input_tokens is not None and output_tokens is not None
                                   else input_tokens if output_tokens is None
                                   else output_tokens if input_tokens is None
                                   else None)
            usage_str = f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}"
            print(usage_str, file=sys.stderr, flush=True)
            if log_path and log_path != "":
                try:
                    import os as _os
                    import datetime as _datetime
                    file_exists = _os.path.exists(log_path)
                    with open(log_path, "a", encoding="utf-8") as f:
                        if not file_exists:
                            f.write("timestamp,round_idx,call_type,input_tokens,output_tokens,total_tokens\n")
                        timestamp = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{timestamp},{round_idx},{call_type},{input_tokens},{output_tokens},{total_tokens}\n")
                except Exception as e:
                    print(f"Warning: Failed to write usage log to {log_path}: {e}", file=sys.stderr, flush=True)

        outputs = []
        finish_reason: Optional[str] = None
        for block in response.content:
            text = getattr(block, "text", None)
            if text is not None:
                outputs.append(text)
                continue

            block_type = getattr(block, "type", "unknown")
            block_name = getattr(block, "name", "")
            extra = f" ({block_name})" if block_name else ""
            print(
                f"Skipping non-text {server_type} content block of type '{block_type}'{extra}",
                file=sys.stderr,
                flush=True,
            )

        finish_reason = getattr(response, "stop_reason", None)
        print(colorize_finish_reason(finish_reason), file=sys.stderr, flush=True)
        if finish_reason in {"length", "max_tokens"}:
            print(
                f"Warning: Output truncated due to max_tokens limit ({max_tokens})",
                file=sys.stderr,
                flush=True,
            )

    else:
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = prompt

        if is_reasoning_model and server_type == "openai":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        outputs = []
        for i, choice in enumerate(response.choices):
            print(
                colorize_finish_reason(choice.finish_reason),
                file=sys.stderr,
                flush=True,
            )
            if i == 0:
                finish_reason = str(choice.finish_reason) if choice.finish_reason is not None else None

            if choice.finish_reason == "length":
                print(
                    f"Warning: Output truncated due to max_tokens limit ({max_tokens})",
                    file=sys.stderr,
                    flush=True,
                )
            outputs.append(choice.message.content)

        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", getattr(response.usage, "input_tokens", 0))
            output_tokens = getattr(response.usage, "completion_tokens", getattr(response.usage, "output_tokens", 0))
            total_tokens = getattr(response.usage, 'total_tokens', input_tokens + output_tokens)

            usage_str = f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}"
            print(usage_str, file=sys.stderr, flush=True)
            if log_path and log_path != "":
                try:
                    import os as _os
                    import datetime as _datetime
                    file_exists = _os.path.exists(log_path)
                    with open(log_path, "a", encoding="utf-8") as f:
                        if not file_exists:
                            f.write("timestamp,round_idx,call_type,input_tokens,output_tokens,total_tokens\n")
                        timestamp = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{timestamp},{round_idx},{call_type},{input_tokens},{output_tokens},{total_tokens}\n")
                except Exception as e:
                    print(f"Warning: Failed to write usage log to {log_path}: {e}", file=sys.stderr, flush=True)

    if len(outputs) == 1:
        return _qs_ret(_pack(outputs[0], finish_reason), False)
    if not return_metadata:
        return _qs_ret(outputs, False)
    first = outputs[0] if outputs else ""
    return _qs_ret({"text": first, "finish_reason": finish_reason}, False)
