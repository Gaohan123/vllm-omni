"""Offline inference demo for Qwen3 TTS via vLLM Omni.

Provides single and batch sample inputs for CustomVoice, VoiceDesign, and Base
tasks, then runs Omni generation and saves output wav files.
"""

import os
from typing import NamedTuple

import soundfile as sf

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import Omni


class QueryResult(NamedTuple):
    """Container for a prepared Omni request."""

    inputs: dict
    model_name: str


def get_custom_voice_query(use_batch_sample: bool = False) -> QueryResult:
    """Build CustomVoice sample inputs.

    Args:
        use_batch_sample: When True, return a batch of prompts; otherwise a single prompt.

    Returns:
        QueryResult with Omni inputs and the CustomVoice model path.
    """
    task_type = "CustomVoice"
    if use_batch_sample:
        texts = [
            "成为科学家，是无数中国孩子的梦想。",
            "你又不认得我们这么久的朋友，你感觉不到。",
            "呜呜，这些年代受的苦，就跟你说上十天半个月也说不完。",
        ]
        instructs = [
            "吐字清晰精准，字正腔圆",
            "",
            "哭腔导致发音略微含混，略显沙哑，带有明显哭腔的紧张感",
        ]
        languages = ["Auto", "Chinese", "Chinese"]
        speakers = ["uncle_fu", "vivian", "eric"]
        inputs = []
        for text, instruct, language, speaker in zip(texts, instructs, languages, speakers):
            prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
            inputs.append(
                {
                    "prompt": prompt,
                    "additional_information": {
                        "task_type": [task_type],
                        "text": [text],
                        "instruct": [instruct],
                        "language": [language],
                        "speaker": [speaker],
                        "max_new_tokens": [2048],
                    },
                }
            )
    else:
        text = "成为科学家，是无数中国孩子的梦想。"
        language = "Auto"
        speaker = "uncle_fu"
        prompts = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = {
            "prompt": prompts,
            "additional_information": {
                "task_type": [task_type],
                "text": [text],
                "language": [language],
                "speaker": [speaker],
                "max_new_tokens": [2048],
            },
        }
    return QueryResult(
        inputs=inputs,
        model_name="/home/dyvm6xra/dyvm6xrauser08/.cache/huggingface/hub/Qwen3-TTS-12Hz-1.7B-CustomVoice/",
    )


def get_voice_design_query(use_batch_sample: bool = False) -> QueryResult:
    """Build VoiceDesign sample inputs.

    Args:
        use_batch_sample: When True, return a batch of prompts; otherwise a single prompt.

    Returns:
        QueryResult with Omni inputs and the VoiceDesign model path.
    """
    task_type = "VoiceDesign"
    if use_batch_sample:
        texts = [
            "我们全船全军都该换换装备了。",
            "我叫通义千问。你叫什么名字呀",
            "I am Qwen. What is your name? Can you tell me a story?",
        ]
        instructs = [
            "以洪亮有力的音量发声,语速偏快,结尾略微放慢。",
            "用低沉有磁性的男性嗓音。",
            "用小女孩的声音。",
        ]
        languages = ["Chinese", "Auto", "English"]
        inputs = []
        for text, instruct, language in zip(texts, instructs, languages):
            prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
            inputs.append(
                {
                    "prompt": prompt,
                    "additional_information": {
                        "task_type": [task_type],
                        "text": [text],
                        "language": [language],
                        "instruct": [instruct],
                        "max_new_tokens": [2048],
                        "non_streaming_mode": [True],
                    },
                }
            )
    else:
        text = "我们全船全军都该换换装备了。"
        instruct = "以洪亮有力的音量发声,展示出男性特有的坚韧与威严感。语速偏快,语调从头至尾保持流畅,结尾略微放慢。"
        language = "Chinese"
        prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = {
            "prompt": prompt,
            "additional_information": {
                "task_type": [task_type],
                "text": [text],
                "language": [language],
                "instruct": [instruct],
                "max_new_tokens": [2048],
                "non_streaming_mode": [True],
            },
        }
    return QueryResult(
        inputs=inputs,
        model_name="/home/dyvm6xra/dyvm6xrauser08/.cache/huggingface/hub/Qwen3-TTS-12Hz-1.7B-VoiceDesign/",
    )


def get_base_query(use_batch_sample: bool = False, mode_tag: str = "icl") -> QueryResult:
    """Build Base (voice clone) sample inputs.

    Args:
        use_batch_sample: When True, return a batch of prompts (Case 2).
        mode_tag: "icl" or "xvec_only" to control x_vector_only_mode behavior.

    Returns:
        QueryResult with Omni inputs and the Base model path.
    """
    task_type = "Base"
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_zh.wav"
    ref_audio_single = ref_audio_path_1
    ref_text_single = "甚至出现交易几乎停滞的情况。"
    syn_text_single = "安安上课爱捣乱，一捣乱四邻不安，安老师劝安安莫捣乱，安安心上课不捣乱。"
    syn_lang_single = "Auto"
    x_vector_only_mode = mode_tag == "xvec_only"
    if use_batch_sample:
        syn_text_batch = [
            "安安上课爱捣乱，一捣乱四邻不安，安老师劝安安莫捣乱，安安心上课不捣乱。",
            "I am Qwen. What is your name? Can you tell me a story?",
        ]
        syn_lang_batch = ["Chinese", "English"]
        inputs = []
        for text, language in zip(syn_text_batch, syn_lang_batch):
            prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
            inputs.append(
                {
                    "prompt": prompt,
                    "additional_information": {
                        "task_type": [task_type],
                        "ref_audio": [ref_audio_single],
                        "ref_text": [ref_text_single],
                        "text": [text],
                        "language": [language],
                        "x_vector_only_mode": [x_vector_only_mode],
                        "max_new_tokens": [2048],
                    },
                }
            )
    else:
        prompt = f"<|im_start|>assistant\n{syn_text_single}<|im_end|>\n<|im_start|>assistant\n"
        inputs = {
            "prompt": prompt,
            "additional_information": {
                "task_type": [task_type],
                "ref_audio": [ref_audio_single],
                "ref_text": [ref_text_single],
                "text": [syn_text_single],
                "language": [syn_lang_single],
                "x_vector_only_mode": [x_vector_only_mode],
                "max_new_tokens": [2048],
            },
        }
    return QueryResult(
        inputs=inputs,
        model_name="/home/dyvm6xra/dyvm6xrauser08/.cache/huggingface/hub/Qwen3-TTS-12Hz-0.6B-Base/",
    )


def main(args):
    """Run offline inference with Omni using prepared sample inputs.

    Args:
        args: Parsed CLI args from parse_args().
    """
    query_func = query_map[args.query_type]
    if args.query_type in {"CustomVoice", "VoiceDesign"}:
        query_result = query_func(use_batch_sample=args.use_batch_sample)
    elif args.query_type == "Base":
        query_result = query_func(
            use_batch_sample=args.use_batch_sample,
            mode_tag=args.mode_tag,
        )
    else:
        query_result = query_func()

    model_name = query_result.model_name
    omni = Omni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.enable_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=2048,
        seed=42,
        detokenize=False,
        repetition_penalty=1.05,
    )

    sampling_params_list = [
        sampling_params,
    ]

    output_dir = args.output_dir if getattr(args, "output_dir", None) else args.output_wav
    os.makedirs(output_dir, exist_ok=True)

    omni_generator = omni.generate(query_result.inputs, sampling_params_list)
    for stage_outputs in omni_generator:
        for output in stage_outputs.request_output:
            request_id = output.request_id
            audio_tensor = output.multimodal_output["audio"]
            output_wav = os.path.join(output_dir, f"output_{request_id}.wav")
            audio_samplerate = output.multimodal_output["sr"].item()
            # Convert to numpy array and ensure correct format
            audio_numpy = audio_tensor.float().detach().cpu().numpy()

            # Ensure audio is 1D (flatten if needed)
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.flatten()

            # Save audio file with explicit WAV format
            sf.write(output_wav, audio_numpy, samplerate=audio_samplerate, format="WAV")
            print(f"Request ID: {request_id}, Saved audio to {output_wav}")


def parse_args():
    """Parse CLI arguments for offline TTS inference.

    Returns:
        argparse.Namespace with CLI options.
    """
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="CustomVoice",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        default=False,
        help="Enable writing detailed statistics (default: disabled)",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing a single stage in seconds (default: 300)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=5,
        help="Timeout for batching in seconds (default: 5)",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing stages in seconds (default: 300)",
    )
    parser.add_argument(
        "--shm-threshold-bytes",
        type=int,
        default=65536,
        help="Threshold for using shared memory in bytes (default: 65536)",
    )
    parser.add_argument(
        "--output-wav",
        default="output_audio",
        help="[Deprecated] Output wav directory (use --output-dir).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line (preferred).",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to a stage configs file.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file. If not provided, uses default audio asset.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate for audio loading (default: 16000).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory (default: logs).",
    )
    parser.add_argument(
        "--py-generator",
        action="store_true",
        default=False,
        help="Use py_generator mode. The returned type of Omni.generate() is a Python Generator object.",
    )
    parser.add_argument(
        "--use-batch-sample",
        action="store_true",
        default=False,
        help="Use batch input sample for CustomVoice/VoiceDesign/Base query.",
    )
    parser.add_argument(
        "--mode-tag",
        type=str,
        default="icl",
        choices=["icl", "xvec_only"],
        help="Mode tag for Base query x_vector_only_mode (default: icl).",
    )

    return parser.parse_args()


query_map = {
    "CustomVoice": get_custom_voice_query,
    "VoiceDesign": get_voice_design_query,
    "Base": get_base_query,
}


if __name__ == "__main__":
    args = parse_args()
    main(args)
