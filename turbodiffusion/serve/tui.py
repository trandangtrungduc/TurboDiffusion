"""
TurboDiffusion TUI Server Mode

A persistent GPU server that loads models once and provides an interactive
text-based interface for video generation.

Supports both T2V (text-to-video) and I2V (image-to-video) modes.

Usage:
    # T2V mode
    PYTHONPATH=turbodiffusion python -m serve --mode t2v --dit_path checkpoints/model.pth

    # I2V mode
    PYTHONPATH=turbodiffusion python -m serve --mode i2v \
        --high_noise_model_path checkpoints/high.pth \
        --low_noise_model_path checkpoints/low.pth
"""

import argparse
import os

from imaginaire.utils import log

from .utils import RUNTIME_PARAMS, validate_args, format_config, set_runtime_param
from .pipeline import load_models, generate_t2v, generate_i2v


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for TUI server mode."""
    parser = argparse.ArgumentParser(
        description="TurboDiffusion TUI Server - Interactive video generation"
    )

    parser.add_argument("--mode", choices=["t2v", "i2v"], default="t2v",
                        help="Generation mode: t2v (text-to-video) or i2v (image-to-video)")

    # T2V model path
    parser.add_argument("--dit_path", type=str, default=None,
                        help="Path to DiT checkpoint (required for t2v mode)")

    # I2V model paths
    parser.add_argument("--high_noise_model_path", type=str, default=None,
                        help="Path to high-noise model (required for i2v mode)")
    parser.add_argument("--low_noise_model_path", type=str, default=None,
                        help="Path to low-noise model (required for i2v mode)")
    parser.add_argument("--boundary", type=float, default=0.9,
                        help="Timestep boundary for model switching (i2v only)")

    # Model configuration
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B", "Wan2.2-A14B"],
                        default=None, help="Model architecture (auto-detected from mode if not set)")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth",
                        help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str,
                        default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to the umT5 text encoder")

    # Resolution
    parser.add_argument("--resolution", default=None, type=str,
                        help="Resolution (default: 480p for t2v, 720p for i2v)")
    parser.add_argument("--aspect_ratio", default="16:9", type=str,
                        help="Aspect ratio (width:height)")
    parser.add_argument("--adaptive_resolution", action="store_true",
                        help="Adapt resolution to input image aspect ratio (i2v only)")

    # Attention/quantization
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"],
                        default="sagesla", help="Attention mechanism type")
    parser.add_argument("--sla_topk", type=float, default=0.1,
                        help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true",
                        help="Use quantized linear layers")
    parser.add_argument("--default_norm", action="store_true",
                        help="Use default LayerNorm/RMSNorm (not optimized)")

    # Sampling options
    parser.add_argument("--ode", action="store_true",
                        help="Use ODE sampling (sharper but less robust, i2v only)")

    # Runtime-adjustable parameters
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4,
                        help="Number of inference steps (1-4)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames to generate")
    parser.add_argument("--sigma_max", type=float, default=None,
                        help="Initial sigma (default: 80 for t2v, 200 for i2v)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def get_multiline_prompt() -> str:
    """Read multi-line prompt from user. Empty line finishes input."""
    lines = []
    while True:
        try:
            line = input("> " if lines else "")
            if line == "":
                break
            lines.append(line)
        except EOFError:
            if not lines:
                return None
            break
    return "\n".join(lines)


def print_help():
    """Print help for slash commands."""
    print("""
Commands:
  /help              Show this help message
  /show              Show current configuration
  /set <param> <val> Set a runtime parameter
  /reset             Reset runtime parameters to defaults
  /quit              Exit the server

Runtime parameters (adjustable with /set):
  num_steps   Number of inference steps (1-4)
  num_samples Number of samples per generation
  num_frames  Number of video frames
  sigma_max   Initial sigma for rCM
""")


def print_config(args: argparse.Namespace, defaults: dict):
    """Print current configuration."""
    print(format_config(args, defaults))


def handle_command(cmd: str, args: argparse.Namespace, defaults: dict) -> bool:
    """Handle slash command. Returns False if should quit."""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command == "/quit":
        return False
    elif command == "/help":
        print_help()
    elif command == "/show":
        print_config(args, defaults)
    elif command == "/set":
        if len(parts) != 3:
            print("Usage: /set <param> <value>")
        else:
            success, msg = set_runtime_param(args, parts[1], parts[2])
            print(msg if success else f"Error: {msg}")
    elif command == "/reset":
        for param, default in defaults.items():
            setattr(args, param, default)
        print("Runtime parameters reset to defaults.")
    else:
        print(f"Unknown command: {command}")
        print("Type /help for available commands.")

    return True


def print_header(args: argparse.Namespace):
    """Print server header with current config."""
    from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    mode_str = "T2V (text-to-video)" if args.mode == "t2v" else "I2V (image-to-video)"
    print(f"""
TurboDiffusion TUI Server
=========================
Mode: {mode_str} | Model: {args.model}
Resolution: {args.resolution} ({w}x{h}) | Steps: {args.num_steps}
Type /help for commands, or enter a prompt to generate.
""")


def run_tui(models: dict, args: argparse.Namespace):
    """Main TUI loop."""
    defaults = {param: getattr(args, param) for param in RUNTIME_PARAMS}
    last_output_path = "output/generated_video.mp4"
    last_image_path = None

    print_header(args)

    while True:
        print("Prompt (empty line to generate):")
        prompt = get_multiline_prompt()

        if prompt is None:
            print("\nGoodbye!")
            break

        prompt = prompt.strip()

        if not prompt:
            continue

        if prompt.startswith("/"):
            if not handle_command(prompt, args, defaults):
                print("Goodbye!")
                break
            continue

        # For I2V mode, get image path
        image_path = None
        if args.mode == "i2v":
            try:
                default_hint = f" [{last_image_path}]" if last_image_path else ""
                user_image = input(f"Image path{default_hint}: ").strip()
                if not user_image and last_image_path:
                    image_path = last_image_path
                elif user_image:
                    image_path = user_image
                else:
                    print("Error: Image path is required for I2V mode.")
                    continue

                if not os.path.isfile(image_path):
                    print(f"Error: Image file not found: {image_path}")
                    continue

                last_image_path = image_path
            except EOFError:
                print("\nGoodbye!")
                break

        # Get output path
        try:
            user_path = input(f"Output path [{last_output_path}]: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break

        output_path = user_path if user_path else last_output_path

        if not output_path.endswith(".mp4"):
            output_path += ".mp4"

        # Generate
        try:
            if args.mode == "t2v":
                result_path = generate_t2v(models, args, prompt, output_path)
            else:
                result_path = generate_i2v(models, args, prompt, image_path, output_path)

            log.success(f"Generated: {result_path}")
            last_output_path = result_path
        except Exception as e:
            log.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()

        print()


def main(passed_args: argparse.Namespace = None):
    """Main entry point for TUI server."""
    args = passed_args if passed_args is not None else parse_arguments()

    validate_args(args)

    models = load_models(args)

    try:
        run_tui(models, args)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")


if __name__ == "__main__":
    main()
