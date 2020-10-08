import argparse
import logging
import numpy as np
import torch
import os
from transformers import CTRLLMHeadModel, CTRLTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 1.0 has no effect, lower tend toward greedy sampling",)
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=10, help="The number of samples to generate.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    logger.warning(
        "device: %s, n_gpu: %s",
        args.device,
        args.n_gpu,
    )
    set_seed(args)

    # Initialize the model and tokenizer
    tokenizer = CTRLTokenizer.from_pretrained('ctrl', cache_dir='/data1/zhangsy/_cache/torch/transformers/ctrl')
    model = CTRLLMHeadModel.from_pretrained('ctrl', cache_dir='/data1/zhangsy/_cache/torch/transformers/ctrl')
    model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    # control code and prompt
    with open('control_codes', 'r', encoding='utf8') as f:
        control_codes = f.readlines()
        control_codes = [_.strip() for _ in control_codes]
    with open('prompts', 'r', encoding='utf8') as f:
        prompts = f.readlines()
        prompts = [_.strip() for _ in prompts]

    os.makedirs('generate', exist_ok=True)
    for control_code in control_codes:
        with open('generate/' + control_code, 'w', encoding='utf8') as f:
            for prompt in prompts:
                prompt_text = control_code + ' ' + prompt
                preprocessed_prompt_text = prepare_ctrl_input(args, model, tokenizer, prompt_text)
                encoded_prompt = tokenizer.encode(preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt",)
                encoded_prompt = encoded_prompt.to(args.device)

                if encoded_prompt.size()[-1] == 0:
                    input_ids = None
                else:
                    input_ids = encoded_prompt

                output_sequences = model.generate(
                    input_ids=input_ids,
                    max_length=args.length + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                )

                # Remove the batch dimension when returning multiple sequences
                if len(output_sequences.shape) > 2:
                    output_sequences.squeeze_()

                generated_sequences = []

                for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                    f.write("\n=== GENERATED SEQUENCE ===\n")
                    generated_sequence = generated_sequence.tolist()
                    # Decode text
                    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                    # Remove all text after the stop token
                    text = text[: text.find(args.stop_token) if args.stop_token else None]
                    # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                    total_sequence = (
                        prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
                    )
                    generated_sequences.append(total_sequence)
                    f.write(total_sequence)


if __name__ == "__main__":
    main()