import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort


PAD_ID = 0
UNK_TOKEN = "<UNK>"
DEFAULT_UNK_ID = 1
MODEL_CONTEXT_LEN = 550
CHUNK_LIMIT = 500


class OnnxArabicDiacritizer:
    def __init__(self, model_path=None, vocab_path=None, output_vocab_path=None, providers=None, pad_to_context=True):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(base_dir, "diacritizer.onnx")
        vocab_path = vocab_path or os.path.join(base_dir, "input_vocab_to_int.json")
        output_vocab_path = output_vocab_path or os.path.join(base_dir, "output_int_to_vocab.json")
        self.pad_to_context = pad_to_context

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        with open(output_vocab_path, "r", encoding="utf-8") as f:
            output_vocab = json.load(f)

        self.class_map = {int(k): v for k, v in output_vocab.items()}
        self.class_map.setdefault(3, "")
        self.class_map.setdefault(17, "\u064b")

        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = "inputs"
        self.output_name = "outputs"

    def _indices_for_chunk(self, text):
        unk_id = self.vocab.get(UNK_TOKEN, DEFAULT_UNK_ID)
        indices = [self.vocab.get(char, unk_id) for char in text]
        if self.pad_to_context:
            padded_len = max(MODEL_CONTEXT_LEN, len(indices))
            indices.extend([PAD_ID] * (padded_len - len(indices)))
        return np.asarray(indices, dtype=np.int64).reshape(1, -1)

    def diacritize_chunk(self, text):
        if not text:
            return ""

        inputs = self._indices_for_chunk(text)
        logits = self.session.run([self.output_name], {self.input_name: inputs})[0][0, : len(text)]
        pred_classes = np.argmax(logits, axis=-1)

        result = []
        for char, pred_class in zip(text, pred_classes):
            diacritic = self.class_map.get(int(pred_class), "")
            if diacritic in ("<PAD>", "<UNK>"):
                diacritic = ""
            result.append(char)
            result.append(diacritic)

        return "".join(result)

    def _split_line(self, line):
        if len(line) <= CHUNK_LIMIT:
            return [line]

        chunks = []
        current_chunk = []
        current_len = 0
        for word in line.split(" "):
            if current_chunk and current_len + len(word) + 1 > CHUNK_LIMIT:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = len(word)
            else:
                current_chunk.append(word)
                current_len += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def diacritize(self, text):
        diacritized_lines = []
        for line in text.split("\n"):
            chunks = self._split_line(line)
            diacritized_lines.append(" ".join(self.diacritize_chunk(chunk) for chunk in chunks))
        return "\n".join(diacritized_lines)


def read_text(args):
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            return f.read()
    if args.text is not None:
        return args.text
    return sys.stdin.read()


def main():
    parser = argparse.ArgumentParser(description="Run Arabic diacritization with ONNX Runtime.")
    parser.add_argument("text", nargs="?", help="Text to diacritize. Reads stdin if omitted.")
    parser.add_argument("--input-file", help="UTF-8 text file to diacritize.")
    parser.add_argument("--output-file", help="Write UTF-8 output to this file.")
    parser.add_argument("--model", help="Path to diacritizer.onnx.")
    parser.add_argument("--vocab", help="Path to input_vocab_to_int.json.")
    parser.add_argument("--output-vocab", help="Path to output_int_to_vocab.json.")
    parser.add_argument("--no-padding", action="store_true", help="Run ONNX on the true chunk length.")
    args = parser.parse_args()

    diacritizer = OnnxArabicDiacritizer(
        model_path=args.model,
        vocab_path=args.vocab,
        output_vocab_path=args.output_vocab,
        pad_to_context=not args.no_padding,
    )
    output = diacritizer.diacritize(read_text(args))

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
