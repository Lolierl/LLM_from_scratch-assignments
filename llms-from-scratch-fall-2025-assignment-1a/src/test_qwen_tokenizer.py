import pytest
from .common import FIXTURES_PATH

from .qwen_tokenizer import (
    encode_file_with_qwen_tokenizer,
    decode_sequence_with_qwen_tokenizer,
)


@pytest.fixture(scope="module")
def setup_test_environment():
    """
    Creates a temporary directory and a sample input file for testing.
    This fixture is run once per module.
    """
    input_file_path = FIXTURES_PATH / "input.txt"
    sample_text = "Hello world! This is the Qwen tokenizer."

    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    return {"path": input_file_path, "text": sample_text}


def test_encode_decode_roundtrip(setup_test_environment):
    """
    Tests the full encode -> decode pipeline.
    Ensures that decoding the encoded tokens restores the original text exactly.
    """
    input_path = setup_test_environment["path"]
    original_text = setup_test_environment["text"]

    encoded_ids = encode_file_with_qwen_tokenizer(input_path)
    
    assert isinstance(encoded_ids, list), "encode function should return a list."
    assert all(isinstance(i, int) for i in encoded_ids), "All items in the encoded list should be integers."

    decoded_text = decode_sequence_with_qwen_tokenizer(encoded_ids)

    assert decoded_text == original_text, "Decoded text does not match the original text."
