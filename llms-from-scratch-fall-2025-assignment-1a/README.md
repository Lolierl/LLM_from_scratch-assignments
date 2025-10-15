## LLMs from Scratch Assignment 1A

#### Environments

- Configure the environment using `conda env create -f environment.yml`.
- Activate the environment: `conda activate llms_from_scratch_1a`.

#### Testing

- To run all tests, use `pytest`.
- To run a specific test script, use `pytest src/test_{your_test_name}.py`. Make sure to run this command from the project root directory (do not cd into src/).
- To run a specific test function, use `pytest -k test_{your_test_name}`. Don't include `src/` or `.py` here!
- `pytest` will capture all outputs by default when the test succeeds. To disable output capturing (and see your debug outputs), use the flag `-s`.

#### Data

- All data needed for the homework is provided in `src/fixtures`, except for the model weights of Qwen3-0.6B.
- The test script `src/test_qwen3_generate.py` will automatically download these weights for you. **Make sure to exclude the model weights when submitting your homework**.
- If the test script fails to download the weights, please manually download them from https://cloud.tsinghua.edu.cn/f/9ae7ca9806254b42bf4a/?dl=1 and place the file at `src/fixtures/qwen3_0.6b_weights/qwen3-0.6B.safetensors`.

#### Submission

- Compress your homework directory into `{your_name}_{your_student_id}.zip`, for example, `xinghan_li_2022012747.zip`. The submission should be organized in the following format:

```
{your_name}_{your_student_id}.zip/
├── src/
    ├── ... (all your implementation files but
    └── exclude fixtures/qwen3_0.6b_weights)
├── environment.yml
├── NPT_merges.txt (your result for Problem 1.6)
├── NPT_vocab.json (your result for Problem 1.6)
├── writeup.pdf (your answers to written problems)
├── qwen3_generate.py
└── README.md
```