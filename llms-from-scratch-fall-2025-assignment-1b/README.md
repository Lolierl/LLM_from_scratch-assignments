## LLMs from Scratch Assignment 1B

#### Environments

- Configure the Python environment by running `pip install -r requirements.txt`.

#### Testing

- To run all tests, use `pytest`.
- To run a specific test script, use `pytest src/test_{your_test_name}.py`. Make sure to run this command from the project root directory (do not cd into `src/`).
- To run a specific test function, use `pytest -k test_{your_test_name}`. Don't include `src/` or `.py` here!
- `pytest` will capture all outputs by default when the test succeeds. To disable output capturing (and see your debug outputs), use the flag `-s`.

#### Submission

- Submit a ZIP file `{your_name}_{your_student_id}.zip`, for example, `xinghan_li_2022012747.zip`. Your submission should contain the following files:

```
{your_name}_{your_student_id}.zip/
├── triton_tutorial.ipynb
├── triton_challenges.py
└── writeup.pdf (your answers to written problems)
```

If you have implemented Stick-Breaking Attention (SBA), submit your code and results to us by email. Check: https://git.tsinghua.edu.cn/luokr24/stick_breaking_attention_leaderboard/
